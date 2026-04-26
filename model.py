import torch
import torch.nn as nn
import numpy as np
from transformers import CLIPModel, AutoProcessor


class _UnfoldPatchEmbed(nn.Module):
    # replaces CLIPVisionEmbeddings.patch_embedding Conv2d(3,1024,14,14)
    # ROCm 6.1 on gfx11000 segfaults on large-kernel Conv2d; unfold+linear avoids it
    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        p, d = conv.kernel_size[0], conv.out_channels
        w = conv.weight.data.reshape(d, -1)
        self.linear = nn.Linear(w.shape[1], d, bias=False)
        self.linear.weight = nn.Parameter(w)
        self.patch_size = p

    @property
    def weight(self):
        return self.linear.weight

    def forward(self, x):
        p = self.patch_size
        patches = x.unfold(2, p, p).unfold(3, p, p)
        B, C, nh, nw, _, _ = patches.shape
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(B, nh * nw, C * p * p)
        return self.linear(patches).permute(0, 2, 1).reshape(B, -1, nh, nw)


def _patch_clip_for_rocm(clip: CLIPModel) -> CLIPModel:
    emb = clip.vision_model.embeddings
    emb.patch_embedding = _UnfoldPatchEmbed(emb.patch_embedding)
    return clip


class CLIPEncoder(nn.Module):
    """Frozen CLIP ViT-L/14@336 image encoder — outputs 768-d pooled features."""

    def __init__(self):
        super().__init__()
        self.clip = _patch_clip_for_rocm(
            CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336")
        )
        self.image_processor = AutoProcessor.from_pretrained(
            "openai/clip-vit-large-patch14-336"
        )
        for p in self.clip.parameters():
            p.requires_grad = False

    def preprocess_image(self, image):
        return self.image_processor(images=image, return_tensors="pt")["pixel_values"]

    def forward(self, x):
        with torch.no_grad():
            out = self.clip.vision_model(pixel_values=x)
        return out.pooler_output  # (B, 768)


class GeoClassifier(nn.Module):
    """
    US geolocation as grid-cell classification.

    Frozen CLIP encodes the image to a 768-d feature vector.
    A small trainable MLP maps that to logits over N geographic cells.
    At inference, argmax → cell centroid gives the predicted (lat, lon).
    """

    def __init__(self, n_cells: int, cell_centers: np.ndarray):
        super().__init__()
        self.encoder = CLIPEncoder()
        self.head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, n_cells),
        )
        self.register_buffer(
            "cell_centers",
            torch.tensor(cell_centers, dtype=torch.float32),  # (N, 2) lat/lon
        )

    def forward(self, x):
        return self.head(self.encoder(x))

    def predict_coords(self, x):
        with torch.no_grad():
            idx = self(x).argmax(dim=-1)
        return self.cell_centers[idx]  # (B, 2)
