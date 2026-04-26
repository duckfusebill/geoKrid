import os
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image

"""bounds"""
US_LAT = (24.0, 49.0)
US_LON = (-125.0, -66.0)


"""transforms"""
def make_train_transform():
    return T.Compose([
        T.RandomResizedCrop(336, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def make_val_transform(size=336):
    return T.Compose([
        T.Resize(size),
        T.CenterCrop(size),
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


"""datasets"""
class GeoTrainDataset(Dataset):
    """Training dataset — returns (image_tensor, cell_id: long)."""

    def __init__(self, csv_path, root_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.root_dir, row["image_path"])
        cell_id = int(row["cell_id"])
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(cell_id, dtype=torch.long)
        except (IOError, OSError):
            return None


class GeoValDataset(Dataset):
    """Val dataset — returns (image_tensor, gps_tensor[lat, lon]) for distance eval."""

    def __init__(self, csv_path, root_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.root_dir, row["image_path"])
        lat, lon = float(row["lat"]), float(row["lon"])
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor([lat, lon], dtype=torch.float32)
        except (IOError, OSError):
            return None


"""collation"""
def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    a, b = zip(*batch)
    return torch.stack(a), torch.stack(b)


"""seeded sampler — deterministic per-epoch shuffle, supports batch skip for mid-epoch resume"""
class SeededSkipSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, epoch, skip_batches=0, batch_size=1):
        self.n = len(dataset)
        self.epoch = epoch
        self.skip = skip_batches * batch_size

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.randperm(self.n, generator=g).tolist()
        return iter(indices[self.skip:])

    def __len__(self):
        return max(0, self.n - self.skip)


"""loaders"""
def build_loaders(cfg, epoch=1, skip_batches=0):
    train_tf = make_train_transform()
    val_tf = make_val_transform(size=336)

    train_ds = GeoTrainDataset(cfg["train_csv"], cfg["osv_root"], train_tf)
    val_ds = GeoValDataset(cfg["val_csv"], cfg["osv_root"], val_tf)

    sampler = SeededSkipSampler(train_ds, epoch=epoch,
                                skip_batches=skip_batches,
                                batch_size=cfg["batch_size"])

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        sampler=sampler,
        num_workers=cfg["num_workers"],
        collate_fn=collate_skip_none,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        collate_fn=collate_skip_none,
        pin_memory=True,
    )
    return train_loader, val_loader
