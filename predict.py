import torch
from PIL import Image
import model as M

"""loading"""
def load_model(checkpoint_path, device="cuda", queue_size=M.QUEUE_SIZE, gps_gallery_file=None):
    m = M.GeoViT384(from_pretrained=False, queue_size=queue_size, gps_gallery_file=gps_gallery_file)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    m.load_state_dict(ckpt["model"])
    m.to(device)
    m.eval()
    return m


"""inference"""
def predict_image(model, image_path, top_k=5):
    # returns (coords [k,2] lat/lon, probs [k]) cpu tensors
    image = Image.open(image_path).convert("RGB")
    x = model.image_encoder.preprocess_image(image).to(model.device)
    gallery = model.gps_gallery.to(model.device)

    with torch.no_grad():
        logits = model(x, gallery)              # (1, gallery_size)
        probs = logits.softmax(dim=-1).cpu()
        top = torch.topk(probs, top_k, dim=-1)

    coords = model.gps_gallery[top.indices[0]]  # (k, 2)
    prob_vals = top.values[0]                   # (k,)
    return coords, prob_vals


"""display"""
def print_predictions(coords, probs):
    for i, (coord, prob) in enumerate(zip(coords, probs)):
        print(f"  {i+1}. lat={coord[0]:.5f}  lon={coord[1]:.5f}  p={prob:.5f}")


"""cli"""
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--gallery", default=None, help="path to us gps gallery csv")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    m = load_model(args.checkpoint, device=args.device, gps_gallery_file=args.gallery)
    coords, probs = predict_image(m, args.image_path, top_k=args.top_k)
    print_predictions(coords, probs)
