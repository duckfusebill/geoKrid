"""
Head-to-head comparison: geoKrid vs GeoCLIP on the US validation set.

Metrics computed per image:
  - Haversine distance error (km) for geoKrid top-1
  - Haversine distance error (km) for geoKrid consensus (geometric median, top-10)
  - Haversine distance error (km) for GeoCLIP top-1
  - Haversine distance error (km) for GeoCLIP consensus (weighted top-5)

Aggregate: mean, median, MSE (km²), RMSE (km), acc@25/100/200/750km

Output:
  report/comparison.txt   — aggregate table
  report/comparison.csv   — per-image results
"""

import os, glob, math, io
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader

import dataset as D
import model as M

# ── config ──────────────────────────────────────────────────────────────────
CELLS_CSV  = os.environ.get("CELLS_CSV",  "cells.csv")
CKPT_DIR   = os.environ.get("CKPT_DIR",   "checkpoints_kmeans")
VAL_CSV    = os.environ.get("VAL_CSV",    "/mnt/b/datasets/geovit/us_val_cls.csv")
DEVICE     = os.environ.get("DEVICE",     "cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
TOP_K      = 10
TEMPERATURE = 1.8
SIGMA_KM    = 150.0
N_LIMIT     = int(os.environ.get("N_LIMIT", 0))   # 0 = full val set
OUT_DIR     = "report"


# ── haversine ────────────────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(np.array(lat2) - np.array(lat1))
    dlon = np.radians(np.array(lon2) - np.array(lon1))
    a = (np.sin(dlat/2)**2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2)
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ── geometric median (Weiszfeld) ─────────────────────────────────────────────
def geometric_median(coords, weights, eps=1e-5, max_iter=50):
    weights = np.array(weights, dtype=float)
    est = np.average(coords, weights=weights, axis=0)
    for _ in range(max_iter):
        dists = np.array([haversine(est[0], est[1], c[0], c[1]) for c in coords])
        dists = np.maximum(dists, eps)
        w = weights / dists
        new_est = np.average(coords, weights=w, axis=0)
        if np.linalg.norm(new_est - est) < eps:
            break
        est = new_est
    return est


# ── spatial re-weighting ──────────────────────────────────────────────────────
def spatial_reweight(coords, probs, sigma=SIGMA_KM):
    probs = np.array(probs)
    sw = []
    for i in range(len(coords)):
        score = sum(probs[j] * math.exp(
            -haversine(coords[i][0], coords[i][1], coords[j][0], coords[j][1]) / sigma)
            for j in range(len(coords)) if j != i)
        sw.append(probs[i] * (1.0 + score))
    total = sum(sw)
    return [w / total for w in sw]


# ── load geoKrid ─────────────────────────────────────────────────────────────
def load_geokrid():
    cells = pd.read_csv(CELLS_CSV)
    cell_centers = cells[["lat_center", "lon_center"]].values
    m = M.GeoClassifier(len(cells), cell_centers).to(DEVICE)
    ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "epoch_*.pt")))
    ckpt = torch.load(ckpts[-1], map_location=DEVICE, weights_only=True)
    m.load_state_dict(ckpt["model"])
    m.eval()
    print(f"geoKrid: loaded {ckpts[-1]}")
    return m


# ── load GeoCLIP ─────────────────────────────────────────────────────────────
class _UnfoldPatchEmbed(torch.nn.Module):
    """Same ROCm gfx1100 fix used in geoKrid's model.py, applied to GeoCLIP's CLIP."""
    def __init__(self, conv):
        super().__init__()
        p, d = conv.kernel_size[0], conv.out_channels
        w = conv.weight.data.reshape(d, -1)
        self.linear = torch.nn.Linear(w.shape[1], d, bias=False)
        self.linear.weight = torch.nn.Parameter(w)
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


def _patch_geoclip_encoder(gc):
    """
    Two fixes needed on ROCm with newer transformers:
    1. Replace Conv2d patch_embedding with unfold+linear (avoids gfx1100 SIGSEGV).
    2. get_image_features() returns BaseModelOutputWithPooling, not a tensor.
    """
    # fix 1: ROCm patch_embedding segfault
    emb = gc.image_encoder.CLIP.vision_model.embeddings
    emb.patch_embedding = _UnfoldPatchEmbed(emb.patch_embedding)

    # fix 2: transformers output type
    import types
    def _patched_forward(self, x):
        out = self.CLIP.get_image_features(pixel_values=x)
        if not isinstance(out, torch.Tensor):
            out = out.pooler_output
        return self.mlp(out)

    gc.image_encoder.forward = types.MethodType(_patched_forward, gc.image_encoder)
    return gc


def load_geoclip():
    from geoclip import GeoCLIP
    gc = GeoCLIP()
    _patch_geoclip_encoder(gc)
    gc.to(DEVICE)
    gc.eval()
    print("GeoCLIP: loaded (pretrained)")
    return gc


# ── GeoCLIP batch inference ───────────────────────────────────────────────────
def geoclip_predict_batch(gc, img_paths, top_k=5):
    """Returns (top1_coords, consensus_coords) each shape (B, 2)."""
    top1s, cons = [], []
    for path in tqdm(img_paths, desc="GeoCLIP"):
        try:
            coords, probs = gc.predict(path, top_k=top_k)
            coords_np = coords.cpu().numpy()
            probs_np  = probs.cpu().numpy()
            top1s.append(coords_np[0])
            rw = spatial_reweight(coords_np, probs_np)
            cons.append(geometric_median(coords_np, rw))
        except Exception:
            top1s.append(np.array([0.0, 0.0]))
            cons.append(np.array([0.0, 0.0]))
    return np.array(top1s), np.array(cons)


# ── metrics ───────────────────────────────────────────────────────────────────
def metrics(errors_km, name):
    e = np.array(errors_km)
    lines = [f"\n  {name}"]
    lines.append(f"    median {np.median(e):.2f} km  |  mean {np.mean(e):.2f} km")
    lines.append(f"    MSE  {np.mean(e**2):.1f} km²  |  RMSE {np.sqrt(np.mean(e**2)):.2f} km")
    for t in [25, 100, 200, 750]:
        lines.append(f"    acc@{t}km   {(e <= t).mean()*100:.2f}%")
    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────────────
def run():
    os.makedirs(OUT_DIR, exist_ok=True)

    geokrid   = load_geokrid()
    geoclip_m = load_geoclip()

    val_df = pd.read_csv(VAL_CSV)
    if N_LIMIT:
        val_df = val_df.sample(N_LIMIT, random_state=42).reset_index(drop=True)
        val_df.to_csv("/tmp/_sub.csv", index=False)

    val_csv_path = "/tmp/_sub.csv" if N_LIMIT else VAL_CSV
    transform = D.make_val_transform(size=336)
    val_ds = D.GeoValDataset(val_csv_path, "", transform)

    loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, collate_fn=D.collate_skip_none, pin_memory=True)

    rows = []
    img_paths = val_df["image_path"].tolist()

    # geoKrid pass
    gk_top1, gk_cons = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="geoKrid"):
            if batch is None:
                continue
            imgs, gps = batch
            imgs = imgs.to(DEVICE, non_blocking=True)
            logits = geokrid(imgs)
            probs  = (logits / TEMPERATURE).softmax(dim=-1)
            top    = torch.topk(probs, TOP_K, dim=-1)

            for i in range(len(gps)):
                idxs   = top.indices[i].tolist()
                pvals  = top.values[i].tolist()
                coords = geokrid.cell_centers[top.indices[i]].cpu().numpy()
                rw     = spatial_reweight(coords, pvals)
                gk_top1.append(geokrid.cell_centers[idxs[0]].cpu().numpy())
                gk_cons.append(geometric_median(coords, rw))

    gk_top1 = np.array(gk_top1)
    gk_cons = np.array(gk_cons)

    # GeoCLIP pass (image-by-image)
    gc_top1, gc_cons = geoclip_predict_batch(geoclip_m, img_paths, top_k=5)

    # actual coords
    actual_df = pd.read_csv(val_csv_path)
    actual_lat = actual_df["lat"].values
    actual_lon = actual_df["lon"].values
    n = min(len(actual_lat), len(gk_top1), len(gc_top1))

    gk_top1_err = haversine(actual_lat[:n], actual_lon[:n], gk_top1[:n,0], gk_top1[:n,1])
    gk_cons_err = haversine(actual_lat[:n], actual_lon[:n], gk_cons[:n,0], gk_cons[:n,1])
    gc_top1_err = haversine(actual_lat[:n], actual_lon[:n], gc_top1[:n,0], gc_top1[:n,1])
    gc_cons_err = haversine(actual_lat[:n], actual_lon[:n], gc_cons[:n,0], gc_cons[:n,1])

    # save per-image CSV
    pd.DataFrame({
        "actual_lat": actual_lat[:n], "actual_lon": actual_lon[:n],
        "gk_top1_lat": gk_top1[:n,0], "gk_top1_lon": gk_top1[:n,1],
        "gk_cons_lat": gk_cons[:n,0], "gk_cons_lon": gk_cons[:n,1],
        "gc_top1_lat": gc_top1[:n,0], "gc_top1_lon": gc_top1[:n,1],
        "gc_cons_lat": gc_cons[:n,0], "gc_cons_lon": gc_cons[:n,1],
        "gk_top1_km": gk_top1_err, "gk_cons_km": gk_cons_err,
        "gc_top1_km": gc_top1_err, "gc_cons_km": gc_cons_err,
    }).to_csv(f"{OUT_DIR}/comparison.csv", index=False)

    # summary
    lines = ["=" * 60,
             "geoKrid vs GeoCLIP — Error Analysis",
             f"n = {n:,} val images",
             "=" * 60]
    lines.append(metrics(gk_top1_err, "geoKrid    top-1"))
    lines.append(metrics(gk_cons_err, "geoKrid    consensus (geo-median, top-10, spatial reweight)"))
    lines.append(metrics(gc_top1_err, "GeoCLIP    top-1"))
    lines.append(metrics(gc_cons_err, "GeoCLIP    consensus (geo-median, top-5, spatial reweight)"))

    improvement_median = np.median(gc_top1_err) - np.median(gk_cons_err)
    improvement_mse    = np.mean(gc_top1_err**2) - np.mean(gk_cons_err**2)
    lines += ["", "=" * 60,
              f"geoKrid consensus vs GeoCLIP top-1:",
              f"  median improvement:  {improvement_median:+.1f} km",
              f"  MSE improvement:     {improvement_mse:+.1f} km²",
              "=" * 60]

    report = "\n".join(lines)
    print(report)
    with open(f"{OUT_DIR}/comparison.txt", "w") as f:
        f.write(report + "\n")
    print(f"\nsaved {OUT_DIR}/comparison.txt")


if __name__ == "__main__":
    run()
