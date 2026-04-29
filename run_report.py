"""
Full validation report — runs all val images through the model and saves
per-image results + aggregate stats broken down by geographic region.

Output:
  report/predictions.csv   — per-image actual vs predicted + error
  report/summary.txt       — overall and regional metrics
"""

import os
import math
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

import dataset as D
import model as M

CELLS_CSV  = os.environ.get("CELLS_CSV",  "cells.csv")
CKPT_DIR   = os.environ.get("CKPT_DIR",   "checkpoints_kmeans")
VAL_CSV    = os.environ.get("VAL_CSV",    "/mnt/b/datasets/geovit/us_val_cls.csv")
DEVICE     = os.environ.get("DEVICE",     "cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
TOP_K      = 5
OUT_DIR    = "report"


def haversine_np(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def region(lat, lon):
    if lon < -100:
        return "West"
    if lat >= 40 and lon >= -100:
        return "Northeast"
    if lat < 40 and lat >= 35 and lon >= -100:
        return "Mid-Atlantic / Appalachia"
    if lat < 35 and lon >= -90:
        return "Southeast"
    if lon >= -100 and lat >= 40:
        return "Northeast"
    if lon < -100 and lon >= -105:
        return "Mountain West"
    return "South / Midwest"


REGIONS = {
    "Northeast":                 lambda la, lo: lo >= -80  and la >= 39,
    "Mid-Atlantic":              lambda la, lo: lo >= -80  and 35 <= la < 39,
    "Southeast":                 lambda la, lo: lo >= -90  and la < 35,
    "South":                     lambda la, lo: -100 <= lo < -90 and la < 37,
    "Midwest":                   lambda la, lo: -100 <= lo < -80 and la >= 37,
    "Texas / Oklahoma":          lambda la, lo: -105 <= lo < -93 and la < 37,
    "Mountain West":             lambda la, lo: -115 <= lo < -100,
    "Pacific Coast":             lambda la, lo: lo < -115,
}


def assign_region(lat, lon):
    for name, fn in REGIONS.items():
        if fn(lat, lon):
            return name
    return "Other"


def latest_checkpoint(ckpt_dir):
    import glob
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "epoch_*.pt")))
    return ckpts[-1] if ckpts else None


def load_model():
    cells = pd.read_csv(CELLS_CSV)
    n_cells = len(cells)
    cell_centers = cells[["lat_center", "lon_center"]].values
    m = M.GeoClassifier(n_cells, cell_centers).to(DEVICE)
    ckpt_path = latest_checkpoint(CKPT_DIR)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    m.load_state_dict(ckpt["model"])
    m.eval()
    print(f"loaded {ckpt_path}")
    return m


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    model = load_model()
    transform = D.make_val_transform(size=336)

    val_ds = D.GeoValDataset(VAL_CSV, "", transform)
    loader = torch.utils.data.DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=8, collate_fn=D.collate_skip_none, pin_memory=True
    )

    rows = []
    offset = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="evaluating"):
            if batch is None:
                continue
            imgs, gps = batch
            imgs = imgs.to(DEVICE, non_blocking=True)

            logits = model(imgs)                      # (B, N)
            probs  = logits.softmax(dim=-1)           # (B, N)
            top    = torch.topk(probs, TOP_K, dim=-1) # indices + values

            # top-1 prediction
            top1_idx = top.indices[:, 0]
            pred_coords = model.cell_centers[top1_idx].cpu().numpy()

            # weighted consensus (top-5)
            top_probs  = top.values.cpu().numpy()          # (B, 5)
            top_coords = model.cell_centers[top.indices.cpu()].cpu().numpy()  # (B, 5, 2)
            w = top_probs / top_probs.sum(axis=1, keepdims=True)
            consensus  = (top_coords * w[:, :, None]).sum(axis=1)       # (B, 2)

            actual = gps.numpy()  # (B, 2) lat/lon

            for i in range(len(actual)):
                a_lat, a_lon = actual[i]
                p_lat, p_lon = pred_coords[i]
                c_lat, c_lon = consensus[i]
                rows.append({
                    "actual_lat":    round(float(a_lat), 6),
                    "actual_lon":    round(float(a_lon), 6),
                    "top1_lat":      round(float(p_lat), 6),
                    "top1_lon":      round(float(p_lon), 6),
                    "consensus_lat": round(float(c_lat), 6),
                    "consensus_lon": round(float(c_lon), 6),
                    "top1_km":       round(float(haversine_np(a_lat, a_lon, p_lat, p_lon)), 3),
                    "consensus_km":  round(float(haversine_np(a_lat, a_lon, c_lat, c_lon)), 3),
                    "region":        assign_region(a_lat, a_lon),
                })
            offset += len(actual)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "predictions.csv"), index=False)
    print(f"saved {len(df):,} predictions → {OUT_DIR}/predictions.csv")

    # summary
    lines = []
    lines.append("=" * 60)
    lines.append("geoKrid  —  Validation Report")
    lines.append(f"checkpoint: {latest_checkpoint(CKPT_DIR)}")
    lines.append(f"val images: {len(df):,}")
    lines.append("=" * 60)

    for label, col in [("Top-1 prediction", "top1_km"), ("Weighted consensus", "consensus_km")]:
        km = df[col]
        lines.append(f"\n{label}")
        lines.append(f"  avg    {km.mean():.1f} km  |  median {km.median():.1f} km")
        for thresh in [1, 25, 100, 200, 750, 2500]:
            pct = (km <= thresh).mean() * 100
            lines.append(f"  acc@{thresh}km{'':>{6-len(str(thresh))}}  {pct:.2f}%")

    lines.append("\n" + "=" * 60)
    lines.append("Regional breakdown  (consensus_km median)")
    lines.append("=" * 60)
    reg = df.groupby("region")["consensus_km"].agg(["median", "mean", "count"]).sort_values("median")
    for region_name, row in reg.iterrows():
        lines.append(f"  {region_name:<30}  median {row['median']:>6.1f} km  "
                     f"mean {row['mean']:>7.1f} km  n={int(row['count']):,}")

    summary = "\n".join(lines)
    print(summary)
    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(summary + "\n")
    print(f"\nsaved {OUT_DIR}/summary.txt")


if __name__ == "__main__":
    run()
