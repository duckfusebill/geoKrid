import torch
import numpy as np
from tqdm import tqdm

KM_THRESHOLDS = [1, 25, 200, 750, 2500]


def haversine_np(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.deg2rad(lat2 - lat1)
    dlon = np.deg2rad(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.deg2rad(lat1))
         * np.cos(np.deg2rad(lat2))
         * np.sin(dlon / 2) ** 2)
    return 2 * R * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))


def eval_model(loader, model, device):
    """
    Evaluates GeoClassifier on val loader.
    Predicts cell → centroid coords, computes haversine distance to true GPS.
    """
    model.eval()
    pred_lats, pred_lons = [], []
    true_lats, true_lons = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="eval"):
            if batch is None:
                continue
            imgs, gps = batch
            imgs = imgs.to(device, non_blocking=True)

            with torch.amp.autocast("cuda"):
                logits = model(imgs)

            idx = logits.argmax(dim=-1)
            centroids = model.cell_centers[idx].cpu().numpy()  # (B, 2)

            pred_lats.append(centroids[:, 0])
            pred_lons.append(centroids[:, 1])
            true_lats.append(gps[:, 0].numpy())
            true_lons.append(gps[:, 1].numpy())

    model.train()

    if not pred_lats:
        print("no predictions; check loader")
        return {}

    pred_lats = np.concatenate(pred_lats)
    pred_lons = np.concatenate(pred_lons)
    true_lats = np.concatenate(true_lats)
    true_lons = np.concatenate(true_lons)

    dists = haversine_np(true_lats, true_lons, pred_lats, pred_lons)

    results = {
        "avg_km":    float(dists.mean()),
        "median_km": float(np.median(dists)),
    }
    for km in KM_THRESHOLDS:
        results[f"acc@{km}km"] = float((dists <= km).mean())

    print(f"  avg {results['avg_km']:.1f} km | median {results['median_km']:.1f} km")
    for km in KM_THRESHOLDS:
        print(f"  acc@{km}km  {results[f'acc@{km}km']:.4f}")

    return results
