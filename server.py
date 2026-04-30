import os
import glob
import io
import torch
import pandas as pd
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
import dataset as D
import model as M

app = Flask(__name__, static_folder="static", static_url_path="")

CELLS_CSV   = os.environ.get("CELLS_CSV",   "cells.csv")
CKPT_DIR    = os.environ.get("CKPT_DIR",    "checkpoints_kmeans")
DEVICE      = os.environ.get("DEVICE",      "cuda" if torch.cuda.is_available() else "cpu")
TOP_K       = 10          # expanded candidate pool
TEMPERATURE = 1.8         # calibrated softmax temperature (flattens overconfident peaks)
SIGMA_KM    = 150.0       # spatial re-weighting bandwidth

_model = None


def _latest_checkpoint(ckpt_dir):
    epoch_ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "epoch_*.pt")))
    return epoch_ckpts[-1] if epoch_ckpts else None


def get_model():
    global _model
    if _model is None:
        cells = pd.read_csv(CELLS_CSV)
        n_cells = len(cells)
        cell_centers = cells[["lat_center", "lon_center"]].values
        _model = M.GeoClassifier(n_cells, cell_centers).to(DEVICE)

        ckpt_path = _latest_checkpoint(CKPT_DIR)
        if ckpt_path:
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
            _model.load_state_dict(ckpt["model"])
            app.logger.info(f"loaded {ckpt_path}")
        else:
            app.logger.warning("no checkpoint found — using random weights")

        _model.eval()
    return _model


def _haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _geometric_median(coords, weights, eps=1e-5, max_iter=50):
    """Weiszfeld algorithm — weighted geometric median, robust to outliers."""
    weights = np.array(weights)
    estimate = np.average(coords, weights=weights, axis=0)
    for _ in range(max_iter):
        dists = np.array([_haversine(estimate[0], estimate[1], c[0], c[1]) for c in coords])
        dists = np.maximum(dists, eps)
        w = weights / dists
        new_estimate = np.average(coords, weights=w, axis=0)
        if np.linalg.norm(new_estimate - estimate) < eps:
            break
        estimate = new_estimate
    return estimate


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "no image uploaded"}), 400

    file = request.files["image"]
    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception:
        return jsonify({"error": "invalid image"}), 400

    transform = D.make_val_transform(size=336)
    x = transform(image).unsqueeze(0).to(DEVICE)

    model = get_model()
    with torch.no_grad():
        logits = model(x)                              # (1, N)
        probs  = (logits / TEMPERATURE).softmax(dim=-1)[0]   # temperature scaling
        top    = torch.topk(probs, TOP_K)

    indices = top.indices.tolist()
    raw_probs = top.values.tolist()

    coords = np.array([model.cell_centers[i].tolist() for i in indices])  # (K, 2)

    # spatial consistency re-weighting
    # boost predictions that cluster geographically with other high-prob predictions
    spatially_weighted = []
    for i in range(TOP_K):
        score = 0.0
        for j in range(TOP_K):
            if i == j:
                continue
            d = _haversine(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
            score += raw_probs[j] * np.exp(-d / SIGMA_KM)
        spatially_weighted.append(raw_probs[i] * (1.0 + score))

    total = sum(spatially_weighted)
    reweighted = [w / total for w in spatially_weighted]

    # geometric median via Weiszfeld iterations (robust to outliers)
    consensus = _geometric_median(coords, reweighted)

    results = []
    for rank, (idx, rw_prob, raw_prob) in enumerate(zip(indices, reweighted, raw_probs)):
        lat, lon = coords[rank]
        results.append({
            "rank":    rank + 1,
            "cell_id": idx,
            "lat":     round(float(lat), 5),
            "lon":     round(float(lon), 5),
            "prob":    round(rw_prob * 100, 2),
            "raw_prob": round(raw_prob * 100, 2),
        })

    return jsonify({
        "predictions": results,
        "consensus": {
            "lat": round(float(consensus[0]), 5),
            "lon": round(float(consensus[1]), 5),
        }
    })


@app.route("/cells")
def cells():
    df = pd.read_csv(CELLS_CSV)
    return jsonify(df[["cell_id", "lat_center", "lon_center", "count"]].to_dict(orient="records"))


@app.route("/status")
def status():
    ckpt = _latest_checkpoint(CKPT_DIR)
    return jsonify({
        "checkpoint": os.path.basename(ckpt) if ckpt else None,
        "device": DEVICE,
        "model_loaded": _model is not None,
    })


if __name__ == "__main__":
    print(f"loading model on {DEVICE}...")
    get_model()
    app.run(host="0.0.0.0", port=5000, debug=False)
