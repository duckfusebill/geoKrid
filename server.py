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
TOP_K       = 10
TEMPERATURE = 1.8
SIGMA_KM    = 150.0

# cache: basename → loaded model
_model_cache: dict = {}
_cells_df = None
_cell_centers_np = None


def _list_checkpoints():
    ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "epoch_*.pt")))
    return [os.path.basename(c) for c in ckpts]


def _latest_checkpoint():
    ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "epoch_*.pt")))
    return os.path.basename(ckpts[-1]) if ckpts else None


def _load_cells():
    global _cells_df, _cell_centers_np
    if _cells_df is None:
        _cells_df = pd.read_csv(CELLS_CSV)
        _cell_centers_np = _cells_df[["lat_center", "lon_center"]].values
    return _cells_df, _cell_centers_np


def get_model(ckpt_name: str | None = None):
    cells_df, cell_centers = _load_cells()
    if ckpt_name is None:
        ckpt_name = _latest_checkpoint()

    if ckpt_name not in _model_cache:
        m = M.GeoClassifier(len(cells_df), cell_centers).to(DEVICE)
        ckpt_path = os.path.join(CKPT_DIR, ckpt_name)
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
            m.load_state_dict(ckpt["model"])
            app.logger.info(f"loaded {ckpt_path}")
        else:
            app.logger.warning(f"checkpoint not found: {ckpt_path}")
        m.eval()
        _model_cache[ckpt_name] = m

    return _model_cache[ckpt_name], ckpt_name


def _haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _geometric_median(coords, weights, eps=1e-5, max_iter=50):
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


@app.route("/checkpoints")
def checkpoints():
    names = _list_checkpoints()
    latest = _latest_checkpoint()
    return jsonify({"checkpoints": names, "default": latest})


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "no image uploaded"}), 400

    file = request.files["image"]
    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception:
        return jsonify({"error": "invalid image"}), 400

    ckpt_name = request.form.get("checkpoint") or None
    head      = request.form.get("head", "consensus")   # "top1" or "consensus"

    transform = D.make_val_transform(size=336)
    x = transform(image).unsqueeze(0).to(DEVICE)

    model, resolved_ckpt = get_model(ckpt_name)
    with torch.no_grad():
        logits = model(x)
        probs  = (logits / TEMPERATURE).softmax(dim=-1)[0]
        top    = torch.topk(probs, TOP_K)

    indices   = top.indices.tolist()
    raw_probs = top.values.tolist()
    coords    = np.array([model.cell_centers[i].tolist() for i in indices])

    if head == "top1":
        # top-1 argmax: consensus == top prediction
        consensus = coords[0]
        reweighted = raw_probs  # display raw probs as-is
    else:
        # optimized consensus: spatial reweight + geometric median
        spatially_weighted = []
        for i in range(TOP_K):
            score = sum(
                raw_probs[j] * np.exp(-_haversine(coords[i][0], coords[i][1],
                                                  coords[j][0], coords[j][1]) / SIGMA_KM)
                for j in range(TOP_K) if j != i
            )
            spatially_weighted.append(raw_probs[i] * (1.0 + score))
        total = sum(spatially_weighted)
        reweighted = [w / total for w in spatially_weighted]
        consensus  = _geometric_median(coords, reweighted)

    results = []
    for rank, (idx, rw_prob, raw_prob) in enumerate(zip(indices, reweighted, raw_probs)):
        lat, lon = coords[rank]
        results.append({
            "rank":     rank + 1,
            "cell_id":  idx,
            "lat":      round(float(lat), 5),
            "lon":      round(float(lon), 5),
            "prob":     round(rw_prob * 100, 2),
            "raw_prob": round(raw_prob * 100, 2),
        })

    return jsonify({
        "predictions": results,
        "consensus": {
            "lat": round(float(consensus[0]), 5),
            "lon": round(float(consensus[1]), 5),
        },
        "checkpoint": resolved_ckpt,
        "head": head,
    })


@app.route("/cells")
def cells():
    df = pd.read_csv(CELLS_CSV)
    return jsonify(df[["cell_id", "lat_center", "lon_center", "count"]].to_dict(orient="records"))


@app.route("/status")
def status():
    return jsonify({
        "checkpoints": _list_checkpoints(),
        "default": _latest_checkpoint(),
        "device": DEVICE,
        "loaded": list(_model_cache.keys()),
    })


if __name__ == "__main__":
    print(f"loading default model on {DEVICE}...")
    get_model()
    app.run(host="0.0.0.0", port=5000, debug=False)
