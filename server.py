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
TOP_K       = 5

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
        logits = model(x)                         # (1, N)
        probs  = logits.softmax(dim=-1)[0]        # (N,)
        top    = torch.topk(probs, TOP_K)

    results = []
    for rank, (idx, prob) in enumerate(zip(top.indices.tolist(), top.values.tolist())):
        lat, lon = model.cell_centers[idx].tolist()
        results.append({
            "rank":    rank + 1,
            "cell_id": idx,
            "lat":     round(lat, 5),
            "lon":     round(lon, 5),
            "prob":    round(prob * 100, 2),
        })

    return jsonify({"predictions": results})


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
