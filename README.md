# geoKrid

A grid-based geolocation model for US street-level imagery. Given a photo, geoKrid predicts GPS coordinates by classifying the image into one of 4,096 adaptive geographic cells and returning the cell centroid.

## Architecture

```
CLIP ViT-L/14@336 (frozen)
        ↓ pooler_output (1024-d)
  Linear(1024 → 512) + ReLU + Dropout(0.3)
        ↓
  Linear(512 → 4096)   ← one logit per cell
        ↓
  argmax → cell centroid (lat, lon)
```

**Backbone:** `openai/clip-vit-large-patch14-336` — fully frozen, used as a feature extractor only.  
**Head:** 2-layer MLP with 2.6M trainable parameters (out of 430M total).  
**Loss:** CrossEntropyLoss over 4,096 geographic cells.  
**Grid:** K-means clustering on 1.2M GPS coordinates — dense urban areas get many small cells, rural areas get fewer large ones. Min 21 / median 221 / max 2,289 images per cell.

## Results

Evaluated on 63K held-out US images using haversine distance to ground-truth GPS.

| Epoch | Avg loss | Median dist | acc@25km | acc@200km | acc@750km |
|-------|----------|-------------|----------|-----------|-----------|
| 1     | 8.36     | 1598 km     | 0.07%    | 2.1%      | 18.3%     |
| 2     | 3.93     | **32.1 km** | 46.1%    | 75.6%     | 94.2%     |

## Dataset

[OSV-5M](https://github.com/gastruc/osv5m) — US subset.  
- Train: 1,203,412 images  
- Val: 63,338 images

## Setup

```bash
pip install -r requirements.txt
```

ROCm (AMD GPU) users: the model patches CLIP's patch embedding Conv2d with an unfold+linear equivalent to avoid a gfx1100 segfault on ROCm 6.1.

## Build the grid

Run once before training to generate `cells.csv` and labeled dataset CSVs:

```bash
python make_grid.py \
  --train-csv /path/to/us_train.csv \
  --val-csv   /path/to/us_val.csv
```

## Training

```bash
python runner.py \
  --train-csv /path/to/us_train_cls.csv \
  --val-csv   /path/to/us_val_cls.csv \
  --cells-csv cells.csv \
  --ckpt-dir  checkpoints \
  --epochs    50 \
  --batch-size 128
```

Key hyperparameters: `lr=3e-4`, `weight_decay=1e-2`, 1 warmup epoch, cosine decay, fp16 mixed precision.

Training supports mid-epoch pause/resume — send SIGTERM or `docker stop` and it saves a checkpoint automatically:

```bash
# resume from mid-epoch checkpoint
python runner.py --resume checkpoints/mid_epoch.pt
```

## Inference

```bash
python predict.py image.jpg --checkpoint checkpoints/epoch_002.pt
```

## Project structure

```
make_grid.py   — K-means grid generation
model.py       — CLIPEncoder + GeoClassifier
dataset.py     — GeoTrainDataset, GeoValDataset, SeededSkipSampler
train.py       — training loop, checkpointing, SIGTERM handler
eval.py        — haversine distance evaluation
runner.py      — entry point, config, orchestration
predict.py     — single-image inference CLI
```

## Acknowledgements

geoKrid builds on the ideas and prior work of:

**GeoCLIP** — Vivanco et al., NeurIPS 2023. Introduced contrastive image–location alignment using CLIP embeddings for worldwide geolocation. geoKrid uses CLIP as its visual backbone and draws on GeoCLIP's framing of geolocation as a retrieval problem.

```
@inproceedings{geoclip,
  title={GeoCLIP: Clip-Inspired Alignment between Locations and Images for Effective Worldwide Geo-localization},
  author={Vivanco, Vicente and Nayak, Gaurav Kumar and Shah, Mubarak},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}
```

**GeoViT** — extended GeoCLIP with a larger ViT-L/14@336 backbone and ROCm compatibility patches. The unfold+linear Conv2d workaround for gfx1100 originates from this work and is preserved in geoKrid's `model.py`.
