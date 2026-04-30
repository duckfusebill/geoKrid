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

Training curve on 63K held-out US images (K-means adaptive grid, 4,096 cells).

| Epoch | Median dist | acc@25km | acc@200km | acc@750km |
|-------|-------------|----------|-----------|-----------|
| 1     | 1598 km     | 0.07%    | 2.1%      | 18.3%     |
| 2     | 32.1 km     | 46.1%    | 75.6%     | 94.2%     |
| …     | …           | …        | …         | …         |
| 12    | **13.1 km** | 64.3%    | 85.8%     | 96.3%     |

The grid switch from a fixed 0.5° grid (plateaued at ~26 km) to K-means clustering dropped median error by 50% and continued improving through epoch 12.

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

## Results — Full Validation (63K images)

Checkpoint `epoch_012.pt`, evaluated on 63,338 held-out US images.

| Method | Median error | Mean error | acc@25km | acc@100km | acc@200km | acc@750km |
|--------|-------------|------------|----------|-----------|-----------|-----------|
| Top-1 prediction | **13.1 km** | 113.4 km | 64.3% | 79.7% | 85.8% | 96.3% |
| Weighted consensus (top-5) | 30.9 km | 116.4 km | 45.3% | 73.8% | 84.7% | 97.3% |

The simple weighted consensus widens median error vs top-1 because the weighted mean is pulled by geographic outliers.  The optimized inference pipeline (temperature scaling + spatial re-weighting + geometric median) corrects this — see [server.py](server.py).

### Regional breakdown (consensus median)

| Region | Median | Mean | n |
|--------|--------|------|---|
| South | 17.0 km | 106.5 km | 5,007 |
| Texas / Oklahoma | 18.6 km | 106.2 km | 337 |
| Pacific Coast | 19.6 km | 98.6 km | 11,553 |
| Mountain West | 20.3 km | 86.5 km | 7,817 |
| Northeast | 31.1 km | 109.9 km | 10,240 |
| Midwest | 45.9 km | 143.1 km | 14,815 |
| Southeast | 47.7 km | 114.2 km | 8,852 |
| Mid-Atlantic | 58.8 km | 166.8 km | 2,400 |

The West and South perform best (denser training coverage, more visually distinctive geography). The Mid-Atlantic and Midwest have the widest mean errors — high prediction variance driven by the large flat-terrain ambiguity zone between cities.

---

## Error Analysis

### vs GeoCLIP

Head-to-head comparison against GeoCLIP (Vivanco et al., NeurIPS 2023) on a random 2,000-image subset of the US validation set. Both models use the same optimized post-processing: temperature scaling, spatial re-weighting (σ = 150 km), and geometric median consensus.

| Method | Median | Mean | MSE (km²) | RMSE | acc@25km | acc@100km | acc@200km | acc@750km |
|--------|--------|------|-----------|------|----------|-----------|-----------|-----------|
| geoKrid top-1 | **13.2 km** | 103.8 km | 85,893 | 293 km | **64.1%** | **80.7%** | **87.4%** | **96.8%** |
| geoKrid consensus | 17.6 km | 104.2 km | **77,147** | **278 km** | 59.0% | 79.4% | 86.8% | 96.9% |
| GeoCLIP top-1 | 264.9 km | 519.9 km | 1,831,472 | 1353 km | 7.8% | 24.0% | 40.8% | 84.9% |
| GeoCLIP consensus | 239.8 km | 493.0 km | 1,831,981 | 1354 km | 8.6% | 25.9% | 43.9% | 86.9% |

geoKrid top-1 median error is **20× lower** than GeoCLIP on US images (13 km vs 265 km). acc@25km is 64% vs 8%. This gap is expected: GeoCLIP is trained worldwide (56M images across all continents), while geoKrid specialises in the US distribution. The comparison illustrates that **domain specificity matters more than model scale** for regional geolocation.

The geoKrid consensus (geometric median) improves MSE/RMSE over top-1 by suppressing outlier predictions, trading a slightly higher median (+4 km) for better tail behavior.

### Failure mode taxonomy

Manual inspection of test images reveals three systematic failure patterns:

**1. Dense-metro gravity pull**
The model is trained on OSV-5M (street-level imagery), which is heavily biased toward large cities. Smaller cities are predicted as nearby metros. Examples:
- Seffner, FL (suburb of Tampa) → consensus ~Orlando. Five predictions distributed between Orlando and Tampa; the Orlando cluster wins spatially.
- Sherman Oaks, CA (San Fernando Valley, LA) → consensus ~San Diego. The Southwest California visual signature is ambiguous enough that San Diego pulls the consensus south.
- NJ suburb (39.9°N, 75.1°W) → consensus shifts ~NYC. The dense NYC cluster dominates adjacent geography.

**2. Flat-terrain ambiguity**
Featureless suburban and rural scenes (flat roads, chain-store signage, overcast sky) produce a diffuse probability distribution spread over the Midwest and South. The model is essentially guessing within a broad region. Mean error in the Midwest (143 km) reflects this.

**3. Small-city data poverty**
Cities with low OSV-5M coverage (Savannah, GA; Richmond, VA; smaller Southern metros) produce confident but wrong predictions — the model has seen too few training images from these areas to learn their distinctive features. Savannah and Richmond both fall into the Southeast region (47.7 km median), the second-worst region.

### Why it fails predictably

Importantly, geoKrid's errors are **structured, not random**. The model almost always places the prediction in the correct state or neighboring state. The median 13.1 km top-1 error means half of all predictions land within a typical metro area's radius. The long tail (mean 113 km) is dominated by the above three failure modes — and all three are fundamentally **data density problems**, not architectural ones.

More training data from underrepresented small cities and rural areas would narrow the mean substantially without any architecture change.

---

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
