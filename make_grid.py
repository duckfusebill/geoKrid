"""
One-time script: builds adaptive geographic cells from OSV5M training data
using K-means clustering on GPS coordinates.

Dense areas (cities) get many small clusters; sparse areas get few large ones.
Saves:
  cells.csv                               - cell_id, lat_center, lon_center, count
  /mnt/b/datasets/geovit/us_train_cls.csv - training split with cell_id column
  /mnt/b/datasets/geovit/us_val_cls.csv   - val split with cell_id column
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

N_CLUSTERS  = 4096
BATCH_SIZE  = 50000
TRAIN_CSV   = "/mnt/b/datasets/geovit/us_train.csv"
VAL_CSV     = "/mnt/b/datasets/geovit/us_val.csv"
OUT_CELLS   = "cells.csv"
OUT_TRAIN   = "/mnt/b/datasets/geovit/us_train_cls.csv"
OUT_VAL     = "/mnt/b/datasets/geovit/us_val_cls.csv"


def main():
    print("loading training CSV...")
    train = pd.read_csv(TRAIN_CSV)
    print(f"  {len(train):,} rows")

    coords = train[["lat", "lon"]].values

    print(f"fitting K-means with {N_CLUSTERS} clusters...")
    km = MiniBatchKMeans(
        n_clusters=N_CLUSTERS,
        batch_size=BATCH_SIZE,
        n_init=3,
        random_state=42,
        verbose=1,
    )
    km.fit(coords)

    train["cell_id"] = km.labels_.astype(np.int32)

    centroids = km.cluster_centers_          # (N, 2)
    counts = train.groupby("cell_id").size()

    cells = pd.DataFrame({
        "cell_id":    np.arange(N_CLUSTERS, dtype=np.int32),
        "lat_center": centroids[:, 0],
        "lon_center": centroids[:, 1],
        "count":      [counts.get(i, 0) for i in range(N_CLUSTERS)],
    })
    cells.to_csv(OUT_CELLS, index=False)
    print(f"  saved {OUT_CELLS}  ({N_CLUSTERS} clusters)")

    train[["image_path", "lat", "lon", "cell_id"]].to_csv(OUT_TRAIN, index=False)
    print(f"  train: {len(train):,} rows → {OUT_TRAIN}")

    print("loading val CSV...")
    val = pd.read_csv(VAL_CSV)
    val_coords = val[["lat", "lon"]].values
    val["cell_id"] = km.predict(val_coords).astype(np.int32)
    val[["image_path", "lat", "lon", "cell_id"]].to_csv(OUT_VAL, index=False)
    print(f"  val: {len(val):,} rows → {OUT_VAL}")

    print(f"\ndone. cluster size stats:")
    print(f"  min {counts.min()}  median {int(counts.median())}  max {counts.max()}  images/cluster")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-clusters", type=int, default=N_CLUSTERS)
    parser.add_argument("--train-csv",  default=TRAIN_CSV)
    parser.add_argument("--val-csv",    default=VAL_CSV)
    parser.add_argument("--out-cells",  default=OUT_CELLS)
    args = parser.parse_args()

    N_CLUSTERS = args.n_clusters
    TRAIN_CSV  = args.train_csv
    VAL_CSV    = args.val_csv
    OUT_CELLS  = args.out_cells

    main()
