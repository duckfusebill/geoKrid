#!/usr/bin/env python3
# download_data.py - download and prepare geovit384 training datasets
# usage: python3 download_data.py [--dry-run]

import os
import sys
import io
import zipfile
import shutil
import argparse
import traceback
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

"""config"""
BASE_DIR = Path(os.environ.get("GEOVIT_DIR", "/mnt/b/datasets/geovit"))
HF_BASE  = "https://huggingface.co/datasets"
US_LAT   = (24.0, 49.0)
US_LON   = (-125.0, -66.0)

# osv5m has 98 image zip shards (train/00.zip .. train/97.zip)
OSV5M_N_SHARDS = 98
OSV5M_DISK_STOP_GB = 10  # stop downloading shards if free space drops below this

"""helpers"""
def in_us(lat, lon):
    return US_LAT[0] <= lat <= US_LAT[1] and US_LON[0] <= lon <= US_LON[1]

def filter_us(df, lat="latitude", lon="longitude"):
    return df[
        (df[lat] >= US_LAT[0]) & (df[lat] <= US_LAT[1]) &
        (df[lon] >= US_LON[0]) & (df[lon] <= US_LON[1])
    ].reset_index(drop=True)

def du(path):
    # human-readable disk usage for a path
    total = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if total < 1024:
            return f"{total:.1f} {unit}"
        total /= 1024
    return f"{total:.1f} PB"

def free_gb(path):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / (1024 ** 3)

def hf_headers():
    token = os.environ.get("HF_TOKEN", "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}

def download_file(url, dest, desc=None, chunk=1 << 20):
    # stream download with progress bar; returns False on failure
    try:
        r = requests.get(url, stream=True, timeout=60, headers=hf_headers())
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) or None
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=desc or dest.name, leave=False
        ) as bar:
            for chunk_data in r.iter_content(chunk):
                f.write(chunk_data)
                bar.update(len(chunk_data))
        return True
    except Exception as e:
        print(f"  download error: {e}")
        if Path(dest).exists():
            Path(dest).unlink()
        return False

def hf_url(repo, filename):
    return f"{HF_BASE}/{repo}/resolve/main/{filename}"

def hf_ds_kwargs():
    token = os.environ.get("HF_TOKEN", "").strip()
    return {"token": token} if token else {}


"""osv5m"""
def download_osv5m(dry_run=False):
    csv_out = BASE_DIR / "us_osv5m.csv"
    if csv_out.exists():
        df = pd.read_csv(csv_out)
        print(f"osv5m: already done — {len(df)} rows at {csv_out}")
        return df

    print("\n=== osv5m ===")

    # --- step 1: metadata csv ---
    meta_cache = BASE_DIR / "_cache" / "osv5m_train.csv"
    if not meta_cache.exists():
        print("osv5m: downloading train metadata csv (~500 MB)...")
        url = hf_url("osv5m/osv5m", "train.csv")
        if dry_run:
            print(f"  [dry-run] would download {url}")
            return None
        ok = download_file(url, meta_cache, desc="train.csv")
        if not ok:
            print("osv5m: failed to download metadata — skipping")
            return None
    else:
        print(f"osv5m: metadata cache found at {meta_cache}")

    # --- step 2: filter to us ---
    print("osv5m: filtering to US bounding box...")
    full_df = pd.read_csv(meta_cache, low_memory=False)
    us_df = filter_us(full_df, lat="latitude", lon="longitude")
    print(f"osv5m: {len(us_df):,} US rows out of {len(full_df):,} total ({len(us_df)/len(full_df)*100:.1f}%)")

    us_ids = set(us_df["id"].astype(str))

    # --- step 3: download image shards, extract us images ---
    img_dir = BASE_DIR / "osv5m" / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    already_have = {p.stem for p in img_dir.glob("*.jpg")}
    remaining_ids = us_ids - already_have
    print(f"osv5m: {len(already_have):,} images already on disk; need {len(remaining_ids):,} more")

    if remaining_ids and not dry_run:
        tmp_zip = BASE_DIR / "_cache" / "current_shard.zip"
        tmp_zip.parent.mkdir(parents=True, exist_ok=True)

        for i in tqdm(range(OSV5M_N_SHARDS), desc="osv5m shards"):
            shard = f"{i:02d}"
            url = hf_url("osv5m/osv5m", f"images/train/{shard}.zip")

            if not remaining_ids:
                break

            gb_free = free_gb(BASE_DIR)
            if gb_free < OSV5M_DISK_STOP_GB:
                print(f"\nosv5m: only {gb_free:.1f} GB free — stopping after shard {i-1} to protect disk")
                break

            ok = download_file(url, tmp_zip, desc=f"shard {shard}")
            if not ok:
                print(f"  osv5m: shard {shard} download failed, skipping")
                continue

            extracted = 0
            try:
                with zipfile.ZipFile(tmp_zip, "r") as zf:
                    for name in zf.namelist():
                        stem = Path(name).stem
                        if stem in remaining_ids:
                            dest = img_dir / Path(name).name
                            with zf.open(name) as src, open(dest, "wb") as dst:
                                dst.write(src.read())
                            remaining_ids.discard(stem)
                            extracted += 1
            except zipfile.BadZipFile:
                print(f"  osv5m: shard {shard} is corrupt, skipping")

            tmp_zip.unlink(missing_ok=True)
            tqdm.write(f"  shard {shard}: extracted {extracted} US images ({len(remaining_ids):,} still needed)")

        if remaining_ids:
            print(f"osv5m: warning — {len(remaining_ids):,} US ids not found in any shard")

    elif dry_run:
        print(f"  [dry-run] would download up to {OSV5M_N_SHARDS} zip shards and extract US images")

    # --- step 4: build final csv ---
    # only include rows where image file actually exists
    us_df = us_df.copy()
    us_df["image_path"] = us_df["id"].apply(lambda x: str(img_dir / f"{x}.jpg"))
    if not dry_run:
        us_df = us_df[us_df["image_path"].apply(os.path.exists)]
    us_out = us_df[["image_path", "latitude", "longitude"]].rename(
        columns={"latitude": "lat", "longitude": "lon"}
    )
    us_out.to_csv(csv_out, index=False)
    print(f"osv5m: saved {len(us_out):,} rows to {csv_out}")
    if img_dir.exists():
        print(f"osv5m: disk usage {du(img_dir)}")
    return us_out


"""mapillary"""
def download_mapillary(dry_run=False):
    csv_out = BASE_DIR / "us_mapillary.csv"
    if csv_out.exists():
        df = pd.read_csv(csv_out)
        print(f"mapillary: already done — {len(df)} rows at {csv_out}")
        return df

    print("\n=== mapillary ===")

    token = os.environ.get("MAPILLARY_TOKEN", "").strip()
    if not token:
        print("mapillary: MAPILLARY_TOKEN env var not set — skipping")
        print("  set it with: export MAPILLARY_TOKEN=<your_token>")
        print("  get a token at: https://www.mapillary.com/developer/api-documentation")
        return None

    if dry_run:
        print(f"  [dry-run] would query Mapillary API with token from env")
        return None

    img_dir = BASE_DIR / "mapillary" / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # query sequences covering the US bounding box using v4 API
    base_url = "https://graph.mapillary.com/images"
    params = {
        "access_token": token,
        "fields": "id,geometry,thumb_1024_url",
        "bbox": f"{US_LON[0]},{US_LAT[0]},{US_LON[1]},{US_LAT[1]}",
        "limit": 2000,
    }

    rows = []
    next_url = base_url
    page = 0

    print("mapillary: querying API for US images...")
    try:
        while next_url:
            r = requests.get(next_url, params=params if page == 0 else None, timeout=30)
            r.raise_for_status()
            data = r.json()

            for item in data.get("data", []):
                try:
                    lon_, lat_ = item["geometry"]["coordinates"]
                    if in_us(lat_, lon_):
                        rows.append({
                            "img_id": item["id"],
                            "lat": lat_,
                            "lon": lon_,
                            "thumb_url": item.get("thumb_1024_url", ""),
                        })
                except (KeyError, TypeError):
                    continue

            paging = data.get("paging", {})
            next_url = paging.get("next")
            params = None  # params embedded in next_url on subsequent pages
            page += 1

            tqdm.write(f"  mapillary: page {page}, {len(rows):,} images so far")
            if page >= 500:  # cap at 1M images to avoid infinite loop
                print("  mapillary: reached page cap, stopping")
                break

    except Exception as e:
        print(f"mapillary: API error — {e}")
        if not rows:
            return None

    if not rows:
        print("mapillary: no US images found")
        return None

    meta_df = pd.DataFrame(rows)
    print(f"mapillary: {len(meta_df):,} US images found; downloading...")

    # download thumbnail images
    downloaded = []
    for _, row in tqdm(meta_df.iterrows(), total=len(meta_df), desc="mapillary images"):
        dest = img_dir / f"{row['img_id']}.jpg"
        if dest.exists():
            downloaded.append(str(dest))
            continue
        if row["thumb_url"]:
            ok = download_file(row["thumb_url"], dest)
            if ok:
                downloaded.append(str(dest))

    meta_df = meta_df[meta_df["img_id"].apply(lambda x: (img_dir / f"{x}.jpg").exists())]
    out_df = meta_df[["img_id", "lat", "lon"]].copy()
    out_df["image_path"] = out_df["img_id"].apply(lambda x: str(img_dir / f"{x}.jpg"))
    out_df = out_df[["image_path", "lat", "lon"]]
    out_df.to_csv(csv_out, index=False)
    print(f"mapillary: saved {len(out_df):,} rows to {csv_out}")
    if img_dir.exists():
        print(f"mapillary: disk usage {du(img_dir)}")
    return out_df


"""streetview"""
def download_streetview(dry_run=False):
    csv_out = BASE_DIR / "us_streetview.csv"
    if csv_out.exists():
        df = pd.read_csv(csv_out)
        print(f"streetview: already done — {len(df)} rows at {csv_out}")
        return df

    print("\n=== streetview (yunusserhat/random_streetview_images) ===")

    try:
        from datasets import load_dataset
    except ImportError:
        print("streetview: `datasets` not installed — skipping")
        return None

    img_dir = BASE_DIR / "streetview" / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    repo_id = "yunusserhat/random_streetview_images"
    try:
        print(f"streetview: loading {repo_id}...")
        if dry_run:
            ds = load_dataset(repo_id, split="train", streaming=True, **hf_ds_kwargs())
            row = next(iter(ds))
            print(f"  [dry-run] columns: {list(row.keys())}")
            return None

        ds = load_dataset(repo_id, split="train", **hf_ds_kwargs())
        rows = []
        for idx, item in enumerate(tqdm(ds, desc="streetview", total=len(ds))):
            lat = item.get("latitude")
            lon = item.get("longitude")
            if lat is None or lon is None:
                continue
            lat, lon = float(lat), float(lon)
            if not in_us(lat, lon):
                continue

            img_id = item.get("id") or idx
            dest = img_dir / f"{img_id}.jpg"

            if not dest.exists():
                img = item.get("image")
                if img is not None:
                    img.save(str(dest))

            if dest.exists():
                rows.append({"image_path": str(dest), "lat": lat, "lon": lon})

        if not rows:
            print("streetview: no US rows found")
            return None

        out_df = pd.DataFrame(rows)
        out_df.to_csv(csv_out, index=False)
        print(f"streetview: saved {len(out_df):,} rows to {csv_out}")
        print(f"streetview: disk usage {du(img_dir)}")
        return out_df

    except Exception as e:
        print(f"streetview: failed — {e}")

    print("streetview: no usable dataset found — skipping")
    return None


"""inaturalist"""
def download_inaturalist(dry_run=False):
    csv_out = BASE_DIR / "us_inaturalist.csv"
    if csv_out.exists():
        df = pd.read_csv(csv_out)
        print(f"inaturalist: already done — {len(df)} rows at {csv_out}")
        return df

    print("\n=== inaturalist ===")

    try:
        from datasets import load_dataset
    except ImportError:
        print("inaturalist: `datasets` not installed — skipping")
        return None

    img_dir = BASE_DIR / "inaturalist" / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    repo_id = "ba188/iNaturalist_v2"
    try:
        print(f"inaturalist: loading {repo_id}...")
        if dry_run:
            ds = load_dataset(repo_id, split="train", streaming=True, **hf_ds_kwargs())
            row = next(iter(ds))
            print(f"  [dry-run] columns: {list(row.keys())}")
            print(f"  [dry-run] sample location={row.get('location')!r}")
            return None

        ds = load_dataset(repo_id, split="train", streaming=True, **hf_ds_kwargs())
        rows = []
        for item in tqdm(ds, desc="inaturalist"):
            loc = item.get("location", "")
            if not loc:
                continue
            try:
                lat_s, lon_s = str(loc).split(",", 1)
                lat, lon = float(lat_s.strip()), float(lon_s.strip())
            except (ValueError, TypeError):
                continue
            if not in_us(lat, lon):
                continue

            img_id = item.get("id") or item.get("image_id") or len(rows)
            dest = img_dir / f"{img_id}.jpg"

            if not dest.exists():
                img = item.get("image")
                if img is not None:
                    img.save(str(dest))

            if dest.exists():
                rows.append({"image_path": str(dest), "lat": lat, "lon": lon})

        if not rows:
            print("inaturalist: no geotagged US rows found")
            return None

        out_df = pd.DataFrame(rows)
        out_df.to_csv(csv_out, index=False)
        print(f"inaturalist: saved {len(out_df):,} rows to {csv_out}")
        print(f"inaturalist: disk usage {du(img_dir)}")
        return out_df

    except Exception as e:
        print(f"inaturalist: failed — {e}")

    print("inaturalist: no usable dataset found — skipping")
    return None


"""usgs"""
def skip_usgs():
    print("\n=== usgs earthexplorer ===")
    print("USGS EarthExplorer requires manual download - skipping")
    print("  visit: https://earthexplorer.usgs.gov")


"""combine"""
def merge_all():
    print("\n=== building us_combined.csv ===")
    sources = {
        "osv5m":       BASE_DIR / "us_osv5m.csv",
        "mapillary":   BASE_DIR / "us_mapillary.csv",
        "streetview":  BASE_DIR / "us_streetview.csv",
        "inaturalist": BASE_DIR / "us_inaturalist.csv",
    }

    frames = []
    for source, path in sources.items():
        if path.exists():
            df = pd.read_csv(path)[["image_path", "lat", "lon"]].copy()
            df["source"] = source
            # keep only rows where image actually exists on disk
            df = df[df["image_path"].apply(os.path.exists)]
            frames.append(df)
            print(f"  {source}: {len(df):,} valid rows")
        else:
            print(f"  {source}: csv not found, skipping")

    if not frames:
        print("combine: no datasets available")
        return

    combined = pd.concat(frames, ignore_index=True)
    out = BASE_DIR / "us_combined.csv"
    combined.to_csv(out, index=False)
    print(f"\ncombined: {len(combined):,} total rows -> {out}")

    # per-source summary
    for src, grp in combined.groupby("source"):
        print(f"  {src}: {len(grp):,}")

    print(f"total disk usage: {du(BASE_DIR)}")


"""entry"""
def main():
    parser = argparse.ArgumentParser(description="download geovit384 datasets")
    parser.add_argument("--dry-run", action="store_true",
                        help="print what would be downloaded without actually downloading")
    parser.add_argument("--only", nargs="+",
                        choices=["osv5m", "mapillary", "streetview", "inaturalist"],
                        help="download only specific datasets")
    args = parser.parse_args()

    try:
        BASE_DIR.mkdir(parents=True, exist_ok=True)
        (BASE_DIR / "_cache").mkdir(exist_ok=True)
    except PermissionError as e:
        print(f"error: cannot create {BASE_DIR}: {e}")
        print("check that the drive is mounted and you have write permissions")
        sys.exit(1)

    want = set(args.only) if args.only else {"osv5m", "mapillary", "streetview", "inaturalist"}

    if "osv5m" in want:
        try:
            download_osv5m(dry_run=args.dry_run)
        except Exception:
            print(f"osv5m: unexpected error\n{traceback.format_exc()}")

    if "mapillary" in want:
        try:
            download_mapillary(dry_run=args.dry_run)
        except Exception:
            print(f"mapillary: unexpected error\n{traceback.format_exc()}")

    if "streetview" in want:
        try:
            download_streetview(dry_run=args.dry_run)
        except Exception:
            print(f"streetview: unexpected error\n{traceback.format_exc()}")

    if "inaturalist" in want:
        try:
            download_inaturalist(dry_run=args.dry_run)
        except Exception:
            print(f"inaturalist: unexpected error\n{traceback.format_exc()}")

    skip_usgs()

    if not args.dry_run:
        merge_all()


if __name__ == "__main__":
    main()
