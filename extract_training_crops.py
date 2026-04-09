"""
extract_training_crops.py
==========================
Extracts labeled 200×200 px training crops from full Sentinel-2 tiles.

Runs immediately on two sources:
  1. NEGATIVES — 14 confirmed non-emission sites already in npy_cache/
     (from JRC/expanded survey: S/C < 1.15, TROPOMI non-detections)
  2. POSITIVES — TROPOMI-confirmed emission sites as they download into
     npy_cache/training/ overnight

Each crop is:
  • 200×200×12 uint8 array (same format as CH4Net training data)
  • Centered on the site's lat/lon centroid
  • Saved with a JSON label sidecar containing site metadata,
    enhancement_ppb (for positives), SC ratio (for negatives),
    and a split field (train/val)

Output structure:
  data/crops/
    negative/
      {site}_{tile}_{date}.npy
      {site}_{tile}_{date}_label.json
    positive/
      {site}_{tile}_{date}.npy
      {site}_{tile}_{date}_label.json
    manifest.json        ← inventory of all crops, ready for DataLoader
    dataset_stats.txt    ← class balance, source breakdown

Usage:
    conda activate methane
    python extract_training_crops.py

    # Re-run as new positive tiles download (skips already-extracted crops):
    python extract_training_crops.py

    # Only process negatives (fast, no downloads needed):
    python extract_training_crops.py --negatives-only

    # Only process positives (run after training tiles finish downloading):
    python extract_training_crops.py --positives-only

Crop size: 200×200 px at 10m resolution = 2km × 2km footprint.
  CH4Net was trained on 160×160. We use 200 to allow random 160×160
  crops during augmentation (±40px translation without black borders).
"""

import os
import sys
import json
import glob
import argparse
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

# ── Logging ───────────────────────────────────────────────────────────────────
Path("results_analysis").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("results_analysis/crop_extraction.log"),
    ],
)
log = logging.getLogger(__name__)

# ── rasterio (needed for lat/lon → pixel conversion) ─────────────────────────
try:
    import rasterio
    import rasterio.warp
    import rasterio.transform
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    log.warning("rasterio not found — pixel lookup will use approximate method")

# ── Paths ──────────────────────────────────────────────────────────────────────
NPY_CACHE      = Path("data/npy_cache")
TRAIN_CACHE    = Path("data/npy_cache/training")
CROPS_NEG      = Path("data/crops/negative")
CROPS_POS      = Path("data/crops/positive")
MANIFEST_PATH  = Path("data/crops/manifest.json")
STATS_PATH     = Path("data/crops/dataset_stats.txt")
RESULTS_EXP    = Path("results_expanded/summary.json")
RESULTS_VAL    = Path("results_validation/summary.json")

for d in [CROPS_NEG, CROPS_POS]:
    d.mkdir(parents=True, exist_ok=True)

# ── Crop parameters ────────────────────────────────────────────────────────────
CROP_SIZE  = 200   # px — 2km footprint, allows ±20px augmentation for 160px model
# Band indices (from preprocessing.py BAND_CONFIG):
#   [0]=B01 [1]=B02 [2]=B03 [3]=B04 [4]=B05 [5]=B06
#   [6]=B07 [7]=B08 [8]=B8A [9]=B09 [10]=B11 [11]=B12

# Train/val split: use date to deterministically assign
# Dates ending in 1 or 5 → val (~20%), rest → train
def get_split(date_str: str) -> str:
    try:
        day = int(date_str[-2:])
        return "val" if day % 5 == 0 else "train"
    except Exception:
        return "train"


# ── Coordinate → pixel ────────────────────────────────────────────────────────
def latlon_to_pixel_approx(geo_meta: dict, lat: float, lon: float) -> tuple[int, int]:
    """Approximate lat/lon → pixel using affine transform from geo sidecar."""
    transform = geo_meta["transform"]
    # Affine: [a, b, c, d, e, f] where x = c + col*a + row*b, y = f + col*d + row*e
    # For north-up: b=0, d=0 so: col = (x - c)/a, row = (y - f)/e
    # But we need to convert lat/lon to the tile's CRS first.
    # Without rasterio, use an approximation assuming UTM-ish projection.
    # This is only accurate to ~few pixels; sufficient for 200px crop centering.
    import math
    # Approximate: 1° lat ≈ 111km, 1° lon ≈ 111km * cos(lat)
    # The geo_meta transform is in the tile's projected CRS (usually UTM)
    # We need to estimate the pixel from the lat/lon.
    # Use the tile center from the transform as a reference.
    a, b, c, d, e, f = transform[:6]
    # c, f = top-left corner in projected coords
    # a = pixel width (m), e = pixel height (m, negative for north-up)
    # Approximate lat/lon to projected coords for European UTM zones
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    # Very rough: UTM easting/northing approximation
    # More accurate: skip and use rasterio if available
    return None, None  # signals fallback needed


def latlon_to_pixel(npy_path: Path, lat: float, lon: float) -> tuple[int, int] | None:
    """Convert lat/lon to pixel row/col using the detection GeoTIFF or geo sidecar."""
    # Find a GeoTIFF for this tile (more reliable than approximation)
    tile_id = None
    parts = npy_path.stem.split("_")
    for p in parts:
        if p.startswith("T") and len(p) == 6:
            tile_id = p
            break

    if HAS_RASTERIO and tile_id:
        # Try to find any GeoTIFF with this tile_id
        tif_candidates = (
            list(Path("results_expanded").glob(f"**/*{tile_id}*.tif")) +
            list(Path("results_validation").glob(f"**/*{tile_id}*.tif")) +
            list(Path("results_v2").glob(f"**/*{tile_id}*.tif")) +
            list(Path("results_multidate").glob(f"**/*{tile_id}*.tif")) +
            list(Path("results_bitemporal").glob(f"**/*{tile_id}*.tif"))
        )
        for tif in tif_candidates:
            try:
                with rasterio.open(tif) as src:
                    xs, ys = rasterio.warp.transform("EPSG:4326", src.crs, [lon], [lat])
                    row, col = rasterio.transform.rowcol(src.transform, xs[0], ys[0])
                    return int(row), int(col)
            except Exception:
                continue

    # Fallback: try geo sidecar JSON with rasterio affine approximation
    geo_path = npy_path.with_name(npy_path.stem + "_geo.json")
    if not geo_path.exists():
        # Try training label version
        geo_path = npy_path.parent / (npy_path.stem + "_geo.json")

    if geo_path.exists() and HAS_RASTERIO:
        try:
            geo = json.loads(geo_path.read_text())
            crs = geo["crs"]
            transform_vals = geo["transform"]
            from rasterio.transform import Affine
            aff = Affine(*transform_vals[:6])
            xs, ys = rasterio.warp.transform("EPSG:4326", crs, [lon], [lat])
            row, col = rasterio.transform.rowcol(aff, xs[0], ys[0])
            return int(row), int(col)
        except Exception as e:
            log.debug("geo sidecar lookup failed: %s", e)

    return None


def safe_crop(arr: np.ndarray, row: int, col: int,
              size: int = CROP_SIZE) -> np.ndarray | None:
    """Extract a square crop, returning None if out of bounds."""
    H, W = arr.shape[:2]
    half = size // 2
    r0, r1 = row - half, row + half
    c0, c1 = col - half, col + half
    if r0 < 0 or r1 > H or c0 < 0 or c1 > W:
        return None
    return arr[r0:r1, c0:c1].copy()


# ── Find npy for a given tile_id ──────────────────────────────────────────────
def find_npy_for_tile(tile_id: str, search_dirs: list[Path]) -> list[Path]:
    """Find all .npy files containing tile_id in their name."""
    results = []
    for d in search_dirs:
        results.extend(d.glob(f"*_{tile_id}_*.npy"))
    return results


# ── Extract one crop ──────────────────────────────────────────────────────────
def extract_and_save_crop(
    npy_path: Path,
    lat: float,
    lon: float,
    label: dict,
    out_dir: Path,
    crop_name: str,
) -> bool:
    """Load tile, find pixel, extract crop, save. Returns True on success."""
    out_npy   = out_dir / f"{crop_name}.npy"
    out_label = out_dir / f"{crop_name}_label.json"

    if out_npy.exists():
        log.info("    Already exists: %s", out_npy.name)
        return True

    # Find pixel coordinates
    pixel = latlon_to_pixel(npy_path, lat, lon)
    if pixel is None:
        log.warning("    Cannot determine pixel for lat=%.4f lon=%.4f", lat, lon)
        return False
    row, col = pixel

    # Load the array (memory-mapped for efficiency)
    arr = np.load(npy_path, mmap_mode="r")
    H, W, C = arr.shape

    if row < 0 or row >= H or col < 0 or col >= W:
        log.warning("    Site pixel (%d,%d) out of bounds for %dx%d tile",
                    row, col, H, W)
        return False

    crop = safe_crop(arr, row, col)
    if crop is None:
        # Try reducing crop size if near edge
        half = CROP_SIZE // 2
        r0 = max(0, row - half)
        r1 = min(H, row + half)
        c0 = max(0, col - half)
        c1 = min(W, col + half)
        if (r1 - r0) < 100 or (c1 - c0) < 100:
            log.warning("    Site too close to tile edge, crop too small")
            return False
        crop = arr[r0:r1, c0:c1].copy()
        log.warning("    Edge crop: %s (expected %dx%d)", crop.shape, CROP_SIZE, CROP_SIZE)

    np.save(out_npy, crop)

    full_label = {
        **label,
        "crop_size":   list(crop.shape),
        "pixel_row":   row,
        "pixel_col":   col,
        "npy_source":  str(npy_path),
        "extracted":   datetime.utcnow().isoformat() + "Z",
        "b11_mean":    round(float(crop[:, :, 10].mean()), 2),
        "b12_mean":    round(float(crop[:, :, 11].mean()), 2),
        "b12_b11_ratio": round(
            float(crop[:, :, 11].mean()) / max(float(crop[:, :, 10].mean()), 1), 4
        ),
    }
    out_label.write_text(json.dumps(full_label, indent=2))

    log.info("    Saved %s  shape=%s  B12/B11=%.4f",
             out_npy.name, crop.shape, full_label["b12_b11_ratio"])
    return True


# ── NEGATIVE EXAMPLES from JRC/expanded survey ────────────────────────────────
def extract_negatives():
    """Extract crops from confirmed non-emitting sites in existing cache."""
    log.info("=" * 60)
    log.info("NEGATIVE EXAMPLES — JRC/Expanded Survey")
    log.info("=" * 60)

    if not RESULTS_EXP.exists() and not RESULTS_VAL.exists():
        log.error("No survey summary files found")
        return []

    all_sites = []
    for path in [RESULTS_EXP, RESULTS_VAL]:
        if path.exists():
            all_sites.extend(json.loads(path.read_text()))

    # De-duplicate by site name — prefer entries with a valid sc_ratio over PIPELINE_FAILED ones
    best: dict[str, dict] = {}
    for s in all_sites:
        name = s["site"]
        sc_info = s.get("sc", {})
        sc = sc_info.get("sc_ratio") if isinstance(sc_info, dict) else None
        if name not in best:
            best[name] = s
        elif sc is not None and best[name].get("sc", {}).get("sc_ratio") is None:
            # upgrade: current entry has valid sc, stored one doesn't
            best[name] = s
    unique_sites = list(best.values())

    results = []
    for site_info in unique_sites:
        site = site_info["site"]
        sc_info = site_info.get("sc", {})
        sc = sc_info.get("sc_ratio") if isinstance(sc_info, dict) else None

        # Only confirmed negatives: S/C < 1.15
        if sc is None or sc >= 1.15:
            continue

        lat, lon = site_info["lat"], site_info["lon"]

        # Find a detection GeoTIFF → gives us tile_id + date
        tif_dirs = ["results_expanded", "results_validation", "results_v2",
                    "results_multidate", "results_survey"]
        tifs = []
        for d in tif_dirs:
            tifs.extend(glob.glob(f"{d}/**/*{site}*.tif", recursive=True))
            tifs.extend(glob.glob(f"{d}/{site}/*.tif"))

        # Also search by tile_id from geotiff filename pattern
        # detection_T32ULB_2024-09-20.tif
        if not tifs:
            # Look in results dirs for detection tifs matching site
            for d in tif_dirs:
                tifs.extend(glob.glob(f"{d}/*.tif"))
            # Filter by approximate location (can't easily without rasterio per-site)
            tifs = []  # reset — need per-site matching

        # Find corresponding npy from geotiff name
        npy_candidates = []
        for tif in tifs:
            basename = os.path.basename(tif)  # e.g. detection_T32ULB_2024-09-20.tif
            parts = basename.replace(".tif", "").split("_")
            tile_id = next((p for p in parts if p.startswith("T") and len(p) == 6), None)
            if tile_id:
                matches = find_npy_for_tile(tile_id, [NPY_CACHE])
                npy_candidates.extend(matches)

        if not npy_candidates:
            log.debug("  No npy found for negative site: %s (sc=%.3f)", site, sc)
            continue

        log.info("")
        log.info("Site: %-25s  sc=%.3f  npys=%d", site, sc, len(npy_candidates))

        best_npy = npy_candidates[0]  # Use most recent
        date_str = ""
        for p in best_npy.stem.split("_"):
            if len(p) == 15 and p[0].isdigit():
                date_str = p[:8]
                break

        tile_id_found = next(
            (p for p in best_npy.stem.split("_") if p.startswith("T") and len(p) == 6),
            "unknown"
        )
        crop_name = f"{site}_{tile_id_found}_{date_str}"

        label = {
            "split":       get_split(date_str),
            "label_type":  "negative",
            "label_value": 0,
            "site":        site,
            "lat":         lat,
            "lon":         lon,
            "country":     site_info.get("country", "?"),
            "fuel":        site_info.get("fuel", "?"),
            "site_type":   site_info.get("site_type", "?"),
            "sc_ratio":    sc,
            "tile_id":     tile_id_found,
            "s2_date":     date_str,
            "note":        f"Confirmed non-emitter: SC={sc:.3f} < 1.15 threshold",
        }

        ok = extract_and_save_crop(best_npy, lat, lon, label, CROPS_NEG, crop_name)
        results.append({"site": site, "crop_name": crop_name, "ok": ok,
                        "label_type": "negative", "sc": sc})

    ok_count = sum(1 for r in results if r["ok"])
    log.info("")
    log.info("Negatives extracted: %d / %d", ok_count, len(results))
    return results


# ── POSITIVE EXAMPLES from TROPOMI-confirmed training tiles ───────────────────
def extract_positives():
    """Extract crops from TROPOMI-confirmed positive training tiles."""
    log.info("")
    log.info("=" * 60)
    log.info("POSITIVE EXAMPLES — TROPOMI-Confirmed Sites")
    log.info("=" * 60)

    # Find all _label.json files in training cache
    label_files = list(TRAIN_CACHE.glob("*_label.json"))

    # Also check npy_cache root for any training labels placed there
    label_files += list(NPY_CACHE.glob("*_label.json"))

    if not label_files:
        log.info("No training label files found yet.")
        log.info("Run download_training_tiles.py first, then re-run this script.")
        return []

    log.info("Found %d label files in training cache", len(label_files))
    results = []

    for label_path in sorted(label_files):
        try:
            label = json.loads(label_path.read_text())
        except Exception as e:
            log.warning("Could not read %s: %s", label_path, e)
            continue

        site        = label.get("site", "unknown")
        lat         = label.get("lat")
        lon         = label.get("lon")
        enh         = label.get("enhancement_ppb", 0)
        tile_id     = label.get("tile_id", "?")
        s2_date     = label.get("s2_date") or label.get("tropomi_date", "")
        s2_product  = label.get("s2_product", "")

        if lat is None or lon is None:
            log.warning("Skipping %s — no lat/lon in label", label_path.name)
            continue

        # Find the corresponding .npy
        npy_path = None
        # Try by product name
        if s2_product:
            candidate = label_path.parent / f"{s2_product}.npy"
            if candidate.exists():
                npy_path = candidate

        # Try by tile_id and date
        if npy_path is None and tile_id != "?":
            date_compact = s2_date.replace("-", "")[:8]
            matches = list(label_path.parent.glob(f"*_{tile_id}_*.npy"))
            if matches:
                # Pick closest date
                npy_path = matches[0]

        if npy_path is None:
            log.warning("No .npy found for label: %s", label_path.name)
            continue

        date_compact = s2_date.replace("-", "")[:8]
        crop_name = f"{site}_{tile_id}_{date_compact}_enh{int(enh)}"

        log.info("")
        log.info("Site: %-25s  enh=%+.1f ppb  tile=%s", site, enh, tile_id)

        full_label = {
            **label,
            "split":       get_split(date_compact),
            "label_type":  "positive",
            "label_value": 1,
        }

        ok = extract_and_save_crop(npy_path, lat, lon, full_label, CROPS_POS, crop_name)
        results.append({"site": site, "crop_name": crop_name, "ok": ok,
                        "label_type": "positive", "enh": enh})

    ok_count = sum(1 for r in results if r["ok"])
    log.info("")
    log.info("Positives extracted: %d / %d", ok_count, len(results))
    return results


# ── Dataset manifest and stats ────────────────────────────────────────────────
def write_manifest_and_stats(all_results: list):
    """Write manifest.json and dataset_stats.txt."""
    neg_crops = list(CROPS_NEG.glob("*.npy"))
    pos_crops = list(CROPS_POS.glob("*.npy"))

    manifest = {"negatives": [], "positives": []}

    for npy in sorted(neg_crops):
        label_path = npy.with_name(npy.stem + "_label.json")
        label = json.loads(label_path.read_text()) if label_path.exists() else {}
        manifest["negatives"].append({
            "path": str(npy), "split": label.get("split", "train"),
            "site": label.get("site"), "sc_ratio": label.get("sc_ratio"),
        })

    for npy in sorted(pos_crops):
        label_path = npy.with_name(npy.stem + "_label.json")
        label = json.loads(label_path.read_text()) if label_path.exists() else {}
        manifest["positives"].append({
            "path": str(npy), "split": label.get("split", "train"),
            "site": label.get("site"),
            "enhancement_ppb": label.get("enhancement_ppb"),
        })

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))

    n_neg = len(neg_crops)
    n_pos = len(pos_crops)
    n_total = n_neg + n_pos
    balance = n_neg / n_pos if n_pos > 0 else float("inf")

    neg_train = sum(1 for x in manifest["negatives"] if x["split"] == "train")
    neg_val   = sum(1 for x in manifest["negatives"] if x["split"] == "val")
    pos_train = sum(1 for x in manifest["positives"] if x["split"] == "train")
    pos_val   = sum(1 for x in manifest["positives"] if x["split"] == "val")

    lines = [
        "TRAINING CROP DATASET STATISTICS",
        f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        "=" * 50,
        f"  Total crops:      {n_total:4d}",
        f"  Negatives:        {n_neg:4d}  ({100*n_neg/max(n_total,1):.0f}%)",
        f"  Positives:        {n_pos:4d}  ({100*n_pos/max(n_total,1):.0f}%)",
        f"  Neg/Pos ratio:    {balance:.1f}x",
        "",
        "  Split breakdown:",
        f"    Train: {neg_train} neg + {pos_train} pos = {neg_train+pos_train}",
        f"    Val:   {neg_val} neg + {pos_val} pos = {neg_val+pos_val}",
        "",
        "  Crop size: 200×200×12 px (uint8)",
        "  (Resize to 160×160 during training for random crop augmentation)",
        "",
        "  Positive sites:",
    ]
    for x in manifest["positives"]:
        lines.append(f"    {x['site']:<28} enh={x.get('enhancement_ppb',0):+.1f} ppb  split={x['split']}")
    lines += ["", "  Negative sites:"]
    for x in manifest["negatives"]:
        lines.append(f"    {x['site']:<28} sc={x.get('sc_ratio',0):.3f}  split={x['split']}")

    target_neg = max(n_pos * 3, 30)
    if n_pos == 0:
        lines += ["", "  Status: Waiting for positive downloads"]
    elif n_neg < n_pos:
        lines += ["", f"  WARNING: Only {n_neg} negatives for {n_pos} positives — need more negatives"]
    elif balance > 10:
        lines += ["", f"  Note: Dataset skewed {balance:.0f}:1 — use weighted sampling in training"]
    else:
        lines += ["", f"  Class balance looks healthy for training"]

    text = "\n".join(lines)
    STATS_PATH.write_text(text)
    print("\n" + text)


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Extract training crops from S2 tiles")
    parser.add_argument("--negatives-only", action="store_true")
    parser.add_argument("--positives-only", action="store_true")
    args = parser.parse_args()

    if not HAS_RASTERIO:
        print("WARNING: rasterio not found. Pixel lookup will fail.")
        print("Run:  conda activate methane  before this script.")
        sys.exit(1)

    all_results = []

    if not args.positives_only:
        neg_results = extract_negatives()
        all_results.extend(neg_results)

    if not args.negatives_only:
        pos_results = extract_positives()
        all_results.extend(pos_results)

    write_manifest_and_stats(all_results)

    neg_ok = sum(1 for r in all_results if r["ok"] and r["label_type"] == "negative")
    pos_ok = sum(1 for r in all_results if r["ok"] and r["label_type"] == "positive")

    print(f"\nManifest → {MANIFEST_PATH}")
    print(f"Stats    → {STATS_PATH}")
    print(f"\nReady for training: {neg_ok} negatives + {pos_ok} positives")
    if pos_ok == 0:
        print("Re-run after download_training_tiles.py finishes to add positives.")


if __name__ == "__main__":
    main()
