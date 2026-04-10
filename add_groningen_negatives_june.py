"""
add_groningen_negatives_june.py
================================
Extracts 6 negative training crops from the Groningen T31UGV June tile.

Why June in addition to August?
--------------------------------
After v9 training, Groningen S/C dropped from 6.0 (v8) → 3.348 (v9), but is
still above the 1.15 detection threshold.  The 4 August crops were not enough
to fully suppress the false positive.  Adding a second seasonal acquisition
(June 2024) teaches the model what Dutch polder terrain looks like in early
summer (greener vegetation, different NDVI) vs late summer (drier).

This script extracts:
  - 4 crops at the same lat/lon positions as the August set (cross-seasonal pair)
  - 2 new positions (NW and SE offsets) to broaden spatial coverage

All 6 are labelled as negatives (label_value=0, ground-truth: TROPOMI -0.99 ppb).
They are saved to data/crops/negative/ in the same format as existing crops and
picked up automatically by approach_c_retrain.py.

Usage:
    cd ~/Downloads/methane-api
    conda activate methane
    python add_groningen_negatives_june.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
import rasterio.warp
import rasterio.transform

# ── Paths ─────────────────────────────────────────────────────────────────────
NPY_CACHE  = Path("data/npy_cache")
CROPS_NEG  = Path("data/crops/negative")
CROPS_NEG.mkdir(parents=True, exist_ok=True)

# ── June tile ─────────────────────────────────────────────────────────────────
TILE_NPY = NPY_CACHE / "S2A_MSIL1C_20240628T105031_N0510_R051_T31UGV_20240628T125124.npy"
TILE_GEO = NPY_CACHE / "S2A_MSIL1C_20240628T105031_N0510_R051_T31UGV_20240628T125124_geo.json"

TILE_ID   = "T31UGV"
S2_DATE   = "20240628"
CROP_SIZE = 200  # px — matches existing training crops (200×200×12)

# ── Crop locations ─────────────────────────────────────────────────────────────
# 4 positions matching the August set (cross-seasonal pair for each)
# + 2 new positions (NW, SE) for broader spatial coverage.
# Offsets ~0.04°–0.06° ≈ 2.8–4.5 km — independent crops, same terrain type.
CROP_SPECS = [
    dict(
        name="groningen_gas_field",
        lat=53.252, lon=6.682,
        note="June seasonal counterpart to August crop. Confirmed non-emitter: "
             "TROPOMI -0.99 ppb. Centre crop on gas field site (polder agricultural terrain).",
    ),
    dict(
        name="groningen_polder_N",
        lat=53.292, lon=6.682,  # ~4.5 km north — same as August
        note="June seasonal counterpart to August polder_N crop. "
             "Polder agricultural terrain north offset (~4.5 km).",
    ),
    dict(
        name="groningen_polder_E",
        lat=53.252, lon=6.742,  # ~4 km east — same as August
        note="June seasonal counterpart to August polder_E crop. "
             "Polder agricultural terrain east offset (~4 km).",
    ),
    dict(
        name="groningen_polder_SW",
        lat=53.212, lon=6.622,  # ~4.5 km SW — same as August
        note="June seasonal counterpart to August polder_SW crop. "
             "Mixed grassland/cropland SW offset (~4.5 km).",
    ),
    # ── New positions ─────────────────────────────────────────────────────────
    dict(
        name="groningen_polder_NW",
        lat=53.292, lon=6.622,  # ~4.5 km NW (new)
        note="New June-only crop NW of gas field. Adds spatial diversity to "
             "the training set for Dutch polder terrain at ~4.5 km NW.",
    ),
    dict(
        name="groningen_polder_SE",
        lat=53.212, lon=6.742,  # ~4.5 km SE (new)
        note="New June-only crop SE of gas field. Adds spatial diversity to "
             "the training set for Dutch polder terrain at ~4.5 km SE.",
    ),
]

# ── Pixel lookup via geo sidecar + rasterio ────────────────────────────────────
def latlon_to_pixel(geo_meta: dict, lat: float, lon: float) -> tuple[int, int]:
    crs = geo_meta["crs"]
    vals = geo_meta["transform"]
    aff = rasterio.transform.Affine(vals[0], vals[1], vals[2],
                                    vals[3], vals[4], vals[5])
    xs, ys = rasterio.warp.transform("EPSG:4326", crs, [lon], [lat])
    row, col = rasterio.transform.rowcol(aff, xs[0], ys[0])
    return int(row), int(col)


def safe_crop(arr: np.ndarray, row: int, col: int, size: int = CROP_SIZE):
    H, W = arr.shape[:2]
    half = size // 2
    r0, r1 = row - half, row + half
    c0, c1 = col - half, col + half
    if r0 < 0 or r1 > H or c0 < 0 or c1 > W:
        return None
    return arr[r0:r1, c0:c1].copy()


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    if not TILE_NPY.exists():
        print(f"ERROR: June tile not found: {TILE_NPY}")
        print("Expected: data/npy_cache/S2A_MSIL1C_20240628T105031_N0510_R051_T31UGV_20240628T125124.npy")
        print("Run download_reference_tiles.py or check npy_cache/.")
        sys.exit(1)

    if not TILE_GEO.exists():
        print(f"ERROR: Geo sidecar not found: {TILE_GEO}")
        print(f"Expected: {TILE_GEO}")
        sys.exit(1)

    geo_meta = json.loads(TILE_GEO.read_text())
    print(f"Loading June tile: {TILE_NPY.name}")
    tile = np.load(TILE_NPY, mmap_mode="r")
    print(f"  Shape: {tile.shape}  dtype: {tile.dtype}")

    saved = 0
    skipped = 0
    for spec in CROP_SPECS:
        name     = spec["name"]
        lat, lon = spec["lat"], spec["lon"]
        out_npy  = CROPS_NEG / f"{name}_{TILE_ID}_{S2_DATE}.npy"
        out_json = CROPS_NEG / f"{name}_{TILE_ID}_{S2_DATE}_label.json"

        if out_npy.exists():
            print(f"  [skip] {out_npy.name} already exists")
            skipped += 1
            continue

        row, col = latlon_to_pixel(geo_meta, lat, lon)
        print(f"  {name}: lat={lat} lon={lon} → pixel ({row}, {col})")

        crop = safe_crop(tile, row, col)
        if crop is None:
            print(f"    WARNING: crop out of bounds — skipping")
            continue

        np.save(out_npy, crop)

        b11_mean = float(crop[:, :, 10].mean())
        b12_mean = float(crop[:, :, 11].mean())

        label = {
            "split":         "train",   # all 6 go to train set
            "label_type":    "negative",
            "label_value":   0,
            "site":          name,
            "lat":           lat,
            "lon":           lon,
            "country":       "NL",
            "fuel":          None,
            "site_type":     "control",
            "sc_ratio":      None,      # ground truth: TROPOMI -0.99 ppb
            "tile_id":       TILE_ID,
            "s2_date":       S2_DATE,
            "note":          spec["note"],
            "crop_size":     list(crop.shape),
            "pixel_row":     row,
            "pixel_col":     col,
            "npy_source":    str(TILE_NPY),
            "extracted":     datetime.now(timezone.utc).isoformat(),
            "b11_mean":      round(b11_mean, 2),
            "b12_mean":      round(b12_mean, 2),
            "b12_b11_ratio": round(b12_mean / max(b11_mean, 1), 4),
        }
        out_json.write_text(json.dumps(label, indent=2))

        print(f"    Saved {out_npy.name}  shape={crop.shape}"
              f"  B12/B11={label['b12_b11_ratio']:.4f}")
        saved += 1

    print(f"\nDone: {saved} new + {skipped} already-existing = "
          f"{saved + skipped}/{len(CROP_SPECS)} June Groningen negative crops ready.")
    print(f"Output dir: {CROPS_NEG.resolve()}")

    # Print updated class balance
    neg_count = len([p for p in CROPS_NEG.glob("*.npy")
                     if "_label" not in p.stem])
    pos_dir   = Path("data/crops/positive")
    pos_count = len([p for p in pos_dir.glob("*.npy")
                     if "_label" not in p.stem]) if pos_dir.exists() else "?"
    syn_dir   = Path("data/crops/synthetic")
    syn_count = len([p for p in syn_dir.glob("*.npy")
                     if "_label" not in p.stem]) if syn_dir.exists() else "?"
    print(f"\nUpdated class balance (before synthetic regen):")
    print(f"  Negative: {neg_count} crops")
    print(f"  Positive: {pos_count} crops (real)")
    print(f"  Synthetic: {syn_count} crops (stale — regenerate next)")
    print()
    print("Next steps:")
    print("  python generate_synthetic_plumes.py --per-crop 3 --clear")
    print("  python approach_c_retrain.py")


if __name__ == "__main__":
    main()
