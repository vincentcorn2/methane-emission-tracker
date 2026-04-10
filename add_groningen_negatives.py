"""
add_groningen_negatives.py
==========================
Extracts 4 negative training crops from the Groningen T31UGV tile.

Groningen is a persistent false positive (S/C 1.679–6.0 across all model
versions) despite TROPOMI confirming -0.99 ppb (non-emitter) on best date.
The model has never seen Dutch polder agricultural terrain during training.

This script extracts:
  - 1 crop centred on the gas field site (lat=53.252, lon=6.682)
  - 3 nearby offset crops capturing surrounding agricultural polder terrain

All 4 are labelled as negatives (label_value=0, sc_ratio < 1.15 from TROPOMI
validation). They are saved to data/crops/negative/ in the same format as
existing crops and are picked up automatically by approach_c_retrain.py.

Usage:
    cd ~/Downloads/methane-api
    python add_groningen_negatives.py
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

# ── Groningen tile ─────────────────────────────────────────────────────────────
TILE_NPY = NPY_CACHE / "S2A_MSIL1C_20240817T105031_N0511_R051_T31UGV_20240817T125303.npy"
TILE_GEO = NPY_CACHE / "S2A_MSIL1C_20240817T105031_N0511_R051_T31UGV_20240817T125303_geo.json"

TILE_ID   = "T31UGV"
S2_DATE   = "20240817"
CROP_SIZE = 200  # px — matches existing training crops (200×200×12)

# ── Crop locations ─────────────────────────────────────────────────────────────
# Centre on the gas field + 3 agricultural offsets to diversify polder terrain.
# Offsets ~0.04° ≈ 2.8 km — different enough to be independent crops, but still
# within the same T31UGV tile and same terrain type (Dutch agricultural polder).
CROP_SPECS = [
    dict(
        name="groningen_gas_field",
        lat=53.252, lon=6.682,
        note="Confirmed non-emitter: TROPOMI -0.99 ppb. FP in v5–v8 and image-level. "
             "Centre crop on gas field site (polder agricultural terrain).",
    ),
    dict(
        name="groningen_polder_N",
        lat=53.292, lon=6.682,  # ~4.5 km north
        note="Groningen polder agricultural terrain north offset (~4.5 km). "
             "Captures same spectral background causing FP; adds terrain diversity.",
    ),
    dict(
        name="groningen_polder_E",
        lat=53.252, lon=6.742,  # ~4 km east
        note="Groningen polder agricultural terrain east offset (~4 km). "
             "Adds variety in land-parcel patterns within same FP-prone region.",
    ),
    dict(
        name="groningen_polder_SW",
        lat=53.212, lon=6.622,  # ~4.5 km SW
        note="Groningen polder agricultural terrain SW offset (~4.5 km). "
             "Targets the mixed grassland/cropland that drives model over-activation.",
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
        print(f"ERROR: Tile not found: {TILE_NPY}")
        print("Run download_reference_tiles.py or check npy_cache/.")
        sys.exit(1)

    if not TILE_GEO.exists():
        print(f"ERROR: Geo sidecar not found: {TILE_GEO}")
        sys.exit(1)

    geo_meta = json.loads(TILE_GEO.read_text())
    print(f"Loading tile: {TILE_NPY.name}")
    tile = np.load(TILE_NPY, mmap_mode="r")
    print(f"  Shape: {tile.shape}  dtype: {tile.dtype}")

    saved = 0
    for spec in CROP_SPECS:
        name     = spec["name"]
        lat, lon = spec["lat"], spec["lon"]
        out_npy  = CROPS_NEG / f"{name}_{TILE_ID}_{S2_DATE}.npy"
        out_json = CROPS_NEG / f"{name}_{TILE_ID}_{S2_DATE}_label.json"

        if out_npy.exists():
            print(f"  [skip] {out_npy.name} already exists")
            saved += 1
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
            "split":         "train",   # all 4 go to train set
            "label_type":    "negative",
            "label_value":   0,
            "site":          name,
            "lat":           lat,
            "lon":           lon,
            "country":       "NL",
            "fuel":          None,
            "site_type":     "control",
            "sc_ratio":      None,      # not computed; ground truth is TROPOMI -0.99 ppb
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

    print(f"\nDone: {saved}/{len(CROP_SPECS)} Groningen negative crops ready.")
    print(f"Output dir: {CROPS_NEG.resolve()}")

    # Print updated class balance
    neg_count = len(list(CROPS_NEG.glob("*.npy")))
    pos_dir   = Path("data/crops/positive")
    pos_count = len(list(pos_dir.glob("*.npy"))) if pos_dir.exists() else "?"
    syn_dir   = Path("data/crops/synthetic")
    syn_count = len(list(syn_dir.glob("*.npy"))) if syn_dir.exists() else "?"
    print(f"\nUpdated class balance:")
    print(f"  Negative: {neg_count} crops")
    print(f"  Positive: {pos_count} crops (real)")
    print(f"  Synthetic: {syn_count} crops")


if __name__ == "__main__":
    main()
