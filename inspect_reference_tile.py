"""
inspect_reference_tile.py
=========================
Diagnostic: inspect B11/B12 values at the Weisweiler plant pixel
in both the winter reference tile and the summer target tile.

Answers the question:
  "Is the T31UGS December reference tile contaminated at Weisweiler,
   causing bitemporal differencing to cancel out the methane signal?"

Expected healthy output:
  - Reference B12 and target B12 should both be non-zero (~80-120 DN)
  - delta_B12 = target - reference should be NEGATIVE (plant absorbs in summer)
  - delta_B11 ≈ 0 (seasonal change similar in both bands → terrain, not gas)
  - If delta_B12 < delta_B11 by a meaningful margin → real methane signal

Red flags indicating a contaminated reference:
  - Reference B12 very low (<30) → snow/frost/cloud shadow in December
  - delta_B12 close to 0 or positive → signal being cancelled by BT subtraction
  - Reference pixel all zeros → wrong tile or zero-padded region

Usage:
    conda activate methane
    python inspect_reference_tile.py
"""

import sys
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass

# ── Config ─────────────────────────────────────────────────────────────────────
WEISWEILER_LAT = 50.837
WEISWEILER_LON = 6.322
TILE_ID        = "T31UGS"
NPY_CACHE      = Path("data/npy_cache")

# Band indices (0-based, matching BAND_CONFIG in preprocessing.py)
B11_IDX = 10   # 1610 nm SWIR reference
B12_IDX = 11   # 2190 nm methane absorption

# Neighbourhood half-size for context around pixel (pixels, 20m resolution)
HALF = 5   # → 10×10 pixel window = 200m × 200m

# ── Geo utilities ───────────────────────────────────────────────────────────────

def latlon_to_rowcol(transform: list, crs: str, lat: float, lon: float):
    """
    Convert WGS-84 lat/lon → row, col in a tile using its affine transform.
    Uses rasterio for the CRS projection.
    """
    import rasterio.warp
    import rasterio.transform

    # transform is [a, b, c, d, e, f] → Affine(a,b,c,d,e,f)
    from rasterio.transform import Affine
    a, b, c, d, e, f = transform
    aff = Affine(a, b, c, d, e, f)

    xs, ys = rasterio.warp.transform("EPSG:4326", crs, [lon], [lat])
    row, col = rasterio.transform.rowcol(aff, xs[0], ys[0])
    return int(row), int(col)


def load_geo_meta(npy_path: Path):
    """Load JSON sidecar for a .npy file."""
    stem = npy_path.stem
    candidates = list(npy_path.parent.glob(f"{stem}*_geo.json"))
    if not candidates:
        # Also try exact stem + _geo.json
        candidate = npy_path.with_suffix("").with_suffix("_geo.json")
        if npy_path.with_name(stem + "_geo.json").exists():
            candidates = [npy_path.with_name(stem + "_geo.json")]
    if not candidates:
        return None
    with open(candidates[0]) as f:
        return json.load(f)


# ── Tile discovery ─────────────────────────────────────────────────────────────

def find_tile(tile_id: str, is_reference: bool) -> Path | None:
    """Find target or reference .npy for a tile."""
    if is_reference:
        candidates = list(NPY_CACHE.glob(f"{tile_id}_ref_*.npy"))
    else:
        candidates = [
            p for p in NPY_CACHE.glob(f"*_{tile_id}_*.npy")
            if "_ref_" not in p.name and "_bitemporal" not in p.name
        ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


# ── Pixel inspection ───────────────────────────────────────────────────────────

def inspect_pixel(arr: np.ndarray, row: int, col: int, label: str):
    """Print B11/B12 values and neighbourhood stats at a pixel."""
    H, W, C = arr.shape
    r0 = max(0, row - HALF)
    r1 = min(H, row + HALF)
    c0 = max(0, col - HALF)
    c1 = min(W, col + HALF)

    b11_pixel = int(arr[row, col, B11_IDX])
    b12_pixel = int(arr[row, col, B12_IDX])
    b11_window = arr[r0:r1, c0:c1, B11_IDX].astype(float)
    b12_window = arr[r0:r1, c0:c1, B12_IDX].astype(float)
    zero_frac = float((arr[row-HALF:row+HALF, col-HALF:col+HALF, :] == 0).mean())

    ratio = b12_pixel / max(b11_pixel, 1)

    print(f"\n  [{label}]")
    print(f"    Pixel ({row}, {col}):")
    print(f"      B11 = {b11_pixel:3d} DN    B12 = {b12_pixel:3d} DN    B12/B11 = {ratio:.3f}")
    print(f"    {HALF*2}×{HALF*2} window (200m × 200m):")
    print(f"      B11: mean={b11_window.mean():.1f}  min={b11_window.min():.0f}  max={b11_window.max():.0f}")
    print(f"      B12: mean={b12_window.mean():.1f}  min={b12_window.min():.0f}  max={b12_window.max():.0f}")
    print(f"    Zero-pixel fraction in window: {zero_frac:.1%}")

    if zero_frac > 0.5:
        print(f"    ⚠️  WARNING: >50% zero pixels — tile may be padded/missing data here")
    if b12_pixel < 30 and zero_frac < 0.5:
        print(f"    ⚠️  WARNING: B12={b12_pixel} is very low — possible snow/cloud/frost in reference")
    if b12_pixel == 0 and b11_pixel == 0:
        print(f"    ❌  ERROR: Both bands zero — this pixel is in a no-data region")

    return {
        "row": row, "col": col,
        "b11_pixel": b11_pixel,
        "b12_pixel": b12_pixel,
        "b12_b11_ratio": round(ratio, 3),
        "b11_window_mean": round(float(b11_window.mean()), 1),
        "b12_window_mean": round(float(b12_window.mean()), 1),
        "zero_frac": round(zero_frac, 3),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Weisweiler Reference Tile Inspection — CH4Net v2")
    print(f"  Tile: {TILE_ID}   Lat={WEISWEILER_LAT}  Lon={WEISWEILER_LON}")
    print("=" * 65)

    # ── 1. Find tiles ──────────────────────────────────────────────────────────
    ref_path = find_tile(TILE_ID, is_reference=True)
    tgt_path = find_tile(TILE_ID, is_reference=False)

    if ref_path is None:
        print(f"\n❌  No reference tile found for {TILE_ID} in {NPY_CACHE}/")
        print(f"    Run:  python download_reference_tiles.py --sites weisweiler")
        sys.exit(1)

    if tgt_path is None:
        print(f"\n❌  No target tile found for {TILE_ID} in {NPY_CACHE}/")
        sys.exit(1)

    print(f"\n  Reference: {ref_path.name}")
    print(f"  Target:    {tgt_path.name}")

    # ── 2. Load geo metadata ───────────────────────────────────────────────────
    geo = load_geo_meta(ref_path)
    if geo is None:
        # Fall back to target geo metadata (same tile, same transform)
        geo = load_geo_meta(tgt_path)
    if geo is None:
        print("\n❌  No _geo.json sidecar found for either tile.")
        print("    Cannot convert lat/lon → pixel without geo metadata.")
        sys.exit(1)

    print(f"\n  CRS:       {geo['crs']}")
    print(f"  Transform: {geo['transform']}")

    # ── 3. Convert lat/lon → pixel ─────────────────────────────────────────────
    try:
        row, col = latlon_to_rowcol(
            geo["transform"], geo["crs"],
            WEISWEILER_LAT, WEISWEILER_LON,
        )
    except Exception as e:
        print(f"\n❌  Coordinate conversion failed: {e}")
        sys.exit(1)

    print(f"\n  Weisweiler pixel: row={row}, col={col}")
    print(f"  Tile dimensions:  {geo['height']} rows × {geo['width']} cols")

    if not (0 <= row < geo["height"] and 0 <= col < geo["width"]):
        print(f"\n❌  PIXEL OUT OF BOUNDS — Weisweiler is not in tile {TILE_ID}")
        print("    Check tile_id assignment in apply_bitemporal_diff.py")
        sys.exit(1)

    # ── 4. Load arrays (may be ~1.4 GB each) ──────────────────────────────────
    print(f"\n  Loading reference array (this may take a moment)...")
    ref_arr = np.load(ref_path, mmap_mode="r")
    print(f"  Reference shape: {ref_arr.shape}  dtype: {ref_arr.dtype}")

    print(f"  Loading target array...")
    tgt_arr = np.load(tgt_path, mmap_mode="r")
    print(f"  Target shape: {tgt_arr.shape}  dtype: {tgt_arr.dtype}")

    # ── 5. Inspect pixels ──────────────────────────────────────────────────────
    ref_stats = inspect_pixel(ref_arr, row, col, f"REFERENCE  ({ref_path.name[:30]}...)")
    tgt_stats = inspect_pixel(tgt_arr, row, col, f"TARGET     ({tgt_path.name[:30]}...)")

    # ── 6. Compute BT delta ────────────────────────────────────────────────────
    print(f"\n  {'─'*60}")
    print(f"  BITEMPORAL DELTA at Weisweiler pixel:")

    delta_b11 = int(tgt_arr[row, col, B11_IDX]) - int(ref_arr[row, col, B11_IDX])
    delta_b12 = int(tgt_arr[row, col, B12_IDX]) - int(ref_arr[row, col, B12_IDX])
    shifted_b11 = max(0, min(255, delta_b11 + 128))
    shifted_b12 = max(0, min(255, delta_b12 + 128))

    print(f"    delta_B11 = {delta_b11:+4d}  → shifted = {shifted_b11:3d}  (neutral=128)")
    print(f"    delta_B12 = {delta_b12:+4d}  → shifted = {shifted_b12:3d}  (neutral=128)")
    print(f"    B12 - B11 delta differential = {delta_b12 - delta_b11:+4d}")

    print(f"\n  INTERPRETATION:")
    if ref_stats["b12_pixel"] == 0 or ref_stats["zero_frac"] > 0.5:
        print(f"  ❌  Reference pixel is zero/padded — BT differencing is invalid here.")
        print(f"     Action: find a different reference date or use baseline-only mode.")
    elif delta_b12 < -10 and abs(delta_b12 - delta_b11) > 10:
        print(f"  ✅  B12 drops more than B11 between winter and summer.")
        print(f"     This is consistent with a real methane absorption signal.")
        print(f"     The BT approach SHOULD preserve Weisweiler detection.")
        print(f"     → If S/C is still <1.15, the model weights may need adjustment.")
    elif abs(delta_b12 - delta_b11) < 5:
        print(f"  ⚠️  B12 and B11 delta are nearly equal ({delta_b12:+d} vs {delta_b11:+d}).")
        print(f"     After differencing, the model sees no SWIR anomaly at Weisweiler.")
        print(f"     This explains the S/C attenuation seen in the eval results.")
        print(f"     → Root cause: seasonal B12 change at this site is indistinguishable")
        print(f"       from terrain change (not methane-specific).")
        print(f"     → Action: try a different reference acquisition date, or use")
        print(f"       baseline (non-BT) inference for Weisweiler.")
    elif delta_b12 > 10:
        print(f"  ⚠️  B12 is BRIGHTER in summer than in winter (delta={delta_b12:+d}).")
        print(f"     This is the opposite of a methane signal (methane → darker B12).")
        print(f"     → Reference B12 may have been depressed by frost/snow in December.")
        print(f"     → BT differencing inverts the expected signal at this site.")
        print(f"     → Action: find a reference tile from late January or February.")
    else:
        print(f"  ℹ️  delta_B12={delta_b12:+d}, delta_B11={delta_b11:+d} — marginal signal.")
        print(f"     The 18 ppb enhancement may be near the Sentinel-2 noise floor.")

    # ── 7. Sample all 12 bands at the pixel for full context ──────────────────
    print(f"\n  ALL 12 BANDS at Weisweiler pixel:")
    band_names = ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B11","B12"]
    print(f"  {'Band':<5} {'Reference':>10} {'Target':>10} {'Delta':>8}")
    print(f"  {'─'*37}")
    for i, name in enumerate(band_names):
        r_val = int(ref_arr[row, col, i])
        t_val = int(tgt_arr[row, col, i])
        d_val = t_val - r_val
        flag = " ← BT channel" if i in (B11_IDX, B12_IDX) else ""
        print(f"  {name:<5} {r_val:>10} {t_val:>10} {d_val:>+8}{flag}")

    # ── 8. Check 5 nearby pixels as spatial context ────────────────────────────
    print(f"\n  SPATIAL CONTEXT — B12 values in 3×3 window around Weisweiler pixel:")
    print(f"  (Reference / Target)")
    for dr in range(-2, 3):
        row_vals = []
        for dc in range(-2, 3):
            rr, cc = row + dr, col + dc
            if 0 <= rr < geo["height"] and 0 <= cc < geo["width"]:
                rv = int(ref_arr[rr, cc, B12_IDX])
                tv = int(tgt_arr[rr, cc, B12_IDX])
                row_vals.append(f"{rv:3d}/{tv:3d}")
            else:
                row_vals.append(" OOB ")
        center = "◄" if dr == 0 else " "
        print(f"  {'  '.join(row_vals)} {center}")

    print(f"\n{'='*65}")
    print(f"  Done. Use these values to decide:")
    print(f"  1. Is the reference tile clean at Weisweiler? (no snow/cloud)")
    print(f"  2. Does delta_B12 differ meaningfully from delta_B11?")
    print(f"  3. Should BT differencing be disabled for Weisweiler?")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
