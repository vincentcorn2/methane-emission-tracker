"""
Approach B: Re-threshold saved GeoTIFFs at multiple thresholds.
Tests whether any threshold separates the emission area from the clean area.

Key question: does the emission GeoTIFF (T40SBJ, Balkanabat, TROPOMI-confirmed high emission)
show MORE above-threshold pixels than the clean GeoTIFF (T40SDH, Magtymguly, TROPOMI-clean)
at any threshold? If yes, we've found the model's actual operating point.

Run with:
  conda activate methane && python approach_b_rethreshold.py
"""

import numpy as np
import rasterio
from scipy import ndimage
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

files = {
    "T40SBJ_2024-06-25 [EMISSION, Balkanabat]": os.path.join(RESULTS_DIR, "detection_T40SBJ_2024-06-25.tif"),
    "T40SDH_2024-06-24 [CLEAN, Magtymguly]":    os.path.join(RESULTS_DIR, "detection_T40SDH_2024-06-24.tif"),
}

thresholds = [0.10, 0.18, 0.30, 0.50, 0.70, 0.90]

print("=" * 80)
print("APPROACH B: Multi-threshold probability map analysis")
print("=" * 80)

results = {}

for label, path in files.items():
    print(f"\n{'─'*70}")
    print(f"File: {label}")
    if not os.path.exists(path):
        print(f"  ERROR: File not found: {path}")
        continue
    with rasterio.open(path) as src:
        prob = src.read(1).astype(np.float32)
        print(f"  Shape: {prob.shape}  |  dtype: {prob.dtype}")
        print(f"  Prob range: [{prob.min():.4f}, {prob.max():.4f}]")
        print(f"  Mean: {prob.mean():.6f}  |  Std: {prob.std():.6f}")

    print(f"\n  {'Threshold':>10}  {'Pixels above':>14}  {'% of tile':>10}  {'Blobs(>=115px)':>16}  {'Max blob px':>12}")
    print(f"  {'-'*10}  {'-'*14}  {'-'*10}  {'-'*16}  {'-'*12}")

    row = {}
    for t in thresholds:
        binary = prob >= t
        n_pixels = int(binary.sum())
        pct = 100.0 * n_pixels / binary.size
        labeled, n_blobs = ndimage.label(binary)
        if n_blobs > 0:
            sizes = np.bincount(labeled.ravel())[1:]
            big_blobs = int((sizes >= 115).sum())
            max_blob = int(sizes.max())
        else:
            big_blobs = 0
            max_blob = 0
        row[t] = {"pixels": n_pixels, "pct": pct, "big_blobs": big_blobs, "max_blob": max_blob}
        print(f"  {t:>10.2f}  {n_pixels:>14,}  {pct:>9.3f}%  {big_blobs:>16,}  {max_blob:>12,}")

    results[label] = row

# ── Comparison table ──────────────────────────────────────────────────────────
emission_key = "T40SBJ_2024-06-25 [EMISSION, Balkanabat]"
clean_key    = "T40SDH_2024-06-24 [CLEAN, Magtymguly]"

if emission_key in results and clean_key in results:
    print(f"\n\n{'='*80}")
    print("COMPARISON: emission area vs. clean area at each threshold")
    print(f"{'='*80}")
    print(f"\n  {'Threshold':>10}  {'Emission px':>14}  {'Clean px':>14}  {'Emission/Clean':>16}  {'Signal?':>22}")
    print(f"  {'-'*10}  {'-'*14}  {'-'*14}  {'-'*16}  {'-'*22}")

    for t in thresholds:
        ep = results[emission_key][t]["pixels"]
        cp = results[clean_key][t]["pixels"]
        ratio = ep / cp if cp > 0 else float("inf")
        signal = "✓ EMISSION > CLEAN" if ep > cp else "✗ clean >= emission"
        print(f"  {t:>10.2f}  {ep:>14,}  {cp:>14,}  {ratio:>16.3f}  {signal}")

    print(f"\n{'='*80}")
    print("INTERPRETATION")
    print(f"{'='*80}")
    print("""
If emission/clean ratio is consistently < 1.0 at ALL thresholds:
  → Model is a terrain/surface detector, not a methane detector.
  → Weights are confounded. Need retraining (Approach C) or replacement (MARS-S2L).

If ratio > 1.0 at any threshold (especially high ones like 0.7-0.9):
  → Model has weak but real methane signal; threshold re-calibration may help.
  → Try running the pipeline at that threshold.

If ratio ≈ 1.0 across all thresholds:
  → Outputs are spatially random (the model learned nothing useful from the data).
""")

# ── Also check the 2021 tile (different season, sanity check) ─────────────────
tif_2021 = os.path.join(RESULTS_DIR, "detection_T40SBJ_2021-01-29.tif")
if os.path.exists(tif_2021):
    print(f"\n{'─'*70}")
    print("BONUS: T40SBJ 2021-01-29 (different season, should differ from 2024)")
    with rasterio.open(tif_2021) as src:
        prob = src.read(1).astype(np.float32)
    for t in [0.18, 0.50]:
        n = int((prob >= t).sum())
        print(f"  Threshold {t:.2f}: {n:,} pixels ({100*n/prob.size:.2f}%)")
