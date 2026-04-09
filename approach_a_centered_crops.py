"""
Approach A: Run centered 200×200 pixel crops on the paper's known emitter coordinates.
Tests whether confidence is spatially concentrated near infrastructure vs. uniform noise.

Sites from Vaughan et al. 2024 AMT Table 1 (confirmed active emitters):
  T6:  lat=39.4616,  lon=53.77502  → UTM Zone 39 → T39SYD
  T7:  lat=39.45965, lon=53.77921  → UTM Zone 39 → T39SYD
  T14: lat=38.55747, lon=54.20049  → UTM Zone 40 → T40SBJ
  T17: lat=38.49393, lon=54.19764  → UTM Zone 40 → T40SBJ

NOTE: T6/T7 are at lon~53.77°E which is in UTM Zone 39 (48–54°E), not Zone 40.
The previous run incorrectly used T40SBJ for all four sites; T6/T7 need T39SYD.

Run with:
  conda activate methane && python approach_a_centered_crops.py
"""

import numpy as np
import rasterio
from rasterio.transform import rowcol
from pyproj import Transformer
from scipy import ndimage
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# Each entry: (lat, lon, geotiff_filename, tile_id)
# T6/T7 → T39SYD (UTM Zone 39),  T14/T17 → T40SBJ (UTM Zone 40)
SITES = {
    "T6  (confirmed emitter)":  (39.4616,  53.77502, "detection_T39SYD_2024-06-25.tif", "T39SYD"),
    "T7  (confirmed emitter)":  (39.45965, 53.77921, "detection_T39SYD_2024-06-25.tif", "T39SYD"),
    "T14 (confirmed emitter)":  (38.55747, 54.20049, "detection_T40SBJ_2024-06-25.tif", "T40SBJ"),
    "T17 (confirmed emitter)":  (38.49393, 54.19764, "detection_T40SBJ_2024-06-25.tif", "T40SBJ"),
}

CONTROLS = {
    "T6  CONTROL (+2km N)":   (39.4796,  53.77502, "detection_T39SYD_2024-06-25.tif", "T39SYD"),
    "T7  CONTROL (+2km N)":   (39.4776,  53.77921, "detection_T39SYD_2024-06-25.tif", "T39SYD"),
    "T14 CONTROL (+2km N)":   (38.5754,  54.20049, "detection_T40SBJ_2024-06-25.tif", "T40SBJ"),
    "T17 CONTROL (+2km N)":   (38.5119,  54.19764, "detection_T40SBJ_2024-06-25.tif", "T40SBJ"),
}

CROP_SIZE = 200   # pixels = 2km at 10m resolution
THRESHOLD = 0.18

# Cache open tiles so we don't re-read the same file repeatedly
_tile_cache = {}

def get_tile(filename):
    if filename not in _tile_cache:
        path = os.path.join(RESULTS_DIR, filename)
        if not os.path.exists(path):
            return None, None, None
        with rasterio.open(path) as src:
            prob = src.read(1).astype(np.float32)
            transform = src.transform
            crs = src.crs
        _tile_cache[filename] = (prob, transform, crs)
    return _tile_cache[filename]


def is_in_bounds(prob_array, row, col, half):
    """Return True if a crop centered at (row, col) is fully inside the array."""
    h, w = prob_array.shape
    return (row - half >= 0 and row + half <= h and
            col - half >= 0 and col + half <= w)


def extract_crop(prob_array, transform, crs, lat, lon, crop_size=200):
    """Extract a crop_size x crop_size window centered on (lat, lon)."""
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    row, col = rowcol(transform, x, y)

    half = crop_size // 2
    in_bounds = is_in_bounds(prob_array, row, col, half)

    r0 = max(0, row - half)
    r1 = min(prob_array.shape[0], row + half)
    c0 = max(0, col - half)
    c1 = min(prob_array.shape[1], col + half)

    crop = prob_array[r0:r1, c0:c1]
    return crop, (row, col, r0, r1, c0, c1), in_bounds


def crop_stats(crop, threshold=0.18):
    if crop.size == 0:
        return {"mean": 0, "max": 0, "pct_above": 0, "n_blobs": 0, "largest_blob": 0}
    binary = crop >= threshold
    labeled, n = ndimage.label(binary)
    sizes = np.bincount(labeled.ravel())[1:] if n > 0 else np.array([])
    return {
        "mean":         float(crop.mean()),
        "max":          float(crop.max()),
        "pct_above":    float(100.0 * binary.sum() / binary.size),
        "n_blobs":      int(n),
        "largest_blob": int(sizes.max()) if len(sizes) > 0 else 0,
    }


print("=" * 80)
print("APPROACH A (v2): Centered 200×200 crops — correct tile per site")
print("=" * 80)

# Print per-tile summary first
for fname in sorted(set(v[2] for v in list(SITES.values()) + list(CONTROLS.values()))):
    prob, transform, crs = get_tile(fname)
    if prob is None:
        print(f"\n  MISSING: {fname}")
    else:
        print(f"\n  Tile {fname}: shape={prob.shape}, "
              f"mean={prob.mean():.6f}, range=[{prob.min():.4f},{prob.max():.4f}]")

print(f"\n{'─'*90}")
print(f"{'Location':<32}  {'Tile':>8}  {'Row,Col':>12}  {'InBnds':>6}  "
      f"{'Mean':>10}  {'Max':>8}  {'%>0.18':>7}  {'Blobs':>6}  {'Largest':>8}")
print(f"{'─'*32}  {'─'*8}  {'─'*12}  {'─'*6}  {'─'*10}  {'─'*8}  {'─'*7}  {'─'*6}  {'─'*8}")

all_results = {}
all_entries = {**SITES, **CONTROLS}

for label, (lat, lon, fname, tile_id) in all_entries.items():
    prob, transform, crs = get_tile(fname)
    if prob is None:
        print(f"{label:<32}  {tile_id:>8}  {'N/A':>12}  {'N/A':>6}  FILE NOT FOUND")
        continue
    try:
        crop, (row, col, r0, r1, c0, c1), in_bounds = extract_crop(
            prob, transform, crs, lat, lon, CROP_SIZE)
        stats = crop_stats(crop, THRESHOLD)
        all_results[label] = stats
        ib_str = "YES" if in_bounds else "EDGE"
        print(f"{label:<32}  {tile_id:>8}  {row:>5},{col:>5}   {ib_str:>6}  "
              f"{stats['mean']:>10.6f}  {stats['max']:>8.4f}  "
              f"{stats['pct_above']:>6.2f}%  {stats['n_blobs']:>6,}  {stats['largest_blob']:>8,}")
    except Exception as e:
        print(f"{label:<32}  ERROR: {e}")

print(f"\n{'='*80}")
print("COMPARISON: Site vs. matched control")
print(f"{'='*80}")

site_keys = list(SITES.keys())
ctrl_keys = list(CONTROLS.keys())

print(f"\n  {'Site':<28}  {'Tile':>8}  {'Site mean':>10}  {'Ctrl mean':>10}  {'S/C ratio':>10}  {'Signal?'}")
print(f"  {'─'*28}  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*20}")

for (sk, sv), (ck, cv) in zip(SITES.items(), CONTROLS.items()):
    sm = all_results.get(sk, {}).get("mean", None)
    cm = all_results.get(ck, {}).get("mean", None)
    if sm is None or cm is None:
        print(f"  {sk.split('(')[0].strip():<28}  {sv[3]:>8}  MISSING DATA")
        continue
    ratio = sm / cm if cm > 0 else float("inf")
    if sm > cm * 1.5:
        signal = "✓✓ STRONG site > control"
    elif sm > cm:
        signal = "✓  site > control"
    else:
        signal = "✗  control >= site"
    name = sk.split("(")[0].strip()
    print(f"  {name:<28}  {sv[3]:>8}  {sm:>10.6f}  {cm:>10.6f}  {ratio:>10.3f}  {signal}")

print(f"""
{'='*80}
INTERPRETATION
{'='*80}

Signal ratio > 1.5 at any site: model has spatial specificity near emitters.
Signal ratio ≈ 1.0 everywhere:  model fires uniformly on terrain type, not plumes.
Site mean << tile mean:          site locations actively suppressed (very bad sign).
""")

# Print tile means for context
print("Tile means (for reference):")
for fname in sorted(set(v[2] for v in list(SITES.values()))):
    prob, _, _ = get_tile(fname)
    if prob is not None:
        print(f"  {fname}: mean={prob.mean():.6f}")
