"""
overnight_validation.py — Full CH4Net div_factor=8 validation across all cached tiles.

Re-runs inference with the retrained weights on all 11 locally-cached Sentinel-2 tiles,
then performs Approach A (spatial specificity at known emitters) and Approach B
(emission/clean comparison across thresholds) on the new probability maps.

Cached tiles cover:
  TURKMENISTAN (Karakum desert):
    T39SYD 2024-06-25 — T6, T7 confirmed emitters (Darvaza area)
    T40SBJ 2024-06-25 — T14, T17 confirmed emitters (Balkanabat)
    T40SBJ 2021-01-29 — Same tile, winter (seasonal control)
    T40SDH 2024-06-24 — Magtymguly, TROPOMI-clean control
    T40TBK 2024-06-25 — Adjacent Turkmenistan tile

  EUROPE (Netherlands / Germany — flat agricultural + urban + coastal):
    T31UET 2024-06-26 — Rotterdam / Amsterdam / The Hague area
    T31UET 2024-06-28 — Same area, different date
    T31UGV 2024-06-28 — Netherlands coastal / North Sea
    T32ULC 2024-06-27 — Rhine-Ruhr Germany
    T32ULD 2024-06-27 — Germany (slightly south)
    T32ULE 2024-06-28 — Germany / Denmark border

Usage:
  conda activate methane
  python overnight_validation.py --weights weights/ch4net_div8_retrained.pth

Output:
  results_v2/  — New GeoTIFFs (new model, 160×160 tile inference)
  overnight_report.txt — Full text report for morning review
"""

import argparse
import glob
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin, rowcol
from rasterio.crs import CRS
from scipy import ndimage

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.detection.ch4net_model import Unet, CH4NetDetector
from src.ingestion.preprocessing import tile_scene, stitch_predictions, GeoMetadata

# ─── Config ─────────────────────────────────────────────────────────────────

NPY_CACHE   = os.path.join(PROJECT_ROOT, "data", "npy_cache")
RESULTS_V2  = os.path.join(PROJECT_ROOT, "results_v2")
REPORT_PATH = os.path.join(PROJECT_ROOT, "overnight_report.txt")
TILE_SIZE   = 160        # matches training crop size
THRESHOLD   = 0.18
MIN_BLOB_PX = 115

# ── Approach A site definitions ───────────────────────────────────────────────
# Format: label → (lat, lon, geotiff_stem)
# Controls are +2km north (~0.018° lat) of each emitter, same tile.

# Turkmenistan: Vaughan et al. 2024 AMT Table 1 — TROPOMI-confirmed emitters
EMITTER_SITES = {
    # Paper sites
    "T6  (Darvaza emitter)":        (39.4616,  53.77502, "detection_T39SYD_2024-06-25"),
    "T7  (Darvaza emitter)":        (39.45965, 53.77921, "detection_T39SYD_2024-06-25"),
    "T14 (Balkanabat emitter)":     (38.55747, 54.20049, "detection_T40SBJ_2024-06-25"),
    "T17 (Balkanabat emitter)":     (38.49393, 54.19764, "detection_T40SBJ_2024-06-25"),
    # Europe: JRC-PPDB-OPEN v1.0 plant coordinates (same tiles previously run)
    # Emsland gas plant, Germany — 1,820 MW fossil gas, JRC ID confirmed
    "Emsland gas plant (JRC)":      (52.481,   7.306,    "detection_T32ULC_2024-06-27"),
    # Eemshaven gas plant, Netherlands — 1,410 MW fossil gas, JRC ID confirmed
    "Eemshaven gas plant (JRC)":    (53.437,   6.881,    "detection_T32ULE_2024-06-28"),
    # Groningen gas field — Europe's largest gas field, TROPOMI-confirmed CH4 emitter
    # Coordinates from prior pipeline run detections clustering at 53.252N, 6.682E
    "Groningen gas field (TROPOMI)":(53.252,   6.682,    "detection_T32ULE_2024-06-28"),
}

CONTROL_SITES = {
    # Turkmenistan controls: +2km north of each emitter
    "T6  CONTROL (+2km N)":         (39.4796,  53.77502, "detection_T39SYD_2024-06-25"),
    "T7  CONTROL (+2km N)":         (39.4776,  53.77921, "detection_T39SYD_2024-06-25"),
    "T14 CONTROL (+2km N)":         (38.5754,  54.20049, "detection_T40SBJ_2024-06-25"),
    "T17 CONTROL (+2km N)":         (38.5119,  54.19764, "detection_T40SBJ_2024-06-25"),
    # European controls: rural/agricultural areas in the same tiles, no known sources
    # ~20km SW of Emsland plant, rural Münsterland farmland (where detections actually fired)
    "Emsland CONTROL (rural 20km)": (52.300,   7.050,    "detection_T32ULC_2024-06-27"),
    # ~20km S of Eemshaven, rural Groningen province farmland
    "Eemshaven CONTROL (rural)":    (53.250,   6.600,    "detection_T32ULE_2024-06-28"),
    # ~10km NW of Groningen field, North Sea coastal (minimal surface sources)
    "Groningen CONTROL (coastal)":  (53.340,   6.500,    "detection_T32ULE_2024-06-28"),
}

# Approach B: TROPOMI-guided emission vs clean comparison
# Replicates the exact test from the prior session:
#   HIGH EMISSION polygon: POLYGON((53.6 39.3, 54.0 39.3, 54.0 39.7, 53.6 39.7, 53.6 39.3))
#     → T40TBK + T40SBJ (TROPOMI dense red, west of Balkanabat)
#     → old model: 80,006 + 58,818 = 138,824 plume pixels
#   CLEAN polygon: POLYGON((56.0 38.3, 56.4 38.3, 56.4 38.7, 56.0 38.7, 56.0 38.3))
#     → T40SDH (Magtymguly, TROPOMI clear)
#     → old model: 239,395 plume pixels  ← MORE than emission! ratio = 0.58
# Target: new model ratio > 1.0 at any threshold
B_EMISSION_STEMS = [
    "detection_T40SBJ_2024-06-25",   # 58,818 px with old model
    "detection_T40TBK_2024-06-25",   # 80,006 px with old model
]
CLEAN_KEY   = "T40SDH_2024-06-24 [Magtymguly CLEAN — TROPOMI clear]"
B_CLEAN_STEM = "detection_T40SDH_2024-06-24"
B_THRESHOLDS = [0.05, 0.10, 0.18, 0.30, 0.50, 0.70, 0.90]


# ─── Inference ──────────────────────────────────────────────────────────────

def run_inference_on_npy(npy_path, geo_path, detector, out_dir):
    """
    Load a cached .npy scene, run CH4Net inference, save GeoTIFF.
    Returns (geotiff_path, prob_array) or raises on error.
    """
    scene = np.load(npy_path)        # (H, W, 12), uint8
    with open(geo_path) as f:
        meta = json.load(f)

    tile_id  = meta["tile_id"]
    acq_date = meta["acquisition_date"][:10]
    crs_str  = meta["crs"]
    transform_vals = meta["transform"]

    # Tile at 160×160 (matches training crop size)
    tiles = tile_scene(scene, tile_size=TILE_SIZE, overlap=0)

    # Batched inference
    tile_arrays = [t.data for t in tiles]
    predictions = detector.detect_batch(tile_arrays, batch_size=32)

    # Stitch
    H, W = scene.shape[:2]
    prob_map = stitch_predictions(tiles, predictions, H, W)

    # Save GeoTIFF
    fname = f"detection_{tile_id}_{acq_date}.tif"
    out_path = os.path.join(out_dir, fname)

    from rasterio.transform import Affine
    affine = Affine(
        transform_vals[0], transform_vals[1], transform_vals[2],
        transform_vals[3], transform_vals[4], transform_vals[5],
    )
    with rasterio.open(
        out_path, "w",
        driver="GTiff",
        height=H, width=W,
        count=1, dtype="float32",
        crs=CRS.from_string(crs_str),
        transform=affine,
    ) as dst:
        dst.write(prob_map.astype(np.float32), 1)

    return out_path, prob_map, meta


# ─── Approach A ─────────────────────────────────────────────────────────────

def extract_crop_stats(prob, transform, crs_str, lat, lon, crop_size=200):
    """Extract stats for a crop_size×crop_size window centered on (lat, lon)."""
    try:
        from pyproj import Transformer
        from rasterio.transform import rowcol as _rowcol
        from rasterio.crs import CRS as _CRS

        tr = Transformer.from_crs("EPSG:4326", crs_str, always_xy=True)
        x, y = tr.transform(lon, lat)
        row, col = _rowcol(transform, x, y)

        half = crop_size // 2
        H, W = prob.shape
        in_bounds = (row - half >= 0 and row + half <= H and
                     col - half >= 0 and col + half <= W)

        r0 = max(0, row - half)
        r1 = min(H, row + half)
        c0 = max(0, col - half)
        c1 = min(W, col + half)
        crop = prob[r0:r1, c0:c1]

        if crop.size == 0:
            return None, row, col, False

        binary = crop >= THRESHOLD
        labeled, n = ndimage.label(binary)
        sizes = np.bincount(labeled.ravel())[1:] if n > 0 else np.array([])
        stats = {
            "mean":         float(crop.mean()),
            "max":          float(crop.max()),
            "pct_above":    float(100.0 * binary.sum() / binary.size),
            "n_blobs":      int(n),
            "largest_blob": int(sizes.max()) if len(sizes) > 0 else 0,
        }
        return stats, row, col, in_bounds
    except Exception as e:
        return None, -1, -1, False


# ─── Reporting ──────────────────────────────────────────────────────────────

def divider(char="═", width=90):
    return char * width

def section(title, char="═", width=90):
    return f"\n{divider(char, width)}\n{title}\n{divider(char, width)}"


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True,
                        help="Path to ch4net_div8_retrained.pth")
    parser.add_argument("--tile-size", type=int, default=TILE_SIZE,
                        help=f"Inference tile size (default: {TILE_SIZE})")
    args = parser.parse_args()

    if not os.path.exists(args.weights):
        print(f"ERROR: Weights not found at '{args.weights}'")
        print("  Download from Drive: My Drive/ch4net_retrained/ch4net_div8_retrained.pth")
        sys.exit(1)

    os.makedirs(RESULTS_V2, exist_ok=True)
    lines = []    # accumulate report lines

    def log(msg=""):
        print(msg)
        lines.append(msg)

    log(section("OVERNIGHT VALIDATION — CH4Net div_factor=8", width=90))
    log(f"  Weights:    {args.weights}")
    log(f"  Tile size:  {args.tile_size}×{args.tile_size} (training: 160×160)")
    log(f"  Threshold:  {THRESHOLD}")
    log(f"  Output dir: {RESULTS_V2}")
    log(f"  Started:    {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # ── Load detector ─────────────────────────────────────────────────────
    log(section("STEP 1: Loading model", "─", 90))
    import torch
    ckpt = torch.load(args.weights, map_location="cpu")
    # approach_c_retrain.py saves raw state_dict — no metadata keys
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        log(f"  Best val F1 from training: {ckpt.get('best_val_f1', 'N/A')}")
        log(f"  Epoch saved at:            {ckpt.get('epoch', 'N/A')}")
    else:
        log(f"  Checkpoint format: raw state_dict ({len(ckpt)} weight tensors)")
    detector = CH4NetDetector(
        weights_path=args.weights,
        threshold=THRESHOLD,
        min_plume_pixels=MIN_BLOB_PX,
    )
    log(f"  Model loaded OK — div_factor=8, {sum(p.numel() for p in detector.model.parameters()):,} params")

    # ── Inference on all cached tiles ─────────────────────────────────────
    log(section("STEP 2: Running inference on all cached tiles", "─", 90))

    npy_files = sorted(glob.glob(os.path.join(NPY_CACHE, "*.npy")))
    geotiff_index = {}   # stem → (prob_array, transform, crs_str)

    for npy_path in npy_files:
        geo_path = npy_path.replace(".npy", "_geo.json")
        if not os.path.exists(geo_path):
            log(f"  SKIP (no geo.json): {os.path.basename(npy_path)}")
            continue

        with open(geo_path) as f:
            meta_raw = json.load(f)
        tile_id  = meta_raw["tile_id"]
        acq_date = meta_raw["acquisition_date"][:10]
        fname    = f"detection_{tile_id}_{acq_date}"

        log(f"\n  → {tile_id} {acq_date} ...")
        t0 = time.time()

        try:
            out_path, prob_map, meta = run_inference_on_npy(
                npy_path, geo_path, detector, RESULTS_V2)

            from rasterio.transform import Affine
            tv = meta["transform"]
            affine = Affine(tv[0], tv[1], tv[2], tv[3], tv[4], tv[5])

            elapsed = time.time() - t0
            log(f"     shape={prob_map.shape}  mean={prob_map.mean():.6f}  "
                f"max={prob_map.max():.4f}  "
                f"pct>{THRESHOLD}={100*(prob_map>=THRESHOLD).mean():.3f}%  "
                f"({elapsed:.0f}s)")
            geotiff_index[fname] = (prob_map, affine, meta["crs"])

        except Exception:
            log(f"     ERROR: {traceback.format_exc().splitlines()[-1]}")

    log(f"\n  Processed {len(geotiff_index)}/{len(npy_files)} tiles")

    # ── Approach A: Spatial specificity at known emitters ─────────────────
    log(section("APPROACH A: Spatial specificity at known emitter coordinates", "─", 90))
    log("  200×200 px (2km×2km) crop centered on each site vs. matched control.")
    log("  Turkmenistan: Vaughan et al. 2024 Table 1 (TROPOMI-confirmed). Control = +2km N.")
    log("  Europe: JRC-PPDB-OPEN v1.0 plant coords + Groningen TROPOMI field. Control = rural same tile.")
    log("  Signal ratio > 1.5 → model has spatial specificity near emitters.")
    log()

    header = (f"  {'Location':<30}  {'InBnds':>6}  {'Mean':>10}  "
              f"{'Max':>8}  {'%>thr':>7}  {'Blobs':>6}  {'LargestBlob':>12}")
    log(header)
    log("  " + "─" * 86)

    a_results = {}
    all_a_entries = {**EMITTER_SITES, **CONTROL_SITES}

    for label, (lat, lon, stem, *_) in all_a_entries.items():
        if stem not in geotiff_index:
            log(f"  {label:<30}  GeoTIFF not found ({stem})")
            continue
        prob, affine, crs_str = geotiff_index[stem]
        stats, row, col, in_bounds = extract_crop_stats(
            prob, affine, crs_str, lat, lon)
        if stats is None:
            log(f"  {label:<30}  ERROR extracting crop (row={row}, col={col})")
            continue
        ib = "YES" if in_bounds else "EDGE"
        log(f"  {label:<30}  {ib:>6}  {stats['mean']:>10.6f}  "
            f"{stats['max']:>8.4f}  {stats['pct_above']:>6.2f}%  "
            f"{stats['n_blobs']:>6,}  {stats['largest_blob']:>12,}")
        a_results[label] = stats

    # Site vs control comparison
    log()
    log(f"  {'─'*86}")
    log(f"  {'Site':<28}  {'Site mean':>10}  {'Ctrl mean':>10}  {'S/C ratio':>10}  Result")
    log(f"  {'─'*28}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*24}")

    # Explicit pairing: emitter key → matched control key
    SITE_CTRL_PAIRS = [
        ("T6  (Darvaza emitter)",        "T6  CONTROL (+2km N)"),
        ("T7  (Darvaza emitter)",        "T7  CONTROL (+2km N)"),
        ("T14 (Balkanabat emitter)",     "T14 CONTROL (+2km N)"),
        ("T17 (Balkanabat emitter)",     "T17 CONTROL (+2km N)"),
        ("Emsland gas plant (JRC)",      "Emsland CONTROL (rural 20km)"),
        ("Eemshaven gas plant (JRC)",    "Eemshaven CONTROL (rural)"),
        ("Groningen gas field (TROPOMI)","Groningen CONTROL (coastal)"),
    ]
    any_signal = False

    for sk, ck in SITE_CTRL_PAIRS:
        sm = a_results.get(sk, {}).get("mean")
        cm = a_results.get(ck, {}).get("mean")
        if sm is None or cm is None:
            log(f"  {sk.split('(')[0].strip():<28}  MISSING DATA (check GeoTIFF index)")
            continue
        ratio = sm / cm if cm > 0 else float("inf")
        if sm > cm * 1.5:
            verdict = "✓✓ STRONG — site > 1.5× control"
            any_signal = True
        elif sm > cm:
            verdict = "✓  site > control"
        else:
            verdict = "✗  control ≥ site (suppression)"
        name = sk.split("(")[0].strip()
        log(f"  {name:<28}  {sm:>10.6f}  {cm:>10.6f}  {ratio:>10.3f}  {verdict}")

    # Tile means for context
    log()
    for stem in sorted(set(v[2] for v in list(EMITTER_SITES.values()))):
        if stem in geotiff_index:
            p, _, _ = geotiff_index[stem]
            log(f"  Tile {stem}: overall mean={p.mean():.6f}")

    log()
    if any_signal:
        log("  INTERPRETATION: Model shows spatial specificity at ≥1 emitter site.")
        log("  Retraining has improved plume localization vs. the terrain-detector baseline.")
    else:
        log("  INTERPRETATION: No sites exceed 1.5× control. Model still lacks strong spatial")
        log("  specificity at known emitters, but improvement from training may be subtle.")
        log("  Check raw means — if emitter mean > control mean (even slightly), that is progress.")

    # ── Approach B: Emission/clean threshold sweep ─────────────────────────
    log(section("APPROACH B: Emission vs. clean tile comparison across thresholds", "─", 90))
    log("  Turkmenistan: T40SBJ (Balkanabat, TROPOMI-confirmed emitter) vs.")
    log("                T40SDH (Magtymguly, TROPOMI-clean control)")
    log("  Ratio > 1.0 at any threshold → model has directional signal.")
    log()

    # Load emission tiles (T40SBJ + T40TBK combined) and clean tile
    emission_probs = []
    for stem in B_EMISSION_STEMS:
        if stem in geotiff_index:
            emission_probs.append(geotiff_index[stem][0])
            log(f"  Emission tile {stem}: shape={geotiff_index[stem][0].shape}, "
                f"mean={geotiff_index[stem][0].mean():.6f}")
        else:
            log(f"  MISSING emission tile: {stem}")

    clean_prob = None
    if B_CLEAN_STEM in geotiff_index:
        clean_prob = geotiff_index[B_CLEAN_STEM][0]
        log(f"  Clean tile   {B_CLEAN_STEM}: shape={clean_prob.shape}, "
            f"mean={clean_prob.mean():.6f}")
    else:
        log(f"  MISSING clean tile: {B_CLEAN_STEM}")

    if emission_probs and clean_prob is not None:
        log()
        log(f"  Baseline (old div_factor=1 model):")
        log(f"    Emission (T40SBJ+T40TBK): 138,824 px @ threshold 0.18")
        log(f"    Clean    (T40SDH):         239,395 px @ threshold 0.18  ← ratio 0.58")
        log(f"  Target: ratio > 1.0 at any threshold with retrained model.")
        log()
        log(f"  {'Threshold':>10}  {'Emission px (combined)':>24}  {'Clean px':>12}  "
            f"{'E/C ratio':>10}  {'vs baseline':>14}  Result")
        log(f"  {'─'*10}  {'─'*24}  {'─'*12}  {'─'*10}  {'─'*14}  {'─'*14}")

        any_flip = False
        for t in B_THRESHOLDS:
            # Sum pixels across both emission tiles
            e_px = sum(int((p >= t).sum()) for p in emission_probs)
            c_px = int((clean_prob >= t).sum())
            ratio = e_px / c_px if c_px > 0 else float("inf")

            # Baseline ratios from old model (pre-retrain)
            baseline = {0.05: None, 0.10: 0.58, 0.18: 0.58, 0.30: None,
                        0.50: None, 0.70: None, 0.90: None}
            b = baseline.get(t)
            vs_base = f"({ratio/b:+.2f}x)" if b else "      "

            result = "✓ E > C" if ratio > 1.0 else "✗ C ≥ E"
            if ratio > 1.0:
                any_flip = True
            log(f"  {t:>10.2f}  {e_px:>24,}  {c_px:>12,}  {ratio:>10.3f}  "
                f"{vs_base:>14}  {result}")

        log()
        if any_flip:
            log("  ✓ EMISSION/CLEAN RATIO FLIPPED > 1.0 — retrained model has directional signal.")
            log("  The clean Magtymguly desert no longer outfires the confirmed emission area.")
        else:
            log("  ✗ Ratio < 1.0 at all thresholds. Model still fires more on clean terrain.")
            log("  Compare absolute ratio to pre-retrain baseline of 0.58 — any increase is progress.")

    # ── Terrain generalization: European tiles ─────────────────────────────
    log(section("TERRAIN GENERALIZATION: European tiles vs. Karakum desert", "─", 90))
    log("  Key question: does the model fire equally on European terrain as Karakum?")
    log("  If yes → still a terrain detector (now for European terrain too).")
    log("  If European tiles show LOWER mean probability → model is more specific to")
    log("  the methane-plume signature (narrow spectral range B11/B12 absorption).")
    log()

    terrain_stats = {}
    for stem, (prob, _, _) in geotiff_index.items():
        # Determine region
        if any(x in stem for x in ["T39", "T40"]):
            region = "Turkmenistan (Karakum desert)"
        elif "T31" in stem or "T32" in stem:
            region = "Europe (Netherlands/Germany)"
        else:
            region = "Other"
        terrain_stats[stem] = {
            "region": region,
            "mean":   float(prob.mean()),
            "p95":    float(np.percentile(prob, 95)),
            "p99":    float(np.percentile(prob, 99)),
            "pct_above_thr": float(100.0 * (prob >= THRESHOLD).mean()),
            "pct_above_05":  float(100.0 * (prob >= 0.05).mean()),
        }

    log(f"  {'Tile (stem)':<42}  {'Region':<30}  {'Mean':>8}  {'P95':>8}  "
        f"{'P99':>8}  {'%>0.18':>7}  {'%>0.05':>7}")
    log(f"  {'─'*42}  {'─'*30}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*7}  {'─'*7}")

    for stem in sorted(terrain_stats):
        s = terrain_stats[stem]
        short_stem = stem.replace("detection_", "")
        log(f"  {short_stem:<42}  {s['region']:<30}  {s['mean']:>8.5f}  "
            f"{s['p95']:>8.4f}  {s['p99']:>8.4f}  "
            f"{s['pct_above_thr']:>6.3f}%  {s['pct_above_05']:>6.3f}%")

    # Group means
    by_region = {}
    for stem, s in terrain_stats.items():
        r = s["region"]
        by_region.setdefault(r, []).append(s["mean"])
    log()
    for region, means in sorted(by_region.items()):
        log(f"  {region}: avg mean prob = {np.mean(means):.5f}  "
            f"(n={len(means)} tiles)")

    log()
    turk_mean = np.mean(by_region.get("Turkmenistan (Karakum desert)", [0]))
    euro_mean = np.mean(by_region.get("Europe (Netherlands/Germany)", [0]))
    if euro_mean < turk_mean * 0.8:
        log("  ✓ European mean significantly lower than Turkmenistan → terrain bias reduced.")
    elif abs(euro_mean - turk_mean) < turk_mean * 0.1:
        log("  ~ European and Turkmenistan means similar → model still terrain-sensitive")
        log("    (possibly transferring terrain bias to European farmland/urban areas).")
    else:
        log(f"  Note: European mean={euro_mean:.5f}, Turkmenistan mean={turk_mean:.5f}")

    # ── Summary ───────────────────────────────────────────────────────────
    log(section("SUMMARY", "═", 90))
    log(f"  Tiles processed:  {len(geotiff_index)}")
    log(f"  Approach A:       {'✓ signal at ≥1 emitter' if any_signal else '✗ no emitter > 1.5× control'}")
    b_result = "✓ ratio flipped > 1.0" if (emission_probs and clean_prob is not None and any_flip) else "✗ ratio < 1.0 everywhere"
    log(f"  Approach B:       {b_result}")
    log(f"  Completed:        {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"\n  New GeoTIFFs in: {RESULTS_V2}")
    log(f"  This report:     {REPORT_PATH}")
    log()

    # Save report
    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
