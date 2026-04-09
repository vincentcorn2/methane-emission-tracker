"""
apply_bitemporal_diff.py
=========================
Step 2 of the bi-temporal retraining pipeline — zero-shot evaluation.

For each key analysis site this script:
  1. Loads the cached summer-2024 (target) .npy tile from data/npy_cache/
  2. Loads the winter-2023 (reference) .npy tile from data/npy_cache/
  3. Computes bi-temporal difference channels:
       delta_B12 = target_B12 - reference_B12   (methane absorption)
       delta_B11 = target_B11 - reference_B11   (atmospheric reference)
     Shifted to [0, 255] by adding 128 (neutral = no seasonal change)
  4. Substitutes the difference channels back into position [10] and [11]
     of the target array — all other 10 bands remain unchanged.
  5. Runs CH4Net v2 inference on both the original and bi-temporal arrays.
  6. Saves GeoTIFFs and computes S/C ratio + ring profile for each.
  7. Prints a side-by-side comparison table.

Expected outcome (hypothesis):
  Groningen / Rybnik:  S/C drops toward 1.0  (terrain artifact suppressed)
  Weisweiler:          S/C stays elevated     (real emission preserved)

If this zero-shot test confirms the hypothesis, bi-temporal input is the
primary feature to add during fine-tuning (Step 3).

Prerequisites:
  1. run download_reference_tiles.py first to fetch winter reference tiles.
  2. conda activate methane  (torch, rasterio required)

Usage:
    python apply_bitemporal_diff.py
    python apply_bitemporal_diff.py --sites weisweiler groningen
    python apply_bitemporal_diff.py --no-baseline   # skip original inference

Output:
    results_bitemporal/<site>/original_<tile>.tif
    results_bitemporal/<site>/bitemporal_<tile>.tif
    results_bitemporal/comparison_summary.json
"""

import os
import sys
import json
import math
import glob
import argparse
import logging
from pathlib import Path
from datetime import datetime

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("results_analysis/bitemporal_eval.log"),
    ],
)
log = logging.getLogger(__name__)

# ── Check dependencies ────────────────────────────────────────────────────────
MISSING = []
try:
    import torch
except ImportError:
    MISSING.append("torch")
try:
    import rasterio
    from rasterio.transform import rowcol, from_bounds
    from rasterio.crs import CRS
    HAS_RASTERIO = True
except ImportError:
    MISSING.append("rasterio")
    HAS_RASTERIO = False

if MISSING:
    print(f"Missing packages: {', '.join(MISSING)}")
    print(f"Install with:  conda install {' '.join(MISSING)}")
    sys.exit(1)

from src.detection.ch4net_model import CH4NetDetector, Unet
from src.ingestion.preprocessing import (
    tile_scene,
    stitch_predictions,
    GeoMetadata,
    save_prediction_geotiff,
)

# ── Configuration ─────────────────────────────────────────────────────────────

NPY_CACHE    = Path("data/npy_cache")
OUT_DIR      = Path("results_bitemporal")
WEIGHTS      = "weights/best_model.pth"
THRESHOLD    = 0.18
TILE_SIZE    = 100    # CH4Net patch size

# B11 → index 10, B12 → index 11 (from preprocessing.py BAND_CONFIG)
B11_IDX = 10
B12_IDX = 11

# S/C ratio parameters (matching run_expanded_sites.py)
SC_CROP_PX      = 100    # side of the site crop in pixels
SC_OFFSET_DEG   = 0.20   # ~22 km — control crop offset

# Ring profile parameters
RING_STEP_KM  = 5
RING_MAX_KM   = 40

# Sites to evaluate — must have tiles in npy_cache
SITES = {
    "weisweiler": dict(lat=50.837, lon=6.322,  tile_id="T32ULB",
                       note="Confirmed emitter (S/C=2.091 on two dates)"),
    "boxberg":    dict(lat=51.416, lon=14.565, tile_id="T33UVT",
                       note="Marginal (S/C=1.517, ~8.6% above Lausitz terrain bg)"),
    "rybnik":     dict(lat=50.135, lon=18.522, tile_id="T34UCA",
                       note="Terrain artefact (ring profile increases outward)"),
    "groningen":  dict(lat=53.252, lon=6.682,  tile_id="T31UGV",
                       note="Terrain artefact (TROPOMI: -0.99 ppb on best date)"),
}

# ── Tile discovery ─────────────────────────────────────────────────────────────

def find_target_npy(tile_id: str) -> Path | None:
    """Find the most recent non-reference .npy for a given tile ID."""
    candidates = [
        p for p in NPY_CACHE.glob(f"*_{tile_id}_*.npy")
        if "_ref_" not in p.name and "_bitemporal" not in p.name
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def find_reference_npy(tile_id: str) -> Path | None:
    """Find the winter reference .npy for a given tile ID."""
    candidates = list(NPY_CACHE.glob(f"{tile_id}_ref_*.npy"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def find_geo_meta(npy_path: Path) -> GeoMetadata | None:
    """Load the JSON sidecar geo metadata for a .npy file."""
    meta_path = npy_path.with_name(npy_path.stem + "_geo.json")
    if not meta_path.exists():
        # Try alternative naming (_ref_ convention)
        stem = npy_path.stem
        meta_candidates = list(npy_path.parent.glob(f"{stem}*_geo.json"))
        if not meta_candidates:
            log.warning("No geo metadata found for %s", npy_path.name)
            return None
        meta_path = meta_candidates[0]
    return GeoMetadata.load(str(meta_path))


# ── Bi-temporal difference ─────────────────────────────────────────────────────

def apply_bitemporal_difference(
    target: np.ndarray,
    reference: np.ndarray,
    replace_channels: list[int] = (B11_IDX, B12_IDX),
) -> np.ndarray:
    """
    Replace specified channels with signed seasonal-difference, shifted to [0,255].

    Strategy:
      delta = int16(target) - int16(reference)    range: [-255, +255]
      delta_shifted = delta + 128                  range: [-127, +383]
      clipped = clip(delta_shifted, 0, 255)        uint8

    Semantics after shifting:
      pixel = 128  →  no seasonal change (neutral)
      pixel > 128  →  target brighter than reference in this channel
      pixel < 128  →  target darker than reference in this channel

    For a genuine methane plume:
      delta_B12 should be markedly different from delta_B11
      (methane absorbs at 2190nm but not at 1610nm)

    For terrain artifact:
      delta_B12 ≈ delta_B11 ≈ 0  →  both shifted to ~128

    Args:
        target:    (H, W, 12) uint8 array (summer 2024)
        reference: (H, W, 12) uint8 array (winter 2023) — must be same tile
        replace_channels: which channel indices to replace with delta values

    Returns:
        Modified uint8 array with replaced channels.
    """
    assert target.shape == reference.shape, (
        f"Shape mismatch: target {target.shape} vs reference {reference.shape}"
    )

    result = target.copy()
    for idx in replace_channels:
        t_ch = target[:, :, idx].astype(np.int16)
        r_ch = reference[:, :, idx].astype(np.int16)
        delta = t_ch - r_ch                         # [-255, +255]
        delta_shifted = np.clip(delta + 128, 0, 255).astype(np.uint8)
        result[:, :, idx] = delta_shifted
        log.info(
            "  Channel %d (B%s): delta range [%d, %d]  shifted mean=%.1f",
            idx, {10: "11", 11: "12"}.get(idx, "?"),
            int(delta.min()), int(delta.max()), float(delta_shifted.mean())
        )
    return result


# ── Inference on a full-tile .npy ─────────────────────────────────────────────

def run_inference(
    scene: np.ndarray,
    detector: CH4NetDetector,
    geo_meta: GeoMetadata,
    out_tif: Path,
) -> np.ndarray:
    """
    Tile a full scene, run CH4Net inference, stitch and save GeoTIFF.

    Args:
        scene:    (H, W, 12) uint8 array
        detector: loaded CH4NetDetector
        geo_meta: GeoMetadata for the scene
        out_tif:  output GeoTIFF path

    Returns:
        (H, W) float32 probability map
    """
    H, W, _ = scene.shape
    log.info("  Tiling %d×%d scene → %d×%d patches...",
             H, W, TILE_SIZE, TILE_SIZE)

    tiles = tile_scene(scene, tile_size=TILE_SIZE, overlap=0)
    log.info("  Running inference on %d tiles...", len(tiles))

    predictions = []
    for i, tile in enumerate(tiles):
        result = detector.detect(tile.data)
        predictions.append(result.probability_map)
        if (i + 1) % 500 == 0:
            log.info("    %d / %d tiles done", i + 1, len(tiles))

    log.info("  Stitching predictions...")
    prob_map = stitch_predictions(tiles, predictions, H, W)

    out_tif.parent.mkdir(parents=True, exist_ok=True)
    save_prediction_geotiff(prob_map, geo_meta, str(out_tif))
    log.info("  Saved → %s", out_tif.name)

    return prob_map


# ── Analysis functions ─────────────────────────────────────────────────────────

def lonlat_to_pixel(tif_path: Path, lon: float, lat: float) -> tuple[int, int]:
    """Convert geographic coordinates to pixel row/col."""
    with rasterio.open(tif_path) as src:
        xs, ys = rasterio.warp.transform(
            "EPSG:4326", src.crs, [lon], [lat]
        )
        row, col = rasterio.transform.rowcol(src.transform, xs[0], ys[0])
    return int(row), int(col)


def safe_crop(arr: np.ndarray, row: int, col: int,
              half: int = SC_CROP_PX // 2) -> np.ndarray | None:
    """Extract a square crop, returning None if out-of-bounds."""
    H, W = arr.shape
    r0, r1 = row - half, row + half
    c0, c1 = col - half, col + half
    if r0 < 0 or r1 > H or c0 < 0 or c1 > W:
        return None
    return arr[r0:r1, c0:c1]


def compute_sc_ratio(tif_path: Path, lat: float, lon: float) -> dict:
    """Compute S/C ratio using 4-directional offset controls."""
    offsets = [
        ( SC_OFFSET_DEG, 0.0,            "N"),
        (-SC_OFFSET_DEG, 0.0,            "S"),
        (0.0,            SC_OFFSET_DEG,  "E"),
        (0.0,           -SC_OFFSET_DEG,  "W"),
    ]
    try:
        with rasterio.open(tif_path) as src:
            prob = src.read(1).astype(np.float32)
        s_row, s_col = lonlat_to_pixel(tif_path, lon, lat)
    except Exception as e:
        return {"error": str(e)}

    site_crop = safe_crop(prob, s_row, s_col)
    if site_crop is None:
        return {"error": "site_out_of_bounds"}

    for dlat, dlon, direction in offsets:
        try:
            c_row, c_col = lonlat_to_pixel(tif_path, lon + dlon, lat + dlat)
        except Exception:
            continue
        ctrl_crop = safe_crop(prob, c_row, c_col)
        if ctrl_crop is None:
            continue
        sm = float(site_crop.mean())
        cm = float(ctrl_crop.mean())
        sc = sm / cm if cm > 1e-9 else float("inf")
        return dict(
            site_mean=round(sm, 6), ctrl_mean=round(cm, 6),
            sc_ratio=round(sc, 4),  ctrl_direction=direction,
        )
    return {"error": "all_directions_oob"}


def compute_ring_profile(tif_path: Path, lat: float, lon: float) -> list:
    """Concentric ring mean probability (decay diagnostic)."""
    rings = []
    try:
        with rasterio.open(tif_path) as src:
            prob = src.read(1).astype(np.float32)
            H, W = prob.shape
            xs, ys = rasterio.warp.transform("EPSG:4326", src.crs, [lon], [lat])
            p_row, p_col = rasterio.transform.rowcol(src.transform, xs[0], ys[0])
            px_m = abs(src.transform.a)
            py_m = abs(src.transform.e)
            cols, rows = np.meshgrid(np.arange(W, np.float32), np.arange(H, np.float32))
            dist_km = np.sqrt(
                ((cols - p_col) * px_m) ** 2 + ((rows - p_row) * py_m) ** 2
            ) / 1000.0
    except Exception as e:
        return [{"error": str(e)}]

    for inner in range(0, RING_MAX_KM, RING_STEP_KM):
        outer = inner + RING_STEP_KM
        mask  = (dist_km >= inner) & (dist_km < outer)
        n     = int(mask.sum())
        rings.append(dict(
            inner_km=inner, outer_km=outer, n_pixels=n,
            mean_prob=round(float(prob[mask].mean()), 6) if n > 0 else None,
        ))
    return rings


def ring_gradient(rings: list) -> float | None:
    """Fit log-linear model to ring profile; return slope (negative = good)."""
    xs, ys = [], []
    for r in rings:
        mp = r.get("mean_prob")
        if mp and mp > 0:
            xs.append((r["inner_km"] + r["outer_km"]) / 2)
            ys.append(math.log(mp))
    if len(xs) < 3:
        return None
    x, y = np.array(xs), np.array(ys)
    xm, ym = x.mean(), y.mean()
    slope = float(((x - xm) @ (y - ym)) / ((x - xm) @ (x - xm)))
    return round(slope, 6)


# ── Main evaluation loop ──────────────────────────────────────────────────────

def evaluate_site(
    site_name: str,
    meta: dict,
    detector: CH4NetDetector,
    run_baseline: bool,
) -> dict:
    """
    Full evaluation for one site: original + bi-temporal.
    Returns dict with results for both modes.
    """
    tile_id = meta["tile_id"]
    lat, lon = meta["lat"], meta["lon"]

    log.info("\n" + "=" * 65)
    log.info("Site: %-15s  tile: %s", site_name.upper(), tile_id)
    log.info("Note: %s", meta["note"])

    # Locate .npy files
    target_npy = find_target_npy(tile_id)
    ref_npy    = find_reference_npy(tile_id)

    if target_npy is None:
        log.warning("  No target .npy found for tile %s — skipping", tile_id)
        return {"status": "no_target", "tile_id": tile_id}

    log.info("  Target:    %s", target_npy.name)
    if ref_npy:
        log.info("  Reference: %s", ref_npy.name)
    else:
        log.warning("  No reference .npy for tile %s — skipping bi-temporal mode", tile_id)

    geo_meta = find_geo_meta(target_npy)
    if geo_meta is None:
        log.warning("  No geo metadata for target — skipping")
        return {"status": "no_geo_meta", "tile_id": tile_id}

    site_dir = OUT_DIR / site_name
    site_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "tile_id": tile_id,
        "target_npy": target_npy.name,
        "ref_npy": ref_npy.name if ref_npy else None,
        "status": "ok",
    }

    # ── Load target array ────────────────────────────────────────────────────
    log.info("  Loading target array (~1.4 GB for full tile, please wait)...")
    target = np.load(target_npy)
    log.info("  Target shape: %s  dtype: %s", target.shape, target.dtype)

    # ── Baseline (original single-date) ────────────────────────────────────
    if run_baseline:
        out_tif_orig = site_dir / f"original_{target_npy.stem}.tif"
        if out_tif_orig.exists():
            log.info("  [baseline] Already exists: %s", out_tif_orig.name)
            prob_orig = None  # don't reload; SC will read from tif
        else:
            log.info("  [baseline] Running inference on original array...")
            prob_orig = run_inference(target, detector, geo_meta, out_tif_orig)

        sc_orig  = compute_sc_ratio(out_tif_orig, lat, lon)
        ring_orig = compute_ring_profile(out_tif_orig, lat, lon)
        grad_orig = ring_gradient(ring_orig)

        results["original"] = {
            "tif":             str(out_tif_orig),
            "sc_ratio":        sc_orig.get("sc_ratio"),
            "site_mean":       sc_orig.get("site_mean"),
            "ctrl_mean":       sc_orig.get("ctrl_mean"),
            "ring_gradient":   grad_orig,
            "sc_error":        sc_orig.get("error"),
        }
        r = sc_orig.get("sc_ratio")
        log.info("  [baseline] S/C = %s  ring_slope = %s",
                 f"{r:.3f}" if r else "—",
                 f"{grad_orig:.5f}" if grad_orig else "—")

    # ── Bi-temporal mode ────────────────────────────────────────────────────
    if ref_npy is not None:
        out_tif_bitemp = site_dir / f"bitemporal_{target_npy.stem}.tif"

        if out_tif_bitemp.exists():
            log.info("  [bitemporal] Already exists: %s", out_tif_bitemp.name)
        else:
            log.info("  [bitemporal] Loading reference array...")
            reference = np.load(ref_npy)
            log.info("  Reference shape: %s", reference.shape)

            # Sanity check — same tile should give same shape
            if reference.shape != target.shape:
                log.error(
                    "  Shape mismatch target=%s ref=%s — cannot diff",
                    target.shape, reference.shape
                )
                results["bitemporal"] = {"error": "shape_mismatch"}
            else:
                log.info("  [bitemporal] Computing B11/B12 difference channels...")
                bt_array = apply_bitemporal_difference(target, reference)

                # Geo metadata is the same (same tile, same CRS/transform)
                log.info("  [bitemporal] Running inference on difference array...")
                run_inference(bt_array, detector, geo_meta, out_tif_bitemp)
                del reference  # free memory

        if out_tif_bitemp.exists():
            sc_bt  = compute_sc_ratio(out_tif_bitemp, lat, lon)
            ring_bt = compute_ring_profile(out_tif_bitemp, lat, lon)
            grad_bt = ring_gradient(ring_bt)

            results["bitemporal"] = {
                "tif":           str(out_tif_bitemp),
                "sc_ratio":      sc_bt.get("sc_ratio"),
                "site_mean":     sc_bt.get("site_mean"),
                "ctrl_mean":     sc_bt.get("ctrl_mean"),
                "ring_gradient": grad_bt,
                "sc_error":      sc_bt.get("error"),
            }
            r = sc_bt.get("sc_ratio")
            log.info("  [bitemporal] S/C = %s  ring_slope = %s",
                     f"{r:.3f}" if r else "—",
                     f"{grad_bt:.5f}" if grad_bt else "—")
    else:
        results["bitemporal"] = {"error": "no_reference_tile"}

    del target  # free memory
    return results


# ── Comparison table ──────────────────────────────────────────────────────────

def print_comparison(all_results: dict):
    """Print a side-by-side comparison table."""
    print("\n")
    print("=" * 80)
    print("  BI-TEMPORAL ZERO-SHOT COMPARISON — CH4Net v2")
    print("=" * 80)
    print(f"  {'Site':<15} {'Orig S/C':>10}  {'BT S/C':>10}  {'Delta S/C':>10}  "
          f"{'Orig slope':>11}  {'BT slope':>10}  {'Assessment'}")
    print("  " + "-" * 76)

    for site, result in all_results.items():
        if result.get("status") not in ("ok", None):
            print(f"  {site:<15} SKIPPED ({result.get('status','?')})")
            continue

        orig  = result.get("original", {})
        bitemp = result.get("bitemporal", {})

        sc_o  = orig.get("sc_ratio")
        sc_bt = bitemp.get("sc_ratio")
        sg_o  = orig.get("ring_gradient")
        sg_bt = bitemp.get("ring_gradient")

        delta_sc = (sc_bt - sc_o) if (sc_o and sc_bt) else None

        # Assessment
        note = SITES[site]["note"].split(" (")[0]  # shorten
        if sc_o and sc_bt:
            if delta_sc < -0.15:
                assessment = "✓ FP suppressed"
            elif delta_sc > 0.15:
                assessment = "↑ Signal preserved"
            elif abs(delta_sc) <= 0.15:
                assessment = "~ Minimal change"
            else:
                assessment = "?"
        else:
            assessment = "— data missing"

        def fmt(v): return f"{v:>10.3f}" if v is not None else f"{'—':>10}"
        def fmtd(v): return f"{v:>+10.3f}" if v is not None else f"{'—':>10}"
        def fmtg(v): return f"{v:>11.5f}" if v is not None else f"{'—':>11}"

        print(f"  {site:<15} {fmt(sc_o)}  {fmt(sc_bt)}  {fmtd(delta_sc)}  "
              f"{fmtg(sg_o)}  {fmt(sg_bt)}  {assessment}")

    print("=" * 80)
    print("\n  Interpretation guide:")
    print("  Orig S/C > 1.5  →  strong single-date detection")
    print("  BT S/C closer to 1.0  →  bi-temporal suppresses terrain artefact")
    print("  ring_gradient < 0  →  probability decays with distance (real plume)")
    print("  ring_gradient > 0  →  probability rises with distance (terrain artefact)")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Bi-temporal zero-shot evaluation")
    parser.add_argument("--sites", nargs="+",
                        default=list(SITES.keys()),
                        help="Sites to evaluate (default: all)")
    parser.add_argument("--no-baseline", action="store_true",
                        help="Skip original single-date inference (only run bi-temporal)")
    parser.add_argument("--weights", default=WEIGHTS,
                        help=f"Path to CH4Net weights (default: {WEIGHTS})")
    args = parser.parse_args()

    run_baseline = not args.no_baseline
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    Path("results_analysis").mkdir(exist_ok=True)

    print("=" * 65)
    print("  Bi-temporal Difference Evaluation — CH4Net v2")
    print(f"  Sites: {', '.join(args.sites)}")
    print(f"  Weights: {args.weights}")
    print(f"  Threshold: {THRESHOLD}")
    print("=" * 65)

    # Check weights exist
    if not Path(args.weights).exists():
        print(f"\nERROR: Weights not found: {args.weights}")
        print("Make sure you are in the methane-api directory and the model is trained.")
        sys.exit(1)

    # Check reference tiles
    missing_refs = []
    for site_name in args.sites:
        if site_name not in SITES:
            print(f"WARNING: Unknown site '{site_name}' — skipping")
            continue
        tile_id = SITES[site_name]["tile_id"]
        if find_reference_npy(tile_id) is None:
            missing_refs.append(f"{site_name} ({tile_id})")

    if missing_refs:
        print(f"\nWARNING: No reference tiles found for: {', '.join(missing_refs)}")
        print("Run download_reference_tiles.py first to fetch winter 2023 tiles.")
        print("Bi-temporal mode will be skipped for these sites.")
        print("(Baseline/original mode will still run.)\n")

    # Load model
    print(f"\n[1] Loading CH4Net v2 weights from {args.weights}...")
    try:
        detector = CH4NetDetector(args.weights, threshold=THRESHOLD)
        print(f"    OK (device: {detector.device})")
    except Exception as e:
        print(f"    FAILED: {e}")
        sys.exit(1)

    # Evaluate each site
    all_results = {}
    for site_name in args.sites:
        if site_name not in SITES:
            continue
        try:
            result = evaluate_site(
                site_name,
                SITES[site_name],
                detector,
                run_baseline=run_baseline,
            )
            all_results[site_name] = result
        except Exception as e:
            log.exception("Unhandled error for site %s", site_name)
            all_results[site_name] = {"status": "error", "error": str(e)}

    # Print and save comparison
    print_comparison(all_results)

    summary_path = Path("results_analysis/bitemporal_comparison.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Full results saved → {summary_path}")

    # Quick decision tree
    print("\n[Next steps based on results]")
    for site, result in all_results.items():
        orig  = result.get("original", {})
        bitemp = result.get("bitemporal", {})
        sc_o   = orig.get("sc_ratio")
        sc_bt  = bitemp.get("sc_ratio")
        if sc_o and sc_bt:
            delta = sc_bt - sc_o
            if site in ("groningen", "rybnik") and delta < -0.1:
                print(f"  {site}: bi-temporal suppresses FP — USE as negative example in fine-tuning")
            elif site == "weisweiler" and sc_bt > 1.3:
                print(f"  {site}: signal preserved in bi-temporal mode — bi-temporal is safe for fine-tuning")
            elif site == "weisweiler" and sc_bt < 1.3:
                print(f"  {site}: WARNING — bi-temporal attenuates Weisweiler signal too much")
                print(f"          Consider partial channel replacement (only B12, not B11+B12)")


if __name__ == "__main__":
    main()
