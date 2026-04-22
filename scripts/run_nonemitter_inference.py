"""
scripts/run_nonemitter_inference.py
=====================================
WS5 — Run CH4Net v8 inference on non-emitter reference tiles and extract
S/C ratios for conformal calibration.

For each location in nonemitter_download_manifest.json that has a downloaded
.npy tile, this script:

  1. Loads the full-tile .npy (same uint8 format as data/npy_cache/).
  2. Runs CH4Net v8 (no bitemporal differencing — we want raw model scores
     on non-emitter terrain to characterise false-positive behaviour).
  3. Saves a probability-map GeoTIFF to results_nonemitter/<loc_id>/*.tif
  4. Computes S/C ratio + CFAR metrics at the reference location.
  5. Appends the result to results_analysis/nonemitter_sc_scores.json.

The S/C ratios collected here form the non-conformity score set for the
conformal threshold calibration in scripts/conformal_threshold.py.

Key design choice: NO bitemporal differencing.
Bitemporal differencing suppresses seasonal vegetation changes — the main
nuisance at non-emitter locations — so using BT would produce artifically
low S/C scores.  For a conservative (high-quality) FPR bound we want the
WORST-CASE (highest) S/C the model produces on non-emitter terrain under
real operating conditions.  Without BT, we get the model's raw response to
the full spectral scene.

Usage:
    conda activate methane
    python scripts/run_nonemitter_inference.py [--ids nonemit_001 ...] [--dry-run]
    python scripts/run_nonemitter_inference.py --weights weights/european_model_v8.pth

Notes:
    - Inference on a full S2 tile takes ~8 min on CPU, ~1 min on GPU.
    - Results are written incrementally so partial runs are safe to resume.
    - Use 'caffeinate -i python ...' on macOS.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS_DIR = Path("results_analysis")
RESULTS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(RESULTS_DIR / "nonemitter_inference.log")),
    ],
)
log = logging.getLogger(__name__)

try:
    import rasterio
    import rasterio.warp
    import rasterio.transform
    from rasterio.transform import rowcol
except ImportError:
    log.error("rasterio not installed: pip install rasterio")
    sys.exit(1)

from src.detection.ch4net_model import CH4NetDetector
from src.ingestion.preprocessing import (
    tile_scene,
    stitch_predictions,
    save_prediction_geotiff,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
MANIFEST_IN    = RESULTS_DIR / "nonemitter_manifest.json"
DOWNLOAD_MANIFEST = RESULTS_DIR / "nonemitter_download_manifest.json"
SCORES_OUT     = RESULTS_DIR / "nonemitter_sc_scores.json"
TIF_OUT_DIR    = Path("results_nonemitter")

# ── Inference parameters (matching apply_bitemporal_diff.py exactly) ───────────
DEFAULT_WEIGHTS = "weights/european_model_v8.pth"
TILE_SIZE       = 100
SC_CROP_PX      = 100
SC_OFFSET_DEG   = 0.20    # ~22 km control crop offset
CFAR_K          = 3.0     # ratio-space CFAR coefficient


def lonlat_to_pixel(tif_path: Path, lon: float, lat: float) -> tuple[int, int]:
    """Convert WGS84 lon/lat to (row, col) in a GeoTIFF.

    S2 tiles use UTM projection, so we must reproject from EPSG:4326 to the
    tile CRS before calling rowcol — otherwise coordinates fall wildly OOB.
    Identical to apply_bitemporal_diff.py lines 338-341.
    """
    with rasterio.open(tif_path) as src:
        xs, ys = rasterio.warp.transform("EPSG:4326", src.crs, [lon], [lat])
        row, col = rasterio.transform.rowcol(src.transform, xs[0], ys[0])
    return int(row), int(col)


def safe_crop(arr: np.ndarray, row: int, col: int,
              half: int = SC_CROP_PX // 2,
              min_size: int = 20) -> np.ndarray | None:
    """Crop arr centred on (row, col) with radius half.

    Clamps to valid array bounds rather than returning None at the tile edge,
    as long as the resulting crop is at least min_size × min_size pixels.
    This handles sites that fall within 50 px of a tile boundary.
    """
    H, W = arr.shape
    r0 = max(0, row - half)
    r1 = min(H, row + half)
    c0 = max(0, col - half)
    c1 = min(W, col + half)
    if (r1 - r0) < min_size or (c1 - c0) < min_size:
        return None  # truly outside tile or degenerate
    return arr[r0:r1, c0:c1]


def compute_sc_ratio(tif_path: Path, lat: float, lon: float) -> dict:
    """
    Compute S/C ratio and CFAR metrics — identical logic to apply_bitemporal_diff.py.

    This is the non-conformity score for conformal calibration.
    """
    offsets = [
        ( SC_OFFSET_DEG, 0.0,           "N"),
        (-SC_OFFSET_DEG, 0.0,           "S"),
        (0.0,            SC_OFFSET_DEG, "E"),
        (0.0,           -SC_OFFSET_DEG, "W"),
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

    sm = float(site_crop.mean())

    ctrl_means = []
    first_result = None

    for dlat, dlon, direction in offsets:
        try:
            c_row, c_col = lonlat_to_pixel(tif_path, lon + dlon, lat + dlat)
        except Exception:
            continue
        ctrl_crop = safe_crop(prob, c_row, c_col)
        if ctrl_crop is None:
            continue
        cm = float(ctrl_crop.mean())
        ctrl_means.append(cm)
        if first_result is None:
            sc = sm / cm if cm > 1e-9 else float("inf")
            first_result = dict(
                site_mean=round(sm, 6),
                ctrl_mean=round(cm, 6),
                sc_ratio=round(sc, 4),
                ctrl_direction=direction,
            )

    if first_result is None:
        return {"error": "all_directions_oob"}

    mu_ctrl    = float(np.mean(ctrl_means))
    sigma_ctrl = float(np.std(ctrl_means, ddof=0)) if len(ctrl_means) >= 2 else 0.0
    sc_cfar          = sm / mu_ctrl if mu_ctrl > 1e-9 else float("inf")
    cv_ctrl          = sigma_ctrl / mu_ctrl if mu_ctrl > 1e-9 else 0.0
    cfar_thresh_ratio = 1.15 + CFAR_K * cv_ctrl
    cfar_detect      = bool(sc_cfar > cfar_thresh_ratio)
    cfar_margin      = round(sc_cfar - cfar_thresh_ratio, 4)

    return {
        **first_result,
        "ctrl_n":            len(ctrl_means),
        "ctrl_all_means":    [round(v, 6) for v in ctrl_means],
        "ctrl_mu":           round(mu_ctrl, 6),
        "ctrl_sigma":        round(sigma_ctrl, 6),
        "cv_ctrl":           round(cv_ctrl, 4),
        "cfar_thresh_ratio": round(cfar_thresh_ratio, 4),
        "cfar_detect":       cfar_detect,
        "cfar_margin":       cfar_margin,
        "sc_cfar":           round(sc_cfar, 4),
    }


def run_inference_on_tile(npy_path: Path, detector: "CH4NetDetector",
                          out_tif: Path) -> np.ndarray:
    """Load .npy tile, run CH4Net v8 inference, save probability GeoTIFF."""
    log.info("  Loading %s ...", npy_path.name)
    scene = np.load(npy_path, mmap_mode="r")
    H, W, _ = scene.shape
    log.info("  Scene shape: %d×%d×12", H, W)

    # ── Locate geo metadata ───────────────────────────────────────────────────
    # download_nonemitter_tiles.py renames the .npy with a "nonemit_XXX_" prefix
    # but safe_to_npy writes the _geo.json under the original product name.
    # Strategy: try several candidate paths in order of likelihood.
    import re as _re
    stem = npy_path.stem   # e.g. "nonemit_001_S2B_MSIL1C_..._T32UNE_..."

    # Strip leading "nonemit_NNN_" if present
    m = _re.match(r'^nonemit_\d+_(.+)$', stem)
    original_stem = m.group(1) if m else stem  # e.g. "S2B_MSIL1C_..._T32UNE_..."

    meta_candidates = [
        npy_path.parent / f"{original_stem}_geo.json",   # standard sidecar
        npy_path.parent / f"{stem}_geo.json",             # prefixed variant
        npy_path.with_suffix(".json"),                    # bare .json
        Path("data/npy_cache") / f"{original_stem}_geo.json",
        Path("data/npy_cache") / f"{original_stem}.json",
    ]

    meta_path = None
    for cand in meta_candidates:
        if cand.exists():
            meta_path = cand
            break

    geo_meta = None
    if meta_path:
        try:
            from src.ingestion.preprocessing import GeoMetadata
            import json as _json
            raw = _json.load(open(meta_path))
            geo_meta = GeoMetadata(**{k: v for k, v in raw.items()
                                      if k in GeoMetadata.__dataclass_fields__})
            log.info("  Geo metadata loaded from %s", meta_path.name)
        except Exception as e:
            log.warning("  Could not load geo metadata: %s", e)
    else:
        log.warning("  Geo metadata not found for %s (tried %d paths)",
                    npy_path.name, len(meta_candidates))

    # ── Check for existing raw prob_map from a previous (broken) run ─────────
    # A previous run may have saved a "_prob_prob.npy" without georeferencing.
    # Reuse it to avoid re-running inference (~8 min/tile on CPU).
    cached_prob_npy = out_tif.parent / (out_tif.stem + "_prob.npy")
    if cached_prob_npy.exists():
        log.info("  Re-using cached prob_map from %s (skipping inference)",
                 cached_prob_npy.name)
        prob_map = np.load(str(cached_prob_npy))
    else:
        log.info("  Tiling %d×%d into %d×%d patches...", H, W, TILE_SIZE, TILE_SIZE)
        tiles = tile_scene(scene, tile_size=TILE_SIZE, overlap=0)
        log.info("  Running batched inference on %d tiles...", len(tiles))

        predictions = detector.detect_batch(
            [tile.data for tile in tiles], batch_size=32
        )

        log.info("  Stitching predictions...")
        prob_map = stitch_predictions(tiles, predictions, H, W)

    out_tif.parent.mkdir(parents=True, exist_ok=True)
    if geo_meta is not None:
        save_prediction_geotiff(prob_map, geo_meta, str(out_tif))
        log.info("  Saved GeoTIFF → %s", out_tif.name)
    else:
        # Save without geo-referencing (numpy fallback)
        np.save(str(cached_prob_npy), prob_map)
        log.warning("  No geo metadata — saved raw prob_map to %s", cached_prob_npy.name)

    return prob_map


def process_location(loc: dict, dl_entry: dict, detector: "CH4NetDetector",
                     dry_run: bool) -> dict:
    """Run inference and extract S/C for one non-emitter location."""
    loc_id  = loc["id"]
    lat     = loc["lat"]
    lon     = loc["lon"]
    label   = loc["label"]
    ecoregion = loc.get("ecoregion", "unknown")
    clc_class = loc.get("clc_class", "unknown")

    status = dl_entry.get("status")
    if status not in ("ok", "cached"):
        log.warning("  %s: no downloaded tile (status=%s) — skipping", loc_id, status)
        return {"location_id": loc_id, "status": f"no_tile_{status}"}

    npy_path = Path(dl_entry["npy"])
    if not npy_path.exists():
        # Try resolving symlink target
        log.warning("  %s: .npy not found at %s", loc_id, npy_path)
        return {"location_id": loc_id, "status": "npy_missing"}

    log.info("── %s  (%s) ──", loc_id, label[:45])

    out_tif = TIF_OUT_DIR / loc_id / f"{loc_id}_prob.tif"

    if dry_run:
        log.info("  [dry-run] Would run inference on %s", npy_path.name)
        return {"location_id": loc_id, "status": "dry_run",
                "npy": str(npy_path), "would_write_tif": str(out_tif)}

    # Check if TIF already exists (incremental resume)
    if out_tif.exists():
        log.info("  TIF already exists — reusing, computing S/C only")
        sc_result = compute_sc_ratio(out_tif, lat, lon)
        return _build_result(loc_id, label, ecoregion, clc_class, loc,
                             dl_entry, out_tif, sc_result)

    # Also check if a cached raw prob_map exists from a previous partial run
    # (named _prob_prob.npy by the old broken code path).  If so, we can
    # convert it to a GeoTIFF without re-running inference.
    cached_prob_npy = out_tif.parent / (out_tif.stem + "_prob.npy")
    if cached_prob_npy.exists() and not out_tif.exists():
        log.info("  Found cached prob_map at %s — converting to GeoTIFF",
                 cached_prob_npy.name)
        # run_inference_on_tile will detect the cached npy and skip inference
        # (it checks for the same cached_prob_npy path internally)

    # Run inference
    try:
        prob_map = run_inference_on_tile(npy_path, detector, out_tif)
    except Exception as e:
        log.error("  Inference failed: %s", e)
        return {"location_id": loc_id, "status": "inference_error", "error": str(e)}

    # Compute S/C ratio
    if out_tif.exists():
        sc_result = compute_sc_ratio(out_tif, lat, lon)
    else:
        # Fallback: compute directly from prob_map (no georef)
        log.warning("  TIF missing — computing S/C from raw prob_map array")
        # Use centre of array as site, offset by ~5% of tile size for control
        H, W = prob_map.shape
        c_row, c_col = H // 2, W // 2
        half = SC_CROP_PX // 2
        site_crop = prob_map[c_row-half:c_row+half, c_col-half:c_col+half]
        sm = float(site_crop.mean())
        sc_result = {"site_mean": round(sm, 6), "sc_ratio": float("nan"),
                     "error": "no_geotiff_fallback"}

    return _build_result(loc_id, label, ecoregion, clc_class, loc,
                         dl_entry, out_tif, sc_result)


def _build_result(loc_id, label, ecoregion, clc_class, loc, dl_entry,
                  out_tif, sc_result) -> dict:
    r = {
        "location_id":       loc_id,
        "label":             label,
        "lat":               loc["lat"],
        "lon":               loc["lon"],
        "ecoregion":         ecoregion,
        "clc_class":         clc_class,
        "mgrs_tile":         loc.get("mgrs_tile"),
        "proximity_flag":    loc.get("proximity_flag", "OK"),
        "min_dist_to_emitter_km": loc.get("min_dist_to_emitter_km"),
        "acquisition_date":  dl_entry.get("acquisition_date"),
        "cloud_cover":       dl_entry.get("cloud_cover"),
        "product_name":      dl_entry.get("product_name"),
        "tif":               str(out_tif) if out_tif.exists() else None,
        "status":            "ok" if "error" not in sc_result else "sc_error",
    }
    r.update(sc_result)

    if "error" not in sc_result:
        # Human-readable interpretation
        sc = sc_result.get("sc_cfar") or sc_result.get("sc_ratio", 0.0)
        cfar_det = sc_result.get("cfar_detect", False)
        log.info("  S/C=%.4f  cv_ctrl=%.3f  cfar_detect=%s  [%s / %s]",
                 sc, sc_result.get("cv_ctrl", 0.0), cfar_det, ecoregion, clc_class)
        if cfar_det:
            log.warning("  ⚠ CFAR triggered on non-emitter site %s — potential FP", loc_id)
        r["status"] = "ok"
    else:
        log.warning("  S/C error: %s", sc_result.get("error"))

    return r


def load_scores(path: Path) -> dict:
    """Load existing scores for incremental resume."""
    if path.exists():
        try:
            return {r["location_id"]: r for r in json.load(open(path))}
        except Exception:
            pass
    return {}


def save_scores(scores: dict, path: Path) -> None:
    records = list(scores.values())
    with open(path, "w") as f:
        json.dump(records, f, indent=2)


def print_summary(scores: dict) -> None:
    print("\n" + "=" * 75)
    print("  WS5 Non-Emitter Inference Results")
    print("=" * 75)
    print(f"  {'ID':<14}  {'Ecoregion':<15}  {'CLC':<22}  {'S/C':>6}  {'CFAR':>7}")
    print("-" * 75)

    all_sc = []
    fp_count = 0
    for loc_id, r in sorted(scores.items()):
        sc_val  = r.get("sc_cfar") or r.get("sc_ratio")
        cfar    = r.get("cfar_detect")
        eco     = r.get("ecoregion", "?")[:14]
        clc     = r.get("clc_class", "?")[:21]
        sc_str  = f"{sc_val:.4f}" if sc_val is not None else "n/a"
        cfar_str = ("⚠ FP" if cfar else "—  ") if cfar is not None else "n/a"
        status  = r.get("status", "?")
        if status != "ok":
            sc_str = status[:8]
            cfar_str = "—"
        print(f"  {loc_id:<14}  {eco:<15}  {clc:<22}  {sc_str:>6}  {cfar_str:>7}")
        if sc_val is not None and status == "ok":
            all_sc.append(sc_val)
        if cfar:
            fp_count += 1

    print("=" * 75)
    if all_sc:
        all_sc_arr = sorted(all_sc)
        p90 = float(np.percentile(all_sc_arr, 90))
        p95 = float(np.percentile(all_sc_arr, 95))
        print(f"\n  N={len(all_sc)}  "
              f"min={min(all_sc):.4f}  mean={np.mean(all_sc):.4f}  "
              f"max={max(all_sc):.4f}")
        print(f"  p90={p90:.4f}  p95={p95:.4f}  "
              f"CFAR false positives: {fp_count}/{len(all_sc)}")
        print()
        print(f"  → Empirical FPR at 1.15 threshold: "
              f"{sum(1 for s in all_sc if s > 1.15)/len(all_sc):.1%}")
        print(f"  → 90th-percentile conformalized threshold: {p90:.4f}")
        print(f"    (Use this as alpha=0.10 S/C bound in conformal_threshold.py)")


def main():
    parser = argparse.ArgumentParser(
        description="WS5: run CH4Net v8 inference on non-emitter tiles"
    )
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS,
                        help=f"Model weights path (default: {DEFAULT_WEIGHTS})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check inputs, no inference")
    parser.add_argument("--ids", nargs="+", default=None,
                        help="Process only specific location IDs")
    parser.add_argument("--force-rescore", nargs="*", metavar="ID",
                        help="Re-score these location IDs even if already cached "
                             "(omit IDs to rescore all)")
    parser.add_argument(
        "--download-manifest", default=str(DOWNLOAD_MANIFEST),
        help=f"Download manifest (default: {DOWNLOAD_MANIFEST})"
    )
    args = parser.parse_args()

    # ── Load manifests ────────────────────────────────────────────────────────
    for path, name in [(MANIFEST_IN, "nonemitter_manifest.json"),
                       (Path(args.download_manifest), "download manifest")]:
        if not path.exists():
            print(f"ERROR: {name} not found: {path}")
            print("Run ws5_sample_nonemitters.py and download_nonemitter_tiles.py first.")
            sys.exit(1)

    with open(MANIFEST_IN) as f:
        locations = {loc["id"]: loc for loc in json.load(f)["locations"]}

    with open(args.download_manifest) as f:
        dl_manifest = json.load(f)

    if args.ids:
        locations = {k: v for k, v in locations.items() if k in args.ids}

    # ── Load existing scores (incremental resume) ─────────────────────────────
    scores = load_scores(SCORES_OUT)
    already_done = set(loc_id for loc_id, r in scores.items()
                       if r.get("status") == "ok")

    # --force-rescore: drop cached scores for specified IDs (or all if no IDs given)
    if args.force_rescore is not None:
        if len(args.force_rescore) == 0:
            force_ids = set(already_done)
            log.info("--force-rescore: re-scoring ALL %d cached locations", len(force_ids))
        else:
            force_ids = set(args.force_rescore)
            log.info("--force-rescore: re-scoring %s", sorted(force_ids))
        for fid in force_ids:
            scores.pop(fid, None)
        already_done -= force_ids

    if already_done:
        log.info("Resuming: %d locations already scored", len(already_done))

    # ── Load model ────────────────────────────────────────────────────────────
    if not args.dry_run:
        weights_path = Path(args.weights)
        if not weights_path.exists():
            print(f"ERROR: Model weights not found: {weights_path}")
            sys.exit(1)
        log.info("Loading CH4Net v8 weights from %s", weights_path)
        detector = CH4NetDetector(str(weights_path))
        log.info("Model loaded.")
    else:
        detector = None

    print("=" * 70)
    print("  WS5 Non-Emitter Inference — CH4Net v8")
    print(f"  Weights: {args.weights}")
    print(f"  Locations: {len(locations)}   "
          f"Already done: {len(already_done)}")
    if args.dry_run:
        print("  MODE: DRY RUN")
    print("=" * 70)

    # ── Process each location ─────────────────────────────────────────────────
    for loc_id, loc in sorted(locations.items()):
        if loc_id in already_done and not args.dry_run:
            log.info("  %s: already scored (S/C=%.4f) — skip",
                     loc_id, scores[loc_id].get("sc_cfar", float("nan")))
            continue

        dl_entry = dl_manifest.get(loc_id, {})
        result = process_location(loc, dl_entry, detector, args.dry_run)
        scores[loc_id] = result

        # Write incrementally after each location
        if not args.dry_run and result.get("status") == "ok":
            save_scores(scores, SCORES_OUT)

    # ── Final save and summary ─────────────────────────────────────────────────
    if not args.dry_run:
        save_scores(scores, SCORES_OUT)
        log.info("S/C scores written → %s", SCORES_OUT)

    ok_scores = {k: v for k, v in scores.items() if v.get("status") == "ok"}
    print_summary(ok_scores)

    if not args.dry_run and ok_scores:
        print(f"\nScores written to: {SCORES_OUT}")
        print(f"Next: python scripts/conformal_threshold.py")


if __name__ == "__main__":
    main()
