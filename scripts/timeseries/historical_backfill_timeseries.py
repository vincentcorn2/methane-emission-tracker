"""
scripts/historical_backfill_timeseries.py
==========================================
WS4 — Compile multi-year S/C time series for confirmed emitter sites.

For each priority site and every .npy tile in data/npy_cache/ matching that
site's MGRS tile_id, this script:
  1. Checks for an existing inference .tif in results_bitemporal/<site>/.
  2. If absent (and --no-inference is not set), runs CH4Net inference to
     produce original_<stem>.tif.
  3. Calls compute_sc_ratio with the site's lat/lon to extract:
       sc_ratio, sc_cfar, cv_ctrl, cfar_thresh_ratio, cfar_detect, cfar_margin
  4. Parses the acquisition date from the S2 filename.
  5. Saves a structured JSON time series per site to:
       results_analysis/historical_backfill_timeseries.json

Run AFTER:
    python scripts/historical_backfill_download.py
    python apply_bitemporal_diff.py --sites weisweiler rybnik belchatow ...

Or run standalone — it will run inference on any tiles not yet processed.

Usage:
    conda activate methane
    python scripts/historical_backfill_timeseries.py
    python scripts/historical_backfill_timeseries.py --sites weisweiler rybnik
    python scripts/historical_backfill_timeseries.py --no-inference
    python scripts/historical_backfill_timeseries.py --years 2020 2021 2022 2023
"""

import os
import sys
import re
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

Path("results_analysis").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("results_analysis/historical_backfill_timeseries.log"),
    ],
)
log = logging.getLogger(__name__)

# Import inference utilities from main evaluation script
from apply_bitemporal_diff import (
    CH4NetDetector,
    run_inference,
    compute_sc_ratio,
    find_geo_meta,
    WEIGHTS,
    THRESHOLD,
    OUT_DIR,
    NPY_CACHE,
)

# ── Site catalogue (must match historical_backfill_download.py) ───────────────
# skip_bitemporal=True  →  use classic S/C > 1.15 as detection criterion.
# skip_bitemporal=False →  use CFAR (ratio-space adaptive threshold).
CLASSIC_THRESH = 1.15
BACKFILL_SITES = {
    "weisweiler":   dict(lat=50.837, lon=6.322,  tile_id="T31UGS", skip_bitemporal=True),
    "rybnik":       dict(lat=50.135, lon=18.522, tile_id="T34UCA", skip_bitemporal=True),
    "belchatow":    dict(lat=51.266, lon=19.315, tile_id="T34UCB", skip_bitemporal=True),
    "lippendorf":   dict(lat=51.178, lon=12.378, tile_id="T33UUS", skip_bitemporal=True),
    "neurath":      dict(lat=51.038, lon=6.616,  tile_id="T32ULB", skip_bitemporal=True),
    "boxberg":      dict(lat=51.416, lon=14.565, tile_id="T33UVT", skip_bitemporal=True),
    "groningen":    dict(lat=53.252, lon=6.682,  tile_id="T31UGV", skip_bitemporal=False),
    "maasvlakte":   dict(lat=51.944, lon=4.067,  tile_id="T31UET", skip_bitemporal=True),
}

TIMESERIES_OUT = Path("results_analysis/historical_backfill_timeseries.json")

# Minimum fraction of non-zero pixels required in the 100×100 site crop before
# we consider a tile valid.  Tiles where the plant falls outside the S2 swath
# boundary have 0% valid pixels — inference on those produces a globally-uniform
# probability field (sc_ratio == 1.0, all crops identical) which is not a real
# measurement.
MIN_SITE_VALID_FRAC = 0.50

# Regex to extract acquisition datetime from S2 product name.
# Matches both:  S2A_MSIL1C_20240608T104631_...   (processing baseline ≥N05xx)
#                S2B_MSIL1C_20200701T095549_...
_DATE_RE = re.compile(r"S2[AB]_MSIL1C_(\d{8})T\d{6}_")


def parse_acquisition_date(stem: str) -> str | None:
    """Extract ISO date (YYYY-MM-DD) from S2 product filename stem."""
    m = _DATE_RE.search(stem)
    if not m:
        return None
    d = m.group(1)
    return f"{d[:4]}-{d[4:6]}-{d[6:8]}"


def parse_year(stem: str) -> int | None:
    d = parse_acquisition_date(stem)
    return int(d[:4]) if d else None


def find_all_target_npys(tile_id: str) -> list[Path]:
    """Return ALL non-reference S2 .npy files for a tile, sorted by date."""
    candidates = [
        p for p in NPY_CACHE.glob(f"*_{tile_id}_*.npy")
        if "_ref_" not in p.name and "_bitemporal" not in p.name
    ]
    return sorted(candidates, key=lambda p: p.name)


def check_site_coverage(
    npy_path: Path,
    tif_path: Path,
    lat: float,
    lon: float,
    sc_crop_px: int = 100,
) -> float:
    """Return fraction of non-zero pixels in the site crop of the NPY.

    A value below MIN_SITE_VALID_FRAC indicates the plant falls outside the
    actual Sentinel-2 swath boundary for this orbital pass (partial-swath tile).
    Running CH4Net inference on such tiles produces a globally-uniform
    probability field and a spurious sc_ratio=1.0.
    """
    import numpy as np
    from pyproj import Transformer

    arr = np.load(npy_path)       # (H, W, 12) uint8
    with rasterio.open(tif_path) as src:
        epsg = src.crs.to_epsg()
        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        xs, ys = transformer.transform([lon], [lat])
        s_row, s_col = rasterio.transform.rowcol(src.transform, xs[0], ys[0])

    half = sc_crop_px // 2
    H, W = arr.shape[:2]
    r0 = max(0, int(s_row) - half)
    r1 = min(H, int(s_row) + half)
    c0 = max(0, int(s_col) - half)
    c1 = min(W, int(s_col) + half)
    patch = arr[r0:r1, c0:c1, :]
    if patch.size == 0:
        return 0.0
    return float(patch.any(axis=-1).mean())


def get_or_run_inference(
    npy_path: Path,
    site_name: str,
    detector: "CH4NetDetector | None",
    dry_run: bool,
) -> Path | None:
    """
    Return path to the original inference .tif for this npy.
    Creates it if it doesn't exist and detector is available.
    Returns None if tif is absent and we can't/won't create it.
    """
    site_dir = OUT_DIR / site_name
    out_tif = site_dir / f"original_{npy_path.stem}.tif"

    if out_tif.exists():
        log.info("    [hit]  %s", out_tif.name)
        return out_tif

    if dry_run or detector is None:
        log.info("    [miss] %s — skipping (no-inference / dry-run)", npy_path.name)
        return None

    # Run inference
    geo_meta = find_geo_meta(npy_path)
    if geo_meta is None:
        log.warning("    No geo metadata for %s — cannot run inference", npy_path.name)
        return None

    site_dir.mkdir(parents=True, exist_ok=True)
    log.info("    [run]  Running inference on %s ...", npy_path.name)
    try:
        import numpy as np
        arr = np.load(npy_path)
        log.info("           shape=%s  dtype=%s", arr.shape, arr.dtype)
        run_inference(arr, detector, geo_meta, out_tif)
        del arr
        log.info("           Saved: %s", out_tif.name)
        return out_tif
    except Exception as e:
        log.error("    Inference failed for %s: %s", npy_path.name, e)
        return None


def process_site(
    site_name: str,
    cfg: dict,
    detector: "CH4NetDetector | None",
    years_filter: list[int] | None,
    dry_run: bool,
) -> list[dict]:
    """
    Process all tiles for one site. Returns list of per-tile result dicts,
    sorted by acquisition date.
    """
    tile_id = cfg["tile_id"]
    lat, lon = cfg["lat"], cfg["lon"]

    npys = find_all_target_npys(tile_id)
    if not npys:
        log.warning("  No .npy tiles found for %s (%s)", site_name, tile_id)
        return []

    log.info("  %s (%s): %d tiles in cache", site_name, tile_id, len(npys))

    records = []
    for npy_path in npys:
        acq_date = parse_acquisition_date(npy_path.stem)
        year = int(acq_date[:4]) if acq_date else None

        if years_filter and year not in years_filter:
            continue

        log.info("  ── %s  date=%s", npy_path.name[:70], acq_date or "?")

        # ── Coverage pre-check ────────────────────────────────────────────────
        # For partial-swath tiles the plant can fall entirely outside the S2
        # swath boundary (0% valid pixels).  Inference on such tiles produces a
        # globally-uniform probability field and sc_ratio=1.0, which is not a
        # real measurement.  Detect this early from the NPY before inference.
        # We need a TIF to map lat/lon → pixel; use a cached one if it exists,
        # otherwise fall through to the inference step and re-check after.
        site_dir = OUT_DIR / site_name
        candidate_tif = site_dir / f"original_{npy_path.stem}.tif"
        if candidate_tif.exists():
            try:
                vf = check_site_coverage(npy_path, candidate_tif, lat, lon)
                if vf < MIN_SITE_VALID_FRAC:
                    log.warning(
                        "    [skip]  %s  valid_frac=%.2f < %.2f — plant outside swath",
                        npy_path.name, vf, MIN_SITE_VALID_FRAC,
                    )
                    records.append({
                        "site":             site_name,
                        "tile_id":          tile_id,
                        "acquisition_date": acq_date,
                        "year":             year,
                        "npy":              npy_path.name,
                        "tif":              str(candidate_tif),
                        "status":           "no_coverage",
                        "valid_fraction":   round(vf, 4),
                        "coverage_note":    (
                            f"site crop {vf*100:.0f}% valid — partial-swath tile, "
                            "plant outside S2 swath"
                        ),
                    })
                    continue
            except Exception as e:
                log.warning("    Coverage check failed: %s", e)

        tif_path = get_or_run_inference(npy_path, site_name, detector, dry_run)

        if tif_path is None:
            records.append({
                "site":             site_name,
                "tile_id":          tile_id,
                "acquisition_date": acq_date,
                "year":             year,
                "npy":              npy_path.name,
                "tif":              None,
                "status":           "no_tif",
            })
            continue

        sc = compute_sc_ratio(tif_path, lat, lon)
        if sc.get("error"):
            log.warning("    SC error: %s", sc["error"])
            records.append({
                "site":             site_name,
                "tile_id":          tile_id,
                "acquisition_date": acq_date,
                "year":             year,
                "npy":              npy_path.name,
                "tif":              str(tif_path),
                "status":           "sc_error",
                "error":            sc["error"],
            })
            continue

        records.append({
            "site":              site_name,
            "tile_id":           tile_id,
            "acquisition_date":  acq_date,
            "year":              year,
            "npy":               npy_path.name,
            "tif":               str(tif_path),
            "status":            "ok",
            "sc_ratio":          sc.get("sc_ratio"),
            "sc_cfar":           sc.get("sc_cfar"),
            "site_mean":         sc.get("site_mean"),
            "ctrl_mean":         sc.get("ctrl_mean"),
            "ctrl_mu":           sc.get("ctrl_mu"),
            "cv_ctrl":           sc.get("cv_ctrl"),
            "cfar_thresh_ratio": sc.get("cfar_thresh_ratio"),
            "cfar_detect":       sc.get("cfar_detect"),
            "cfar_margin":       sc.get("cfar_margin"),
            "ctrl_n":            sc.get("ctrl_n"),
        })

        r = sc.get("sc_ratio")
        sc_c = sc.get("sc_cfar")
        thr  = sc.get("cfar_thresh_ratio")
        det  = sc.get("cfar_detect")
        log.info(
            "    S/C=%-8s  sc_cfar=%-8s  thr_ratio=%-6s  CFAR=%s",
            f"{r:.4f}"    if r    is not None else "—",
            f"{sc_c:.4f}" if sc_c is not None else "—",
            f"{thr:.3f}"  if thr  is not None else "—",
            "DETECT" if det else "no",
        )

    return sorted(records, key=lambda r: r.get("acquisition_date") or "")


def is_detection(record: dict, skip_bitemporal: bool) -> bool:
    """True if this tile-record counts as a detection under the correct criterion."""
    sc = record.get("sc_ratio")
    if sc is None:
        return False
    if skip_bitemporal:
        return sc > CLASSIC_THRESH          # industrial emitters: classic S/C
    return bool(record.get("cfar_detect"))  # heterogeneous-bg sites: CFAR


def print_summary(timeseries: dict):
    """Print a compact multi-year summary table.

    Detection criterion per site:
      skip_bitemporal=True  →  classic S/C > 1.15  (✓ on any tile that year)
      skip_bitemporal=False →  CFAR adaptive (✓ on any tile that year)
    Value shown = max S/C for that year.
    """
    years_all = sorted({r["year"] for site_recs in timeseries.values()
                        for r in site_recs if r.get("year")})

    col_w = 9
    header_years = "  ".join(f"{y:>{col_w}}" for y in years_all)
    print("\n" + "=" * (20 + (col_w + 2) * len(years_all) + 24))
    print("  WS4 Historical Backfill — S/C Ratio by Site and Year")
    print(f"  {'Site':<18}  {header_years}  {'Detections'}")
    print("  " + "-" * (18 + (col_w + 2) * len(years_all) + 26))

    for site, records in sorted(timeseries.items()):
        cfg = BACKFILL_SITES.get(site, {})
        skip_bt = cfg.get("skip_bitemporal", True)

        by_year: dict[int, list] = {}
        for r in records:
            yr = r.get("year")
            if yr:
                by_year.setdefault(yr, []).append(r)

        cells = []
        detect_years = []
        for yr in years_all:
            recs = by_year.get(yr, [])
            ok = [r for r in recs if r.get("status") == "ok" and r.get("sc_ratio") is not None]
            if not ok:
                cells.append(f"{'—':>{col_w}}")
            else:
                best    = max(ok, key=lambda r: r.get("sc_ratio") or 0)
                sc_val  = best["sc_ratio"]
                # ✓ if ANY tile that year is a detection under the correct criterion
                any_det = any(is_detection(r, skip_bt) for r in ok)
                marker  = "✓" if any_det else " "
                cells.append(f"{marker}{sc_val:>{col_w - 1}.3f}")
                if any_det:
                    detect_years.append(str(yr))

        crit = "S/C>1.15" if skip_bt else "CFAR"
        detect_str = ", ".join(detect_years) if detect_years else "none"
        print(f"  {site:<18}  {'  '.join(cells)}  {detect_str}  [{crit}]")

    print("=" * (20 + (col_w + 2) * len(years_all) + 24))
    print("  ✓ = detection on any tile that year  |  value = max S/C ratio for year")
    print("  S/C>1.15 = classic criterion (industrial emitters, skip_bitemporal sites)")
    print("  CFAR = ratio-space adaptive threshold (heterogeneous-bg sites)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="WS4: compile multi-year S/C time series from npy_cache"
    )
    parser.add_argument("--sites", nargs="+", default=None,
                        choices=list(BACKFILL_SITES.keys()),
                        help="Process only these sites (default: all)")
    parser.add_argument("--years", nargs="+", type=int, default=None,
                        help="Filter to specific years (default: all years in cache)")
    parser.add_argument("--no-inference", action="store_true",
                        help="Skip inference; only report on tiles with existing .tifs")
    parser.add_argument("--weights", default=WEIGHTS,
                        help=f"Path to CH4Net weights (default: {WEIGHTS})")
    args = parser.parse_args()

    sites_to_run = {k: v for k, v in BACKFILL_SITES.items()
                    if args.sites is None or k in args.sites}
    years_filter = args.years

    print("=" * 72)
    print("  WS4 Historical Backfill — Time Series Compilation")
    print(f"  Sites: {list(sites_to_run.keys())}")
    if years_filter:
        print(f"  Years filter: {years_filter}")
    if args.no_inference:
        print("  MODE: NO-INFERENCE (only existing tifs)")
    print("=" * 72)

    # Load model (skip if no-inference)
    detector = None
    if not args.no_inference:
        print(f"\n[1] Loading CH4Net weights from {args.weights}...")
        if not Path(args.weights).exists():
            print(f"    ERROR: Weights not found: {args.weights}")
            print("    Run with --no-inference to skip, or check path.")
            sys.exit(1)
        try:
            detector = CH4NetDetector(args.weights, threshold=THRESHOLD)
            print(f"    OK (device: {detector.device})")
        except Exception as e:
            print(f"    FAILED: {e}")
            sys.exit(1)
    else:
        print("\n[1] Skipping model load (--no-inference)")

    # Load existing timeseries (incremental resume)
    timeseries: dict = {}
    if TIMESERIES_OUT.exists():
        try:
            timeseries = json.load(open(TIMESERIES_OUT))
            n_existing = sum(len(v) for v in timeseries.values())
            log.info("Loaded existing timeseries: %d records across %d sites",
                     n_existing, len(timeseries))
        except Exception:
            timeseries = {}

    # Process sites
    print()
    for site_name, cfg in sites_to_run.items():
        log.info("\n── %s ──────────────────────────────────────────────", site_name.upper())
        new_records = process_site(site_name, cfg, detector, years_filter, args.no_inference)

        if years_filter and site_name in timeseries:
            # Merge: keep existing records for years NOT in filter
            kept = [r for r in timeseries[site_name] if r.get("year") not in years_filter]
            timeseries[site_name] = sorted(
                kept + new_records,
                key=lambda r: r.get("acquisition_date") or ""
            )
        else:
            timeseries[site_name] = new_records

        # Save incrementally
        with open(TIMESERIES_OUT, "w") as f:
            json.dump(timeseries, f, indent=2, default=str)

    # Summary
    print_summary(timeseries)

    total_ok = sum(1 for recs in timeseries.values() for r in recs if r.get("status") == "ok")
    total_detect = sum(1 for recs in timeseries.values() for r in recs
                       if r.get("status") == "ok" and r.get("cfar_detect"))
    total = sum(len(recs) for recs in timeseries.values())

    print(f"  Total tiles processed: {total}")
    print(f"  OK (S/C computed):     {total_ok}")
    print(f"  CFAR detections:       {total_detect}")
    print(f"\n  Time series saved → {TIMESERIES_OUT}")

    if total_ok > 0:
        print("\n  Next step:  open results_analysis/historical_backfill_timeseries.json")
        print("              or run the report update to add WS4 Gap-4 section to ch4net_report.docx")


if __name__ == "__main__":
    main()
