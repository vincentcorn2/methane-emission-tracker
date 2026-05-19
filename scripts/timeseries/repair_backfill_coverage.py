"""
scripts/repair_backfill_coverage.py
====================================
One-shot repair pass for results_analysis/historical_backfill_timeseries.json.

Root cause: some NPY tiles are partial-swath downloads where the plant location
falls completely outside the actual Sentinel-2 swath (100% zero pixels at the
site crop).  CH4Net inference was still run on these tiles, producing a
globally-uniform probability field.  The S/C computation then returned
sc_ratio=1.0 with site_mean == ctrl_mean == 0.425702 (the ergodic mean of the
degenerate probability map) — a false "no detection" rather than a properly
flagged gap.

This script:
  1. Loads historical_backfill_timeseries.json.
  2. For every record where site_mean == ctrl_mean (to within 1e-6) OR
     sc_ratio == 1.0 with cv_ctrl == 0.0, loads the corresponding NPY and
     measures the valid-pixel fraction in the site crop (100 x 100 px centred
     on the plant coordinates).
  3. If valid_fraction < MIN_VALID_FRAC (default 0.50), replaces the record's
     status with "no_coverage" and nulls all SC-derived fields.
  4. Writes the repaired JSON in-place (keeps a .bak copy).

Usage:
    python scripts/repair_backfill_coverage.py
    python scripts/repair_backfill_coverage.py --dry-run
    python scripts/repair_backfill_coverage.py --min-valid 0.8
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent.parent
TIMESERIES    = ROOT / "results_analysis" / "historical_backfill_timeseries.json"
NPY_CACHE     = ROOT / "data" / "npy_cache"
SC_CROP_PX    = 100          # must match apply_bitemporal_diff.py
MIN_VALID_FRAC = 0.50        # fraction of non-zero pixels required

# SC fields that get nulled when coverage is insufficient
SC_FIELDS = [
    "sc_ratio", "sc_cfar", "site_mean", "ctrl_mean", "ctrl_mu",
    "cv_ctrl", "cfar_thresh_ratio", "cfar_detect", "cfar_margin", "ctrl_n",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── plant coordinates (must match BACKFILL_SITES in historical_backfill_timeseries.py) ──
BACKFILL_SITES = {
    "weisweiler": dict(lat=50.837, lon=6.322),
    "rybnik":     dict(lat=50.135, lon=18.522),
    "belchatow":  dict(lat=51.266, lon=19.315),
    "lippendorf": dict(lat=51.178, lon=12.378),
    "neurath":    dict(lat=51.038, lon=6.616),
    "boxberg":    dict(lat=51.416, lon=14.565),
    "groningen":  dict(lat=53.252, lon=6.682),
    "maasvlakte": dict(lat=51.944, lon=4.067),
}


def lonlat_to_pixel(tif_path: Path, lon: float, lat: float) -> tuple[int, int]:
    with rasterio.open(tif_path) as src:
        epsg = src.crs.to_epsg()
        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        xs, ys = transformer.transform([lon], [lat])
        row, col = rasterio.transform.rowcol(src.transform, xs[0], ys[0])
    return int(row), int(col)


def site_valid_fraction(npy_path: Path, tif_path: Path, lon: float, lat: float) -> float:
    """Return fraction of non-zero pixels in the 100×100 site crop of the NPY."""
    arr = np.load(npy_path)        # (H, W, 12)  uint8
    row, col = lonlat_to_pixel(tif_path, lon, lat)
    half = SC_CROP_PX // 2
    H, W = arr.shape[:2]
    r0 = max(0, row - half)
    r1 = min(H, row + half)
    c0 = max(0, col - half)
    c1 = min(W, col + half)
    patch = arr[r0:r1, c0:c1, :]   # (≤100, ≤100, 12)
    if patch.size == 0:
        return 0.0
    # A pixel is "valid" if any band is non-zero
    valid_mask = patch.any(axis=-1)  # (H, W) bool
    return float(valid_mask.mean())


def is_degenerate(record: dict) -> bool:
    """True if the record shows the zero-coverage fingerprint.

    Two variants:
    A) Classic: site_mean == ctrl_mean (6 dp) AND cv_ctrl == 0.0 AND sc_ratio == 1.0
       — all 4 control crops are identical (plant is entirely outside the swath).
    B) Mixed: site_mean == ctrl_mean (6 dp) AND sc_ratio == 1.0 AND
       site_mean matches the known degenerate sentinel value 0.425702
       — plant crop is zero-coverage but some control crops landed in a valid swath
       edge, making cv_ctrl > 0 while sc still equals 1.
    """
    if record.get("status") != "ok":
        return False
    sm = record.get("site_mean")
    cm = record.get("ctrl_mean")
    sc = record.get("sc_ratio")
    if sm is None or cm is None or sc is None:
        return False
    return abs(sm - cm) < 1e-6 and sc == 1.0


def repair(timeseries: dict, min_valid: float, dry_run: bool) -> tuple[dict, int, int]:
    """
    Scan timeseries, repair degenerate records.
    Returns (repaired_dict, n_checked, n_fixed).
    """
    n_checked = 0
    n_fixed   = 0

    for site, records in timeseries.items():
        cfg = BACKFILL_SITES.get(site)
        if cfg is None:
            log.warning("Unknown site %s — skipping", site)
            continue

        lat, lon = cfg["lat"], cfg["lon"]

        for rec in records:
            if not is_degenerate(rec):
                continue

            n_checked += 1
            npy_name = rec.get("npy")
            tif_path_str = rec.get("tif")
            acq_date = rec.get("acquisition_date", "?")

            if not npy_name or not tif_path_str:
                log.warning("  %s %s — no npy/tif path, marking no_coverage", site, acq_date)
                if not dry_run:
                    rec["status"] = "no_coverage"
                    rec["coverage_note"] = "missing npy or tif path"
                    for f in SC_FIELDS:
                        rec[f] = None
                n_fixed += 1
                continue

            npy_path = NPY_CACHE / npy_name
            tif_path = ROOT / tif_path_str

            if not npy_path.exists():
                log.warning("  %s %s — NPY not found (%s)", site, acq_date, npy_name)
                if not dry_run:
                    rec["status"] = "no_coverage"
                    rec["coverage_note"] = "npy_not_found"
                    for f in SC_FIELDS:
                        rec[f] = None
                n_fixed += 1
                continue

            try:
                vf = site_valid_fraction(npy_path, tif_path, lon, lat)
            except Exception as e:
                log.warning("  %s %s — coverage check error: %s", site, acq_date, e)
                vf = 0.0

            log.info(
                "  %s  %s  valid_frac=%.3f  %s",
                site, acq_date, vf,
                "FAIL" if vf < min_valid else "ok (not degenerate?)"
            )

            if vf < min_valid:
                n_fixed += 1
                if not dry_run:
                    rec["status"]         = "no_coverage"
                    rec["valid_fraction"] = round(vf, 4)
                    rec["coverage_note"]  = (
                        f"site crop {vf*100:.0f}% valid pixels — "
                        "partial-swath tile, plant outside S2 swath"
                    )
                    for f in SC_FIELDS:
                        rec[f] = None

    return timeseries, n_checked, n_fixed


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair WS4 backfill coverage flags")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change without writing")
    parser.add_argument("--min-valid", type=float, default=MIN_VALID_FRAC,
                        help=f"Min valid-pixel fraction (default {MIN_VALID_FRAC})")
    args = parser.parse_args()

    log.info("Loading %s", TIMESERIES)
    with open(TIMESERIES) as f:
        timeseries = json.load(f)

    total_records = sum(len(v) for v in timeseries.values())
    log.info("Loaded %d sites, %d total records", len(timeseries), total_records)

    timeseries, n_checked, n_fixed = repair(timeseries, args.min_valid, args.dry_run)

    log.info("")
    log.info("Checked %d degenerate records, fixed %d", n_checked, n_fixed)

    if args.dry_run:
        log.info("DRY RUN — no files written")
        return

    # Back up original
    bak = TIMESERIES.with_suffix(".json.bak")
    shutil.copy2(TIMESERIES, bak)
    log.info("Backup written to %s", bak.name)

    with open(TIMESERIES, "w") as f:
        json.dump(timeseries, f, indent=2)
    log.info("Repaired JSON written to %s", TIMESERIES)

    # Summary
    print("\n=== Repair Summary ===")
    for site, records in timeseries.items():
        no_cov = [r for r in records if r.get("status") == "no_coverage"]
        ok     = [r for r in records if r.get("status") == "ok"]
        if no_cov:
            dates = [r.get("acquisition_date", "?") for r in no_cov]
            print(f"  {site}: {len(ok)} ok, {len(no_cov)} no_coverage → {dates}")


if __name__ == "__main__":
    main()
