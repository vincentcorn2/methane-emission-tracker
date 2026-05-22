"""
scripts/run_new_site_timeseries.py
====================================
Run the full CH4Net + CEMF/IME pipeline on a new open-pit lignite site and
produce a multi-year detection/quantification record for report Section 6
("Additional Case Studies").

Mirrors belchatow_annual_timeseries.py exactly — same inference stack, same
conformal threshold (τ read from calibrated_threshold.json), same 7.5 km crop, same ERA5 winds — so
results are directly comparable.

Pre-configured sites
--------------------
  turow       Turów, Poland        50.929°N  14.926°E  (~7,000 t/yr CH₄)
  welzow      Welzow-Süd, Germany  51.599°N  14.279°E  (~25,800 t/yr CH₄)

Usage
-----
    conda activate methane
    # dry-run to check catalog availability first:
    caffeinate -i python scripts/run_new_site_timeseries.py --site turow --dry-run
    caffeinate -i python scripts/run_new_site_timeseries.py --site welzow --dry-run

    # full run — expect ~2-4 h per site on Apple M-series:
    caffeinate -i python scripts/run_new_site_timeseries.py --site turow
    caffeinate -i python scripts/run_new_site_timeseries.py --site welzow

    # single year to test first:
    caffeinate -i python scripts/run_new_site_timeseries.py --site turow --years 2023

    # custom site via args:
    caffeinate -i python scripts/run_new_site_timeseries.py \\
        --site mysite --lat 51.2 --lon 14.5 --years 2022 2023 2024

Output
------
  results_analysis/{site}_timeseries.json   — one record per acquisition
  results_analysis/{site}_timeseries.log    — full run log

The JSON schema is identical to belchatow_annual_timeseries.json so the
same analysis/reporting code can consume it without modification.
"""
from __future__ import annotations

import argparse
import getpass
import json
import logging
import os
import re
import shutil
import sys
import tempfile
from calendar import monthrange
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.copernicus_client import CopernicusClient
from src.ingestion.preprocessing import safe_to_npy
from src.ingestion.era5_client import ERA5Client
from src.quantification.runner import SiteCfg, run_quantification

from apply_bitemporal_diff import (
    B11_IDX,
    B12_IDX,
    CH4NetDetector,
    NPY_CACHE,
    OUT_DIR,
    WEIGHTS,
    compute_sc_ratio,
    find_geo_meta,
    lonlat_to_pixel,
    run_inference,
)

# ── Pre-configured sites ───────────────────────────────────────────────────────
SITES = {
    "turow": {
        "label":       "Turów open-pit lignite mine, Poland (PGE)",
        "lat":          50.929,
        "lon":          14.926,
        "climate_trace_asset": "KWB Turów",
        "climate_trace_ch4_2024_t": 7000,   # approximate
        "operator":    "PGE Polska Grupa Energetyczna",
    },
    "welzow": {
        "label":       "Welzow-Süd open-pit lignite mine, Germany (LEAG)",
        "lat":          51.599,
        "lon":          14.279,
        "climate_trace_asset": "Welzow-Süd",
        "climate_trace_ch4_2024_t": 25800,  # approximate
        "operator":    "LEAG (Lausitz Energie Bergbau AG)",
    },
}

# ── Inference / quantification constants (identical to belchatow script) ──────
def _load_tau(default: float = 3.5796) -> float:
    """Read τ(α=0.10) from calibrated_threshold.json; fall back to default."""
    try:
        p = Path(__file__).resolve().parent.parent / "results_analysis" / "calibrated_threshold.json"
        d = json.load(open(p))
        tau = d.get("global_thresholds", {}).get("tau_alpha_10", {}).get("tau")
        if tau is not None:
            return float(tau)
    except Exception:
        pass
    return default

CONFORMAL_TAU       = _load_tau()   # α = 0.10 — auto-read from calibrated_threshold.json
DETECTION_THRESHOLD = 1.15          # classic S/C threshold (pre-CFAR gate)
CROP_PX_QUANT       = 750      # 750 px × 10 m/px = 7.5 km crop for CEMF/IME

ACQ_DATE_RE = re.compile(r"_(\d{8})T")


# ── Logging (configured in main after site name is known) ─────────────────────
log = logging.getLogger("new_site_timeseries")


# ── Helpers ───────────────────────────────────────────────────────────────────
def bbox_wkt(lat: float, lon: float, margin: float = 0.25) -> str:
    return (f"POLYGON(({lon-margin} {lat-margin},{lon+margin} {lat-margin},"
            f"{lon+margin} {lat+margin},{lon-margin} {lat+margin},"
            f"{lon-margin} {lat-margin}))")


def month_window(year: int, month: int) -> tuple[str, str]:
    last = monthrange(year, month)[1]
    return (f"{year}-{month:02d}-01T00:00:00.000Z",
            f"{year}-{month:02d}-{last:02d}T23:59:59.999Z")


def get_credentials() -> tuple[str, str]:
    user = os.environ.get("CDSE_USERNAME")
    pw   = os.environ.get("CDSE_PASSWORD")
    if user and pw:
        return user, pw
    log.info("CDSE credentials not set in environment — prompting.")
    user = input("CDSE username: ")
    pw   = getpass.getpass("CDSE password: ")
    return user, pw


# ── Download / cache ───────────────────────────────────────────────────────────
def cached_for_date(date_str: str, site_name: str) -> list[Path]:
    """Look up cached .npy files by acquisition date (any MGRS tile)."""
    date_compact = date_str.replace("-", "")
    out = []
    for p in NPY_CACHE.glob(f"*{date_compact}*.npy"):
        if "_ref_" in p.name or "MSIL1C" not in p.name:
            continue
        m = ACQ_DATE_RE.search(p.name)
        if m and m.group(1) == date_compact:
            out.append(p)
    return sorted(out)


def download_one(client, product, site_name: str, download_dir: Path) -> Path:
    download_dir.mkdir(parents=True, exist_ok=True)
    NPY_CACHE.mkdir(parents=True, exist_ok=True)

    # Extract tile_id from product name (e.g. _T33UVR_)
    m = re.search(r"_T(\d{2}[A-Z]{3})_", product.name)
    tile_id = ("T" + m.group(1)) if m else None

    log.info("  Downloading %s ...", product.name[:70])
    zip_path = client.download_product(product, str(download_dir))
    if zip_path is None:
        raise RuntimeError("download_product returned None")

    log.info("  Converting to .npy ...")
    extract_dir = tempfile.mkdtemp(prefix=f"s2_{site_name}_")
    try:
        npy_path, _ = safe_to_npy(
            zip_path=zip_path,
            output_dir=str(NPY_CACHE),
            tile_id=tile_id,
            acquisition_date=product.acquisition_date,
            satellite=product.satellite,
            extract_dir=extract_dir,
        )
        log.info("  Saved: %s", Path(npy_path).name)
        # Delete zip immediately after successful conversion
        try:
            zip_size = Path(zip_path).stat().st_size / 1e9
            Path(zip_path).unlink()
            log.info("  Deleted zip (%.1f GB freed)", zip_size)
        except OSError:
            pass
        return Path(npy_path)
    finally:
        shutil.rmtree(extract_dir, ignore_errors=True)


def acquire_month(year: int, month: int, lat: float, lon: float,
                  site_name: str, client, max_cloud: float,
                  max_cloud_fb: float, download_dir: Path,
                  dry_run: bool) -> tuple[Path | None, dict]:
    start, end = month_window(year, month)
    log.info("── %d-%02d : searching CDSE (cloud ≤ %.0f%%) ──", year, month, max_cloud)

    try:
        products = client.search_products(
            wkt_polygon=bbox_wkt(lat, lon),
            start_date=start,
            end_date=end,
            collection="SENTINEL-2",
            max_cloud_cover=max_cloud,
        )
    except Exception as e:
        log.error("  CDSE search failed: %s", e)
        return None, {"status": "search_failed", "error": str(e)}

    l1c = [p for p in products if "MSIL1C" in p.name]
    if not l1c and max_cloud < max_cloud_fb:
        log.info("  No products at ≤%.0f%% — retrying at ≤%.0f%%", max_cloud, max_cloud_fb)
        try:
            products = client.search_products(
                wkt_polygon=bbox_wkt(lat, lon),
                start_date=start,
                end_date=end,
                collection="SENTINEL-2",
                max_cloud_cover=max_cloud_fb,
            )
        except Exception as e:
            return None, {"status": "search_failed", "error": str(e)}
        l1c = [p for p in products if "MSIL1C" in p.name]

    log.info("  %d L1C candidates", len(l1c))
    if not l1c:
        return None, {"status": "no_products", "month": f"{year}-{month:02d}"}

    def _cc(p):
        v = getattr(p, "cloud_cover", None)
        return 100.0 if v is None else float(v)
    l1c.sort(key=_cc)
    best = l1c[0]
    cc_str = f"{_cc(best):.1f}%"

    # Check cache (any tile covering this date)
    cached = cached_for_date(best.acquisition_date[:10], site_name)
    if cached:
        log.info("  Cache hit: %s", cached[0].name)
        return cached[0], {
            "status": "cached",
            "product_name": best.name,
            "cloud_cover": getattr(best, "cloud_cover", None),
            "acquisition_date": best.acquisition_date,
        }

    if dry_run:
        log.info("  Would download %s (cloud %s)", best.name[:60], cc_str)
        return None, {
            "status": "would_download",
            "product_name": best.name,
            "cloud_cover": getattr(best, "cloud_cover", None),
            "acquisition_date": best.acquisition_date,
        }

    try:
        npy_path = download_one(client, best, site_name, download_dir)
        return npy_path, {
            "status": "downloaded",
            "product_name": best.name,
            "cloud_cover": getattr(best, "cloud_cover", None),
            "acquisition_date": best.acquisition_date,
        }
    except Exception as e:
        log.error("  Download failed: %s", e)
        return None, {"status": "download_failed", "error": str(e),
                      "product_name": best.name}


# ── Inference & S/C ───────────────────────────────────────────────────────────
def inference_and_sc(npy_path: Path, lat: float, lon: float,
                     site_name: str, detector) -> dict:
    geo_meta = find_geo_meta(npy_path)
    if geo_meta is None:
        return {"status": "no_geo_meta"}

    site_dir = OUT_DIR / site_name
    site_dir.mkdir(parents=True, exist_ok=True)
    out_tif = site_dir / f"original_{npy_path.stem}.tif"

    if not out_tif.exists():
        log.info("  Running CH4Net inference ...")
        target = np.load(npy_path)
        run_inference(target, detector, geo_meta, out_tif)
        del target
    else:
        log.info("  Inference cached: %s", out_tif.name)

    sc = compute_sc_ratio(out_tif, lat, lon)
    return {
        "tif":               str(out_tif),
        "sc_ratio":          sc.get("sc_ratio"),
        "sc_cfar":           sc.get("sc_cfar"),
        "site_mean":         sc.get("site_mean"),
        "ctrl_mean":         sc.get("ctrl_mean"),
        "ctrl_mu":           sc.get("ctrl_mu"),
        "ctrl_sigma":        sc.get("ctrl_sigma"),
        "cv_ctrl":           sc.get("cv_ctrl"),
        "cfar_thresh_ratio": sc.get("cfar_thresh_ratio"),
        "cfar_detect":       sc.get("cfar_detect"),
        "cfar_margin":       sc.get("cfar_margin"),
    }


# ── Quantification ────────────────────────────────────────────────────────────
def quantify(npy_path: Path, sc_record: dict, lat: float, lon: float,
             site_name: str, era5_client) -> dict:
    sc = sc_record.get("sc_ratio")
    if sc is None or sc <= DETECTION_THRESHOLD:
        return {"status": "below_threshold", "sc_ratio": sc}

    log.info("  S/C = %.2f ≥ %.2f → running CEMF+IME quantification", sc, DETECTION_THRESHOLD)

    site_row, site_col = lonlat_to_pixel(Path(sc_record["tif"]), lon, lat)
    half = CROP_PX_QUANT // 2
    r0, r1 = site_row - half, site_row + half
    c0, c1 = site_col - half, site_col + half
    log.info("  Crop: rows %d–%d, cols %d–%d (%.1f km)",
             r0, r1, c0, c1, CROP_PX_QUANT * 10 / 1000)

    arr = np.load(npy_path)
    b11 = arr[r0:r1, c0:c1, B11_IDX].astype(np.float32) / 255.0
    b12 = arr[r0:r1, c0:c1, B12_IDX].astype(np.float32) / 255.0
    del arr

    import rasterio
    with rasterio.open(sc_record["tif"]) as src:
        prob_full = src.read(1).astype(np.float32)
    prob = prob_full[r0:r1, c0:c1]
    del prob_full
    mask_original = (prob >= 0.18).astype(np.float32)

    m = ACQ_DATE_RE.search(npy_path.name)
    acq = m.group(1) if m else None
    if acq is None:
        return {"status": "no_acquisition_date"}
    acq_iso = f"{acq[:4]}-{acq[4:6]}-{acq[6:8]}T10:00:00Z"

    cfg = SiteCfg(
        site=site_name,
        scene_id=npy_path.stem,
        acquisition_timestamp=acq_iso,
        lat=lat,
        lon=lon,
        b11=b11,
        b12=b12,
        mask_original=mask_original,
        mask_bitemporal=None,
        era5_hour="10:00",
        ch4net_peak_probability=float(prob.max()),
    )

    try:
        record = run_quantification(cfg, dry_run=True, era5_client=era5_client)
    except Exception as e:
        log.error("  Quantification failed: %s", e)
        return {"status": "quant_failed", "error": str(e)}

    flags = []
    if record.wind_source != "ERA5_reanalysis":
        flags.append("WIND_FALLBACK")

    return {
        "status":                "quantified",
        "sc_ratio":              sc,
        "flow_rate_kgh":         record.flow_rate_kgh,
        "flow_rate_lower_kgh":   record.flow_rate_lower_kgh,
        "flow_rate_upper_kgh":   record.flow_rate_upper_kgh,
        "wind_speed_ms":         record.wind_speed_ms,
        "wind_dir_deg":          record.wind_dir_deg,
        "wind_source":           record.wind_source,
        "uncertainty_pct":       record.uncertainty_pct,
        "annual_tonnes_if_continuous": record.annual_tonnes_if_continuous,
        "n_plume_pixels":        record.n_plume_pixels,
        "governance_flags":      flags,
    }


# ── Summary helper ────────────────────────────────────────────────────────────
def print_summary(records: list[dict], site_cfg: dict, tau: float = CONFORMAL_TAU) -> None:
    total     = len(records)
    missing   = sum(1 for r in records if r.get("status") == "partial_swath")
    valid     = [r for r in records if r.get("status") not in
                 ("no_products", "search_failed", "download_failed",
                  "no_geo_meta", "partial_swath", "would_download")]
    above_tau = [r for r in valid if (r.get("sc_cfar") or 0) > tau
                 and r.get("cfar_detect")]
    quantified = [r for r in above_tau if r.get("quant", {}).get("status") == "quantified"]
    flow_rates = [r["quant"]["flow_rate_kgh"] for r in quantified
                  if r.get("quant", {}).get("flow_rate_kgh")]

    log.info("=" * 65)
    log.info("SUMMARY — %s", site_cfg["label"])
    log.info("  Total acquisition-months processed : %d", total)
    log.info("  Missing / partial-swath             : %d", missing)
    log.info("  Valid observations                  : %d", len(valid))
    log.info("  Above-threshold (τ=%.4f, CFAR gate) : %d", tau, len(above_tau))
    log.info("  Quantified records                  : %d", len(quantified))
    if flow_rates:
        import numpy as _np
        mean_kgh = _np.mean(flow_rates)
        ann_t    = mean_kgh * 8760 / 1000
        log.info("  Mean flow rate                      : %.0f kg/hr", mean_kgh)
        log.info("  Detection-weighted annual est.      : %.0f t CH₄/yr", ann_t)
        log.info("  Climate TRACE 2024 (approx.)        : %d t CH₄/yr",
                 site_cfg.get("climate_trace_ch4_2024_t", 0))
        ratio = ann_t / site_cfg["climate_trace_ch4_2024_t"] * 100 if site_cfg.get("climate_trace_ch4_2024_t") else float("nan")
        log.info("  Recovery ratio vs. Climate TRACE    : %.0f%%", ratio)
    log.info("=" * 65)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CH4Net + CEMF/IME time series on a new open-pit lignite site."
    )
    parser.add_argument("--site", required=True,
                        help="Pre-configured site name ('turow', 'welzow') or custom name with --lat/--lon.")
    parser.add_argument("--lat",  type=float, help="Latitude (for custom site).")
    parser.add_argument("--lon",  type=float, help="Longitude (for custom site).")
    parser.add_argument("--years", nargs="+", type=int,
                        default=[2021, 2022, 2023, 2024],
                        help="Years to process (default: 2021–2024).")
    parser.add_argument("--months", nargs="+", type=int,
                        default=list(range(1, 13)),
                        help="Months to process within each year (default: all 12).")
    parser.add_argument("--max-cloud", type=float, default=20.0)
    parser.add_argument("--max-cloud-fallback", type=float, default=40.0)
    parser.add_argument("--dry-run", action="store_true",
                        help="Search catalog only — no downloads or inference.")
    parser.add_argument("--skip-quant", action="store_true",
                        help="Run inference/S-C only — skip CEMF+IME quantification.")
    args = parser.parse_args()

    # Resolve site config
    site_name = args.site.lower().replace("-", "_")
    if site_name in SITES:
        site_cfg = SITES[site_name]
        lat, lon = site_cfg["lat"], site_cfg["lon"]
    elif args.lat is not None and args.lon is not None:
        lat, lon = args.lat, args.lon
        site_cfg = {
            "label": f"Custom site ({lat:.3f}°N, {lon:.3f}°E)",
            "climate_trace_ch4_2024_t": 0,
        }
    else:
        parser.error(
            f"Unknown site '{args.site}'. Known sites: {list(SITES.keys())}. "
            "For a custom site, supply --lat and --lon."
        )

    # Output paths
    out_json = Path(f"results_analysis/{site_name}_timeseries.json")
    log_path = Path(f"results_analysis/{site_name}_timeseries.log")
    download_dir = Path(f"data/downloads/{site_name}")

    # Configure logging now that we know the site name
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_path)),
        ],
    )
    log.info("Site     : %s", site_cfg.get("label", site_name))
    log.info("Coords   : %.4f°N  %.4f°E", lat, lon)
    log.info("Years    : %s", args.years)
    log.info("Dry-run  : %s", args.dry_run)

    # Load existing records (idempotent — skip already-processed months)
    existing: list[dict] = []
    if out_json.exists():
        existing = json.loads(out_json.read_text())
        log.info("Resuming — %d existing records in %s", len(existing), out_json.name)
    # Statuses that should be retried on re-run (space failures, transient errors)
    RETRYABLE_STATUSES = {
        "would_download",      # dry-run placeholder
        "download_failed",     # Errno 28 no space / network error
        "download_error",      # other download exception
        "no_products",         # CDSE catalog had nothing — worth retrying later
        "conversion_failed",   # safe_to_npy raised
        "inference_error",     # torch crash
    }
    processed_keys = {(r["year"], r["month"]) for r in existing
                      if r.get("status") not in RETRYABLE_STATUSES}

    if args.dry_run:
        # For dry-run we still need credentials/client to search catalog
        user, pw = get_credentials()
        client = CopernicusClient(username=user, password=pw)
        detector = None
        era5_client = None
    else:
        user, pw = get_credentials()
        client = CopernicusClient(username=user, password=pw)
        log.info("Loading CH4Net v8 weights ...")
        detector = CH4NetDetector(WEIGHTS)
        era5_client = ERA5Client() if not args.skip_quant else None

    # Start from existing records, but drop retryable ones so they get re-appended
    records = [r for r in existing if r.get("status") not in RETRYABLE_STATUSES]

    for year in sorted(args.years):
        for month in sorted(args.months):
            if (year, month) in processed_keys:
                log.info("── %d-%02d : already processed — skipping", year, month)
                continue

            base_record = {
                "site":  site_name,
                "year":  year,
                "month": month,
            }

            # 1. Acquire tile
            npy_path, acq_meta = acquire_month(
                year, month, lat, lon, site_name,
                client, args.max_cloud, args.max_cloud_fallback,
                download_dir, args.dry_run,
            )

            record = {**base_record, **acq_meta}

            if npy_path is None:
                records.append(record)
                out_json.write_text(json.dumps(records, indent=2))
                continue

            # 2. Partial-swath fingerprint check (S/C = 1.0 exactly → no data)
            # We run inference first; if the result is exactly 1.0 it's a swath gap.

            # 3. Inference + S/C
            if not args.dry_run:
                sc_record = inference_and_sc(npy_path, lat, lon, site_name, detector)
                record.update(sc_record)

                sc_cfar = sc_record.get("sc_cfar")
                above_conformal = (
                    sc_cfar is not None
                    and sc_cfar > CONFORMAL_TAU
                    and sc_record.get("cfar_detect", False)
                )
                record["above_conformal_threshold"] = above_conformal
                record["conformal_tau"] = CONFORMAL_TAU

                # Partial-swath fingerprint: site_mean == ctrl_mean exactly → no data
                if (sc_record.get("site_mean") is not None
                        and sc_record.get("site_mean") == sc_record.get("ctrl_mean")):
                    record["status"] = "partial_swath"
                    log.info("  Partial-swath fingerprint detected — reclassified as missing.")
                else:
                    log.info(
                        "  sc_cfar=%.4f  above_τ=%s  cfar_detect=%s",
                        sc_cfar or 0,
                        above_conformal,
                        sc_record.get("cfar_detect"),
                    )

                # 4. Quantification
                if not args.skip_quant and above_conformal:
                    quant = quantify(npy_path, sc_record, lat, lon, site_name, era5_client)
                    record["quant"] = quant
                elif not args.skip_quant:
                    record["quant"] = {"status": "below_threshold",
                                       "sc_cfar": sc_cfar}

                # 5. Disk cleanup — delete .npy now that inference + quant are done.
                # TIF is already saved; JSON has all derived metrics. Re-running a
                # month is blocked by the JSON idempotency check, not file existence.
                if npy_path is not None and npy_path.exists():
                    try:
                        npy_size = npy_path.stat().st_size / 1e9
                        npy_path.unlink()
                        log.info("  Deleted .npy (%.1f GB freed)", npy_size)
                    except OSError:
                        pass

            # Save after each record (incremental, safe to interrupt)
            records.append(record)
            out_json.write_text(json.dumps(records, indent=2))
            log.info("  Saved (%d records total) → %s", len(records), out_json.name)

    # Final summary
    if not args.dry_run:
        print_summary(records, site_cfg)
    else:
        log.info("Dry-run complete. %d months would be processed.", len(records))

    log.info("Output: %s", out_json)


if __name__ == "__main__":
    main()
