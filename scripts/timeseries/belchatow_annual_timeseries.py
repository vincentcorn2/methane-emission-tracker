"""
scripts/belchatow_annual_timeseries.py
=======================================
Build an annual Belchatow CH4Net + CEMF+IME time series for Section 4 / Figure 6
and the financial annualization in the report.

For each target month in the configured year:
  1. Search CDSE for cloud-free T34UCB Sentinel-2 L1C acquisitions
  2. Pick the lowest-cloud product not already cached
  3. Download + convert to .npy if needed
  4. Run CH4Net v8 inference, compute S/C against the site catalogue entry
  5. If S/C > 1.15 (Belchatow uses classic CFAR with skip_bitemporal=True):
     extract B11/B12, build a SiteCfg, run CEMF+IME with ERA5 wind
  6. Log every record to results_analysis/belchatow_annual_timeseries.json

Defaults to 2024 (Climate TRACE coverage year) with one acquisition per month.

Usage:
    conda activate methane
    python scripts/belchatow_annual_timeseries.py
    python scripts/belchatow_annual_timeseries.py --year 2024
    python scripts/belchatow_annual_timeseries.py --year 2024 --max-cloud 30
    python scripts/belchatow_annual_timeseries.py --dry-run
"""

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("results_analysis/belchatow_annual_timeseries.log"),
    ],
)
log = logging.getLogger("belchatow_timeseries")


# ── Site config ───────────────────────────────────────────────────────────────
SITE_NAME = "belchatow"
LAT, LON  = 51.266, 19.315
TILE_ID   = "T34UCB"
DETECTION_THRESHOLD = 1.15   # classic S/C threshold for continuous emitter
CONFORMAL_TAU       = 4.1052 # global conformal threshold at alpha=0.10

# Quantification crop size at 10m resolution. 750 px = 7.5 km × 7.5 km.
# Previous default of 500 px (5 km) saturated plume_length_m at 4980 m for
# Belchatow plumes at typical wind speeds — IME flow rates were upper-bounded.
# 750 px captures full plume extent for winds up to ~4 m/s.
CROP_PX_QUANT = 750

DOWNLOAD_DIR = Path("data/downloads/annual")
OUT_JSON     = Path("results_analysis/belchatow_annual_timeseries.json")
ACQ_DATE_RE  = re.compile(r"_(\d{8})T")


# ── Utility ──────────────────────────────────────────────────────────────────
def bbox_wkt(lat, lon, margin=0.25):
    return (
        f"POLYGON(({lon-margin} {lat-margin},{lon+margin} {lat-margin},"
        f"{lon+margin} {lat+margin},{lon-margin} {lat+margin},"
        f"{lon-margin} {lat-margin}))"
    )


def month_window(year, month):
    last_day = monthrange(year, month)[1]
    start = f"{year}-{month:02d}-01T00:00:00.000Z"
    end   = f"{year}-{month:02d}-{last_day:02d}T23:59:59.999Z"
    return start, end


def cached_for_date(date_str):
    """Return list of cached .npy paths matching the acquisition date YYYYMMDD."""
    out = []
    for p in NPY_CACHE.glob(f"*{TILE_ID}*.npy"):
        if "_ref_" in p.name or "MSIL1C" not in p.name:
            continue
        m = ACQ_DATE_RE.search(p.name)
        if m and m.group(1) == date_str.replace("-", ""):
            out.append(p)
    return sorted(out)


def get_credentials():
    user = os.environ.get("CDSE_USERNAME")
    pw   = os.environ.get("CDSE_PASSWORD")
    if user and pw:
        return user, pw
    log.info("CDSE_USERNAME / CDSE_PASSWORD env vars not set; prompting.")
    user = input("CDSE username: ")
    pw   = getpass.getpass("CDSE password: ")
    return user, pw


# ── Download / cache management ───────────────────────────────────────────────
def download_one(client, product):
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    NPY_CACHE.mkdir(parents=True, exist_ok=True)
    log.info("  Downloading %s ...", product.name[:70])
    zip_path = client.download_product(product, str(DOWNLOAD_DIR))
    if zip_path is None:
        raise RuntimeError(f"download_product returned None for {product.name}")
    log.info("  Converting to .npy ...")
    extract_dir = tempfile.mkdtemp(prefix=f"s2_{SITE_NAME}_annual_")
    try:
        npy_path, meta_path = safe_to_npy(
            zip_path=zip_path,
            output_dir=str(NPY_CACHE),
            tile_id=TILE_ID,
            acquisition_date=product.acquisition_date,
            satellite=product.satellite,
            extract_dir=extract_dir,
        )
        log.info("  Saved: %s", Path(npy_path).name)
        return Path(npy_path)
    finally:
        shutil.rmtree(extract_dir, ignore_errors=True)


def acquire_month(year, month, client, max_cloud, dry_run):
    """Return path to a usable .npy for this (year, month), downloading if needed."""
    start, end = month_window(year, month)
    log.info("── %d-%02d : searching CDSE (cloud <= %.0f%%) ──", year, month, max_cloud)

    try:
        products = client.search_products(
            wkt_polygon=bbox_wkt(LAT, LON),
            start_date=start,
            end_date=end,
            collection="SENTINEL-2",
            max_cloud_cover=max_cloud,
        )
    except Exception as e:
        log.error("  CDSE search failed: %s", e)
        return None, {"status": "search_failed", "error": str(e)}

    l1c = [p for p in products if TILE_ID in getattr(p, "tile_id", "")
           and "MSIL1C" in p.name]
    log.info("  %d L1C candidates", len(l1c))
    if not l1c:
        return None, {"status": "no_products", "month": f"{year}-{month:02d}"}

    # Sort by cloud cover, pick lowest. CDSE returns cloud_cover=None for some
    # products, which breaks comparison if multiple Nones land next to each other.
    # Treat None as "unknown" → send to back of the queue (100.0).
    def _cc(p):
        v = getattr(p, "cloud_cover", None)
        return 100.0 if v is None else v
    l1c.sort(key=_cc)
    best = l1c[0]

    # Check cache first
    cached = cached_for_date(best.acquisition_date[:10])
    if cached:
        log.info("  Already cached: %s", cached[0].name)
        return cached[0], {
            "status":           "cached",
            "product_name":     best.name,
            "cloud_cover":      getattr(best, "cloud_cover", None),
            "acquisition_date": best.acquisition_date,
        }

    if dry_run:
        return None, {
            "status":           "would_download",
            "product_name":     best.name,
            "cloud_cover":      getattr(best, "cloud_cover", None),
            "acquisition_date": best.acquisition_date,
        }

    try:
        npy_path = download_one(client, best)
        return npy_path, {
            "status":           "downloaded",
            "product_name":     best.name,
            "cloud_cover":      getattr(best, "cloud_cover", None),
            "acquisition_date": best.acquisition_date,
        }
    except Exception as e:
        log.error("  Download failed: %s", e)
        return None, {"status": "download_failed", "error": str(e),
                      "product_name": best.name}


# ── Pipeline steps per acquisition ────────────────────────────────────────────
def inference_and_sc(npy_path, detector):
    geo_meta = find_geo_meta(npy_path)
    if geo_meta is None:
        return {"status": "no_geo_meta"}

    site_dir = OUT_DIR / SITE_NAME
    site_dir.mkdir(parents=True, exist_ok=True)
    out_tif = site_dir / f"original_{npy_path.stem}.tif"

    if not out_tif.exists():
        log.info("  Running CH4Net inference ...")
        target = np.load(npy_path)
        run_inference(target, detector, geo_meta, out_tif)
        del target
    else:
        log.info("  Inference cached: %s", out_tif.name)

    sc = compute_sc_ratio(out_tif, LAT, LON)
    sc_record = {
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
    return sc_record


def quantify(npy_path, sc_record, era5_client):
    """If S/C clears threshold, run CEMF+IME with ERA5 winds."""
    sc = sc_record.get("sc_ratio")
    if sc is None or sc <= DETECTION_THRESHOLD:
        return {"status": "below_threshold", "sc_ratio": sc}

    log.info("  S/C = %.2f >= %.2f -> running quantification", sc, DETECTION_THRESHOLD)

    # Crop b11/b12/mask to CROP_PX_QUANT × CROP_PX_QUANT around the site BEFORE
    # passing to run_quantification. Without explicit cropping, IME computes
    # plume_length over the full 10980×10980 tile, which inflates L and drives
    # Q artificially small. Site row/col come from the inference TIF's CRS.
    site_row, site_col = lonlat_to_pixel(Path(sc_record["tif"]), LON, LAT)
    half = CROP_PX_QUANT // 2
    r0, r1 = site_row - half, site_row + half
    c0, c1 = site_col - half, site_col + half
    log.info("  Quantification crop: rows %d–%d, cols %d–%d (%d×%d px @ 10m = %.1f km)",
             r0, r1, c0, c1, CROP_PX_QUANT, CROP_PX_QUANT, CROP_PX_QUANT * 10 / 1000)

    arr = np.load(npy_path)
    b11 = arr[r0:r1, c0:c1, B11_IDX].astype(np.float32) / 255.0
    b12 = arr[r0:r1, c0:c1, B12_IDX].astype(np.float32) / 255.0
    del arr

    # Load mask from the inference TIF, crop to the same window
    import rasterio
    with rasterio.open(sc_record["tif"]) as src:
        prob_full = src.read(1).astype(np.float32)
    prob = prob_full[r0:r1, c0:c1]
    del prob_full
    mask_original = (prob >= 0.18).astype(np.float32)

    # Acquisition timestamp from filename
    m = ACQ_DATE_RE.search(npy_path.name)
    acq = m.group(1) if m else None
    if acq is None:
        return {"status": "no_acquisition_date"}
    acq_iso = f"{acq[:4]}-{acq[4:6]}-{acq[6:8]}T10:00:00Z"

    cfg = SiteCfg(
        site=SITE_NAME,
        scene_id=npy_path.stem,
        acquisition_timestamp=acq_iso,
        lat=LAT,
        lon=LON,
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

    # Derive governance flags from record state (canonical schema doesn't expose
    # them as a dataclass attribute; they're injected into the dict at write time
    # via apply_governance_to_record). We replicate the rules we care about here.
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


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int,
                        default=[2021, 2022, 2023, 2024],
                        help="Years to process (default: 2021..2024 = full Climate TRACE coverage)")
    parser.add_argument("--max-cloud", type=float, default=20.0,
                        help="CDSE cloud-cover ceiling (%%)")
    parser.add_argument("--max-cloud-fallback", type=float, default=40.0,
                        help="Retry ceiling if first search returns no products")
    parser.add_argument("--dry-run", action="store_true",
                        help="Search catalog only, no downloads or inference")
    parser.add_argument("--months", nargs="+", type=int,
                        default=list(range(1, 13)),
                        help="Months to process per year (default: 1..12)")
    args = parser.parse_args()

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    # Load or initialise store
    if OUT_JSON.exists():
        with open(OUT_JSON) as f:
            store = json.load(f)
        bak = OUT_JSON.with_suffix(".json.bak")
        with open(bak, "w") as f:
            json.dump(store, f, indent=2)
        log.info("Loaded existing JSON (%d records); backup -> %s",
                 len(store.get("records", [])), bak.name)
    else:
        store = {"years": args.years, "site": SITE_NAME, "records": []}

    records = store.setdefault("records", [])
    store["years"] = args.years
    store["site"] = SITE_NAME

    # Init clients
    user, pw = get_credentials()
    cdse = CopernicusClient(username=user, password=pw)
    era5 = ERA5Client()

    if not args.dry_run:
        log.info("Loading CH4Net v8 weights ...")
        detector = CH4NetDetector(WEIGHTS)
    else:
        detector = None

    # Build already-processed lookup so reruns skip months we've already done.
    # Keyed by "YYYY-MM" — matches the record["month"] format used below.
    done = {r.get("month") for r in records if r.get("month")}

    # Process each (year, month) combination
    for year in args.years:
        for month in args.months:
            month_key = f"{year}-{month:02d}"
            if month_key in done:
                log.info("=" * 65)
                log.info("── %s : already in store, skipping", month_key)
                continue

            log.info("=" * 65)
            npy_path, search_meta = acquire_month(
                year, month, cdse, args.max_cloud, args.dry_run
            )
            if npy_path is None and search_meta.get("status") == "no_products" and \
               args.max_cloud_fallback > args.max_cloud:
                log.info("  Retrying with cloud <= %.0f%% ...", args.max_cloud_fallback)
                npy_path, search_meta = acquire_month(
                    year, month, cdse, args.max_cloud_fallback, args.dry_run
                )

            rec = {
                "month":  month_key,
                "search": search_meta,
            }

            if npy_path is None or args.dry_run:
                records.append(rec)
                continue

            rec["npy"] = npy_path.name
            sc_rec = inference_and_sc(npy_path, detector)
            rec["detection"] = sc_rec

            if sc_rec.get("sc_ratio") is not None:
                quant_rec = quantify(npy_path, sc_rec, era5)
                rec["quantification"] = quant_rec
                log.info(
                    "  %s  S/C=%.2f  CFAR=%s  Q=%s kg/h",
                    rec["month"],
                    sc_rec["sc_ratio"],
                    "DETECT" if sc_rec.get("cfar_detect") else "no",
                    f"{quant_rec.get('flow_rate_kgh'):.0f}" if quant_rec.get("flow_rate_kgh") else "—",
                )

            records.append(rec)

            # Save incrementally so we don't lose progress
            with open(OUT_JSON, "w") as f:
                json.dump(store, f, indent=2)

    # ── Final summary ─────────────────────────────────────────────────────────
    log.info("=" * 65)
    log.info("Annual time series complete. Records: %d", len(records))

    detected = [
        r for r in records
        if r.get("quantification", {}).get("status") == "quantified"
    ]
    above_tau = [
        r for r in records
        if (r.get("detection") or {}).get("sc_ratio") is not None
        and r["detection"]["sc_ratio"] > CONFORMAL_TAU
    ]
    cfar_pass = [
        r for r in records
        if (r.get("detection") or {}).get("cfar_detect")
    ]

    summary = {
        "years":                      args.years,
        "site":                       SITE_NAME,
        "acquisitions_processed":     len([r for r in records if "detection" in r]),
        "detections_above_threshold": len(detected),
        "detections_above_tau":       len(above_tau),
        "detections_cfar_pass":       len(cfar_pass),
        "detection_rate":             (len(detected) / len(records)) if records else 0,
        "quantification_crop_px":     CROP_PX_QUANT,
    }

    if detected:
        flows = [r["quantification"]["flow_rate_kgh"] for r in detected
                 if r["quantification"].get("flow_rate_kgh") is not None]
        summary["mean_flow_kg_h"]      = float(np.mean(flows))
        summary["median_flow_kg_h"]    = float(np.median(flows))
        summary["min_flow_kg_h"]       = float(np.min(flows))
        summary["max_flow_kg_h"]       = float(np.max(flows))

        # ── Three annualisation framings ──────────────────────────────────────
        # We report three numbers because non-detection months at a continuous
        # emitter like Bełchatów are NOT zero-emission months (Climate TRACE
        # confirms multi-thousand-tonne emission every month including winter).
        # Non-detections are missing observations, not zero observations.

        n_det = len(flows)
        n_obs = len([r for r in records if "detection" in r])
        n_nondet = n_obs - n_det

        # (1) UPPER — assumes every day emits at the mean detected rate.
        # Biased high because detection days preferentially fire on favourable
        # atmospheric conditions (clear skies, moderate wind) which can also
        # correlate with stronger plume visibility.
        summary["annualised_t_per_yr_upper"] = float(np.mean(flows) * 8760 / 1000)

        # (2) DETECTION-FLOOR IMPUTATION — non-detection months imputed at the
        # CH4Net + Sentinel-2 detection floor for temperate latitudes. The
        # floor is the smallest Q the system can resolve above the background
        # noise; non-detections tell us Q < floor, not Q = 0. Varon (2021) and
        # Sherwin (2024) report typical S2 IME detection floors of ~200–500
        # kg/h depending on wind and surface; we use 300 kg/h as a midpoint.
        DETECTION_FLOOR_KGH = 300.0
        imputed_total = (sum(flows) + DETECTION_FLOOR_KGH * n_nondet) * 8760 / 1000 / n_obs * n_obs
        summary["annualised_t_per_yr_floor_imputed"] = float(
            (np.mean(flows) * (n_det / n_obs) +
             DETECTION_FLOOR_KGH * (n_nondet / n_obs)) * 8760 / 1000
        )
        summary["detection_floor_kgh_assumed"] = DETECTION_FLOOR_KGH

        # (3) LOWER BOUND — non-detection months treated as Q = 0. This is
        # physically implausible for a continuous emitter and serves only as
        # a strawman lower bound.
        summary["annualised_t_per_yr_lower_bound"] = float(
            np.mean(flows) * 8760 / 1000 * (n_det / n_obs)
        )

        summary["n_detections"]     = n_det
        summary["n_non_detections"] = n_nondet
        summary["n_observations"]   = n_obs

    store["summary"] = summary
    with open(OUT_JSON, "w") as f:
        json.dump(store, f, indent=2)

    # Console summary
    print("\n" + "=" * 72)
    years_str = "–".join(str(y) for y in (min(args.years), max(args.years))) \
                if len(args.years) > 1 else str(args.years[0])
    print(f"Bełchatów {years_str} time series  (quant crop {CROP_PX_QUANT} px = {CROP_PX_QUANT*10/1000:.1f} km)")
    print("=" * 72)
    print(f"Acquisitions processed:      {summary['acquisitions_processed']}")
    print(f"Detections (S/C > 1.15):     {summary['detections_above_threshold']}")
    print(f"Detections above conformal:  {summary['detections_above_tau']}")
    print(f"Detections CFAR-confirmed:   {summary['detections_cfar_pass']}")
    if detected:
        print(f"Detections / non-det / obs:  {summary['n_detections']} / {summary['n_non_detections']} / {summary['n_observations']}")
        print(f"Mean flow rate (detections): {summary['mean_flow_kg_h']:.0f} kg/h")
        print(f"Range:                       {summary['min_flow_kg_h']:.0f}–{summary['max_flow_kg_h']:.0f} kg/h")
        print(f"Annualisation — three framings (non-detections are missing obs, not zero):")
        print(f"  Upper (det-mean × 8760):           {summary['annualised_t_per_yr_upper']:.0f} t/yr")
        print(f"  Floor-imputed (Q_floor={summary['detection_floor_kgh_assumed']:.0f} kg/h):     {summary['annualised_t_per_yr_floor_imputed']:.0f} t/yr")
        print(f"  Lower bound (non-det = 0):         {summary['annualised_t_per_yr_lower_bound']:.0f} t/yr")
        print(f"Climate TRACE 2024 annual:           29,636 t/yr  (mine asset 16168)")
    print(f"Output: {OUT_JSON}")
    print("=" * 72)


if __name__ == "__main__":
    main()
