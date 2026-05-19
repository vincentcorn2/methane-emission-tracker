"""
scripts/belchatow_max_data.py
==============================
Maximum-coverage Belchatow time series: 2021-2024, top-N lowest-cloud
acquisitions per month, .npy deleted after inference to keep disk bounded.

Reuses the existing belchatow_annual_timeseries record schema so results
land in the same JSON (results_analysis/belchatow_annual_timeseries.json).
Skips months/acquisitions already processed.

Usage:
    conda activate methane
    caffeinate -i python scripts/belchatow_max_data.py
    caffeinate -i python scripts/belchatow_max_data.py --years 2023 2024 --n-per-month 3
    caffeinate -i python scripts/belchatow_max_data.py --max-cloud 30 --keep-npy
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
from pathlib import Path

import numpy as np
import rasterio

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
        logging.FileHandler("results_analysis/belchatow_max_data.log"),
    ],
)
log = logging.getLogger("belchatow_max_data")


# Site config
SITE_NAME = "belchatow"
LAT, LON  = 51.266, 19.315
TILE_ID   = "T34UCB"
CROP_PX_QUANT = 750

DOWNLOAD_DIR = Path("data/downloads/maxdata")
OUT_JSON     = Path("results_analysis/belchatow_annual_timeseries.json")
ACQ_DATE_RE  = re.compile(r"_(\d{8})T")


def bbox_wkt(lat, lon, margin=0.25):
    return (
        f"POLYGON(({lon-margin} {lat-margin},{lon+margin} {lat-margin},"
        f"{lon+margin} {lat+margin},{lon-margin} {lat+margin},"
        f"{lon-margin} {lat-margin}))"
    )


def month_window(year, month):
    last_day = monthrange(year, month)[1]
    return (
        f"{year}-{month:02d}-01T00:00:00.000Z",
        f"{year}-{month:02d}-{last_day:02d}T23:59:59.999Z",
    )


def acq_key(product_name):
    """Stable key for an acquisition record: YYYY-MM-DD/product-id."""
    m = ACQ_DATE_RE.search(product_name)
    return product_name if m is None else f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:8]}/{product_name[:60]}"


def get_credentials():
    user = os.environ.get("CDSE_USERNAME")
    pw   = os.environ.get("CDSE_PASSWORD")
    if user and pw:
        return user, pw
    log.info("CDSE_USERNAME / CDSE_PASSWORD env vars not set; prompting.")
    user = input("CDSE username: ")
    pw   = getpass.getpass("CDSE password: ")
    return user, pw


def download_one(client, product):
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    NPY_CACHE.mkdir(parents=True, exist_ok=True)
    log.info("  Downloading %s ...", product.name[:70])
    zip_path = client.download_product(product, str(DOWNLOAD_DIR))
    if zip_path is None:
        raise RuntimeError(f"download_product returned None for {product.name}")
    extract_dir = tempfile.mkdtemp(prefix=f"s2_{SITE_NAME}_max_")
    try:
        npy_path, meta_path = safe_to_npy(
            zip_path=zip_path,
            output_dir=str(NPY_CACHE),
            tile_id=TILE_ID,
            acquisition_date=product.acquisition_date,
            satellite=product.satellite,
            extract_dir=extract_dir,
        )
        return Path(npy_path), zip_path
    finally:
        shutil.rmtree(extract_dir, ignore_errors=True)


def cleanup_after_inference(npy_path, zip_path, keep_npy):
    """Delete .npy and source .zip after inference TIF is written."""
    if keep_npy:
        return
    try:
        if npy_path and Path(npy_path).exists():
            Path(npy_path).unlink()
            log.info("  Cleaned up .npy")
        geo = Path(str(npy_path).replace(".npy", "_geo.json"))
        if geo.exists():
            geo.unlink()
        if zip_path and Path(zip_path).exists():
            Path(zip_path).unlink()
            log.info("  Cleaned up source .zip")
    except Exception as e:
        log.warning("  Cleanup partial-failed: %s", e)


def inference_and_sc(npy_path):
    geo_meta = find_geo_meta(npy_path)
    if geo_meta is None:
        return None, {"status": "no_geo_meta"}
    site_dir = OUT_DIR / SITE_NAME
    site_dir.mkdir(parents=True, exist_ok=True)
    out_tif = site_dir / f"original_{Path(npy_path).stem}.tif"

    if not out_tif.exists():
        target = np.load(npy_path)
        run_inference(target, detector, geo_meta, out_tif)
        del target

    sc = compute_sc_ratio(out_tif, LAT, LON)
    return out_tif, {
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


def quantify(npy_path, sc_rec, era5_client):
    if sc_rec.get("sc_ratio") is None or sc_rec["sc_ratio"] <= 1.15:
        return {"status": "below_threshold", "sc_ratio": sc_rec.get("sc_ratio")}
    if not Path(npy_path).exists():
        # NPY was cleaned up; can't quantify without raw bands
        return {"status": "npy_cleaned", "sc_ratio": sc_rec["sc_ratio"]}

    arr = np.load(npy_path)
    # lonlat_to_pixel signature is (tif_path, lon, lat). Earlier versions passed
    # (LAT, LON), interpreting Bełchatów latitude as longitude — projecting the
    # point to Saudi Arabia and triggering "site_outside_crop" on every record.
    # Pass arguments in the correct order.
    site_row, site_col = lonlat_to_pixel(sc_rec["tif"], LON, LAT)
    half = CROP_PX_QUANT // 2
    H, W, _ = arr.shape
    r0 = max(0, site_row - half); r1 = min(H, site_row + half)
    c0 = max(0, site_col - half); c1 = min(W, site_col + half)

    # Empty-crop guard. Happens on orbit-edge tiles (R079 / R108) where the site
    # falls outside or right at the swath boundary. Without this guard,
    # prob.max() below blows up on a zero-size array.
    if r1 <= r0 or c1 <= c0:
        del arr
        return {
            "status":   "site_outside_crop",
            "sc_ratio": sc_rec.get("sc_ratio"),
            "note":     f"site pixel ({site_row}, {site_col}) outside tile ({H}, {W}) "
                        f"after crop_px={CROP_PX_QUANT}",
        }

    b11 = arr[r0:r1, c0:c1, B11_IDX].astype(np.float32) / 255.0
    b12 = arr[r0:r1, c0:c1, B12_IDX].astype(np.float32) / 255.0
    del arr

    with rasterio.open(sc_rec["tif"]) as src:
        prob_full = src.read(1).astype(np.float32)
    prob = prob_full[r0:r1, c0:c1]

    # Secondary guard — if the cropped probability map is empty or all-zero, the
    # crop is degenerate even if the bounds themselves were non-empty.
    if prob.size == 0:
        return {"status": "site_outside_crop", "sc_ratio": sc_rec.get("sc_ratio")}

    mask = (prob >= 0.18).astype(np.float32)

    m = ACQ_DATE_RE.search(Path(npy_path).name)
    acq = m.group(1) if m else None
    acq_iso = f"{acq[:4]}-{acq[4:6]}-{acq[6:8]}T10:00:00Z" if acq else None

    cfg = SiteCfg(
        site=SITE_NAME, scene_id=Path(npy_path).stem,
        acquisition_timestamp=acq_iso, lat=LAT, lon=LON,
        b11=b11, b12=b12, mask_original=mask, mask_bitemporal=None,
        era5_hour="10:00", ch4net_peak_probability=float(prob.max()),
    )
    try:
        record = run_quantification(cfg, dry_run=True, era5_client=era5_client)
    except Exception as e:
        return {"status": "quant_failed", "error": str(e)}

    # QuantificationRecord doesn't expose governance flags as an attribute;
    # they're injected into the dict at write time via apply_governance_to_record.
    # Derive the only one we care about here from wind_source.
    flags = []
    if record.wind_source != "ERA5_reanalysis":
        flags.append("WIND_FALLBACK")

    return {
        "status": "quantified",
        "sc_ratio":               sc_rec["sc_ratio"],
        "flow_rate_kgh":          record.flow_rate_kgh,
        "flow_rate_lower_kgh":    record.flow_rate_lower_kgh,
        "flow_rate_upper_kgh":    record.flow_rate_upper_kgh,
        "wind_speed_ms":          record.wind_speed_ms,
        "wind_dir_deg":           record.wind_dir_deg,
        "wind_source":            record.wind_source,
        "uncertainty_pct":        record.uncertainty_pct,
        "annual_tonnes_if_continuous": record.flow_rate_kgh * 8760 / 1000,
        "n_plume_pixels":         int(mask.sum()),
        "governance_flags":       flags,
    }


def already_processed(store, key):
    for r in store.get("records", []):
        if r.get("acq_key") == key:
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int,
                        default=[2021, 2022, 2023, 2024])
    parser.add_argument("--months", nargs="+", type=int,
                        default=list(range(1, 13)))
    parser.add_argument("--n-per-month", type=int, default=3,
                        help="Top N lowest-cloud acquisitions to process per month")
    parser.add_argument("--max-cloud", type=float, default=20.0)
    parser.add_argument("--max-cloud-fallback", type=float, default=40.0)
    parser.add_argument("--keep-npy", action="store_true",
                        help="Don't delete .npy after inference (uses much more disk)")
    args = parser.parse_args()

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    if OUT_JSON.exists():
        store = json.loads(OUT_JSON.read_text())
        bak = OUT_JSON.with_suffix(".json.bak")
        bak.write_text(json.dumps(store, indent=2))
        log.info("Loaded existing JSON (%d records); backup -> %s",
                 len(store.get("records", [])), bak.name)
    else:
        store = {"site": SITE_NAME, "records": []}
    store.setdefault("records", [])

    user, pw = get_credentials()
    cdse = CopernicusClient(username=user, password=pw)
    era5 = ERA5Client()

    log.info("Loading CH4Net v8 weights ...")
    global detector
    detector = CH4NetDetector(WEIGHTS)

    total_new = 0
    for year in args.years:
        for month in args.months:
            start, end = month_window(year, month)
            log.info("=" * 65)
            log.info("── %d-%02d : searching CDSE (cloud <= %.0f%%) ──",
                     year, month, args.max_cloud)
            try:
                products = cdse.search_products(
                    wkt_polygon=bbox_wkt(LAT, LON),
                    start_date=start, end_date=end,
                    collection="SENTINEL-2",
                    max_cloud_cover=args.max_cloud,
                )
            except Exception as e:
                log.error("  CDSE search failed: %s", e)
                continue

            l1c = [p for p in products
                   if TILE_ID in getattr(p, "tile_id", "")
                   and "MSIL1C" in p.name]
            if not l1c and args.max_cloud_fallback > args.max_cloud:
                log.info("  Retrying at cloud <= %.0f%%", args.max_cloud_fallback)
                try:
                    products = cdse.search_products(
                        wkt_polygon=bbox_wkt(LAT, LON),
                        start_date=start, end_date=end,
                        collection="SENTINEL-2",
                        max_cloud_cover=args.max_cloud_fallback,
                    )
                    l1c = [p for p in products
                           if TILE_ID in getattr(p, "tile_id", "")
                           and "MSIL1C" in p.name]
                except Exception as e:
                    log.error("  Fallback search failed: %s", e)
                    continue

            l1c.sort(key=lambda p: 100.0 if getattr(p, "cloud_cover", None) is None
                     else p.cloud_cover)
            l1c = l1c[: args.n_per_month]
            log.info("  %d candidates to process", len(l1c))

            for product in l1c:
                key = acq_key(product.name)
                if already_processed(store, key):
                    log.info("  Skip (already processed): %s", product.name[:60])
                    continue

                log.info("  Processing %s (cloud=%s%%)",
                         product.name[:60],
                         getattr(product, "cloud_cover", "?"))

                npy_path, zip_path = None, None
                try:
                    # Use cache if .npy already on disk
                    existing = list(NPY_CACHE.glob(
                        f"*{product.name.replace('.SAFE', '')[:60]}*.npy"))
                    if existing:
                        npy_path = existing[0]
                        log.info("  Using cached .npy: %s", npy_path.name)
                    else:
                        npy_path, zip_path = download_one(cdse, product)
                except Exception as e:
                    log.error("  Download failed: %s", e)
                    store["records"].append({
                        "acq_key":          key,
                        "month":            f"{year}-{month:02d}",
                        "product_name":     product.name,
                        "search":           {"status": "download_failed",
                                             "error": str(e),
                                             "cloud_cover": getattr(product, "cloud_cover", None)},
                    })
                    OUT_JSON.write_text(json.dumps(store, indent=2))
                    continue

                rec = {
                    "acq_key":         key,
                    "month":           f"{year}-{month:02d}",
                    "product_name":    product.name,
                    "search": {
                        "status":           "downloaded" if zip_path else "cached",
                        "cloud_cover":      getattr(product, "cloud_cover", None),
                        "acquisition_date": product.acquisition_date,
                    },
                    "npy": Path(npy_path).name,
                }

                # Inference + S/C
                out_tif, sc_rec = inference_and_sc(npy_path)
                rec["detection"] = sc_rec
                if sc_rec.get("sc_ratio") is not None and sc_rec["sc_ratio"] > 1.15:
                    log.info("    S/C=%.2f >= 1.15 -> quantifying", sc_rec["sc_ratio"])
                    rec["quantification"] = quantify(npy_path, sc_rec, era5)
                    q = rec["quantification"]
                    if q.get("status") == "quantified":
                        log.info("    Q=%.0f kg/h [%.0f-%.0f]",
                                 q["flow_rate_kgh"],
                                 q["flow_rate_lower_kgh"],
                                 q["flow_rate_upper_kgh"])
                else:
                    log.info("    S/C=%s  CFAR=%s",
                             f"{sc_rec.get('sc_ratio'):.2f}" if sc_rec.get("sc_ratio") else "—",
                             "DETECT" if sc_rec.get("cfar_detect") else "no")

                # Cleanup .npy + .zip to keep disk bounded
                cleanup_after_inference(npy_path, zip_path, args.keep_npy)

                store["records"].append(rec)
                OUT_JSON.write_text(json.dumps(store, indent=2))
                total_new += 1

    log.info("=" * 65)
    log.info("Done. New records added: %d. Total: %d",
             total_new, len(store["records"]))


if __name__ == "__main__":
    main()
