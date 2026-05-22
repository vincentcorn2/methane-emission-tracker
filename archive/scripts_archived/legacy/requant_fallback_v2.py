"""
scripts/requant_fallback_v2.py
================================
Re-quantification of `site_outside_crop` records with fallback crop sizes.

Fix vs v1: the CDSE search by product name returns multiple products from the
same date+orbit covering adjacent MGRS tiles (e.g. T34UCC, T33UYS, T33UYT next
to T34UCB). v1's filter was too loose and picked the wrong tile for ~18 of 22
re-downloads. v2 strictly requires "T34UCB" in the product name and verifies
the saved .npy filename before quantification.

For each failed record:
  1. Search CDSE by acquisition date and exact tile_id "T34UCB"
  2. Strict-filter: name must contain "T34UCB"
  3. If no T34UCB match found, log "no_t34ucb_in_search" and skip
  4. Download the matched product
  5. Verify the saved .npy filename contains "T34UCB"
  6. Run quantify at crop sizes [750, 500, 250]; first successful size wins
  7. Clean up .npy after quantification

Usage:
    conda activate methane
    export CDSE_USERNAME=<...>
    export CDSE_PASSWORD=<...>
    caffeinate -i python scripts/requant_fallback_v2.py
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
        logging.FileHandler("results_analysis/requant_fallback_v2.log"),
    ],
)
log = logging.getLogger("requant_fallback_v2")

SITE_NAME = "belchatow"
LAT, LON  = 51.266, 19.315
TILE_ID   = "T34UCB"

CROP_SIZES = [750, 500, 250]   # try 7.5km first, fall back to 5km, then 2.5km

DOWNLOAD_DIR = Path("data/downloads/requant_v2")
OUT_JSON     = Path("results_analysis/belchatow_annual_timeseries.json")
REQUANT_OUT  = Path("results_analysis/requant_fallback_v2.json")

ACQ_DATE_RE  = re.compile(r"_(\d{8})T")


def get_credentials():
    user = os.environ.get("CDSE_USERNAME")
    pw   = os.environ.get("CDSE_PASSWORD")
    if user and pw:
        return user, pw
    log.info("CDSE_USERNAME / CDSE_PASSWORD env vars not set; prompting.")
    user = input("CDSE username: ")
    pw   = getpass.getpass("CDSE password: ")
    return user, pw


def bbox_wkt(lat, lon, margin=0.25):
    return (
        f"POLYGON(({lon-margin} {lat-margin},{lon+margin} {lat-margin},"
        f"{lon+margin} {lat+margin},{lon-margin} {lat+margin},"
        f"{lon-margin} {lat-margin}))"
    )


def search_strict(client, product_name):
    """
    Search CDSE for the exact T34UCB product matching this date+orbit.

    Strategy:
      1. Parse the date from the product name (YYYYMMDD).
      2. Search CDSE for the full day window.
      3. Strict-filter: name must contain 'T34UCB' and 'MSIL1C'.
      4. If multiple T34UCB matches (different processing baselines/dates),
         prefer the exact name match; otherwise lowest cloud_cover.
    """
    m = ACQ_DATE_RE.search(product_name)
    if m is None:
        return None, "no_date_in_product_name"

    date = m.group(1)
    start = f"{date[:4]}-{date[4:6]}-{date[6:8]}T00:00:00.000Z"
    end   = f"{date[:4]}-{date[4:6]}-{date[6:8]}T23:59:59.999Z"

    try:
        products = client.search_products(
            wkt_polygon=bbox_wkt(LAT, LON),
            start_date=start,
            end_date=end,
            collection="SENTINEL-2",
            max_cloud_cover=100.0,
        )
    except Exception as e:
        return None, f"search_failed: {e}"

    # STRICT filter: T34UCB must be in the name
    t34ucb = [p for p in products
              if "T34UCB" in p.name and "MSIL1C" in p.name]

    if not t34ucb:
        log.warning("  No T34UCB product on %s; got %d products: %s",
                    date, len(products),
                    [p.name[:60] for p in products[:3]])
        return None, "no_t34ucb_in_search"

    # Prefer exact name match
    for p in t34ucb:
        if product_name in p.name or p.name.startswith(product_name[:50]):
            log.info("  Exact T34UCB match: %s", p.name[:70])
            return p, "ok"

    # Otherwise lowest cloud cover among T34UCB matches
    t34ucb.sort(key=lambda p: getattr(p, "cloud_cover", 100.0) or 100.0)
    log.info("  Closest T34UCB match: %s (cloud=%s)",
             t34ucb[0].name[:70], getattr(t34ucb[0], "cloud_cover", "?"))
    return t34ucb[0], "ok"


def download_one(client, product):
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    NPY_CACHE.mkdir(parents=True, exist_ok=True)
    zip_path = client.download_product(product, str(DOWNLOAD_DIR))
    if zip_path is None:
        raise RuntimeError(f"download_product returned None for {product.name}")
    extract_dir = tempfile.mkdtemp(prefix=f"s2_{SITE_NAME}_v2_")
    try:
        npy_path, _ = safe_to_npy(
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


def cleanup(npy_path, zip_path):
    try:
        if npy_path and Path(npy_path).exists():
            Path(npy_path).unlink()
            log.info("  Cleaned up .npy")
        geo = Path(str(npy_path).replace(".npy", "_geo.json"))
        if geo.exists():
            geo.unlink()
        if zip_path and Path(zip_path).exists():
            Path(zip_path).unlink()
    except Exception as e:
        log.warning("  Cleanup partial-failed: %s", e)


def quantify_with_crop(npy_path, sc_rec, era5_client, crop_px):
    """Try quantification at the given crop_px size. Returns dict with status."""
    half = crop_px // 2

    if not Path(npy_path).exists():
        return {"status": "npy_missing", "crop_px": crop_px}

    arr = np.load(npy_path)
    # BUG FIX: lonlat_to_pixel signature is (tif_path, lon, lat) — lon first.
    # The original max-data script and v1 of this file passed (LAT, LON) which
    # interpreted Bełchatów's latitude (51.266) as longitude and vice versa.
    # The function then projected a point in eastern Saudi Arabia into UTM 34N,
    # giving pixel coordinates ~212,000 — far outside the [0, 10980] tile range.
    # That's why every record returned crop_too_clipped at every crop size.
    site_row, site_col = lonlat_to_pixel(sc_rec["tif"], LON, LAT)
    H, W, _ = arr.shape
    r0 = max(0, site_row - half); r1 = min(H, site_row + half)
    c0 = max(0, site_col - half); c1 = min(W, site_col + half)
    if r1 <= r0 or c1 <= c0:
        del arr
        return {"status": "crop_too_clipped", "crop_px": crop_px,
                "site_row": int(site_row), "site_col": int(site_col),
                "tile_shape": [H, W]}

    b11 = arr[r0:r1, c0:c1, B11_IDX].astype(np.float32) / 255.0
    b12 = arr[r0:r1, c0:c1, B12_IDX].astype(np.float32) / 255.0
    del arr

    with rasterio.open(sc_rec["tif"]) as src:
        prob = src.read(1).astype(np.float32)[r0:r1, c0:c1]
    if prob.size == 0:
        return {"status": "prob_empty", "crop_px": crop_px}
    mask = (prob >= 0.18).astype(np.float32)
    if mask.sum() == 0:
        return {"status": "mask_empty", "crop_px": crop_px}

    m = ACQ_DATE_RE.search(Path(npy_path).name)
    if m is None:
        return {"status": "no_acq_date", "crop_px": crop_px}
    acq = m.group(1)
    acq_iso = f"{acq[:4]}-{acq[4:6]}-{acq[6:8]}T10:00:00Z"

    cfg = SiteCfg(
        site=SITE_NAME, scene_id=Path(npy_path).stem,
        acquisition_timestamp=acq_iso, lat=LAT, lon=LON,
        b11=b11, b12=b12, mask_original=mask, mask_bitemporal=None,
        era5_hour="10:00", ch4net_peak_probability=float(prob.max()),
    )
    try:
        record = run_quantification(cfg, dry_run=True, era5_client=era5_client)
    except Exception as e:
        return {"status": "quant_failed", "crop_px": crop_px, "error": str(e)}

    # QuantificationRecord doesn't expose governance flags as an attribute;
    # they're injected into the dict at write time via apply_governance_to_record.
    # Derive the only one we care about here from wind_source.
    flags = []
    if record.wind_source != "ERA5_reanalysis":
        flags.append("WIND_FALLBACK")

    return {
        "status":                "quantified",
        "crop_px":               crop_px,
        "crop_km":               crop_px / 100.0,
        "flow_rate_kgh":         record.flow_rate_kgh,
        "flow_rate_lower_kgh":   record.flow_rate_lower_kgh,
        "flow_rate_upper_kgh":   record.flow_rate_upper_kgh,
        "wind_speed_ms":         record.wind_speed_ms,
        "wind_dir_deg":          record.wind_dir_deg,
        "wind_source":           record.wind_source,
        "uncertainty_pct":       record.uncertainty_pct,
        "annual_tonnes_if_continuous": record.flow_rate_kgh * 8760 / 1000,
        "n_plume_pixels":        int(mask.sum()),
        "governance_flags":      flags,
    }


def find_inference_tif(npy_path):
    """Locate the inference TIF for this .npy, or return None if not cached."""
    site_dir = OUT_DIR / SITE_NAME
    out_tif = site_dir / f"original_{Path(npy_path).stem}.tif"
    return out_tif if out_tif.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="List which records would be retried; do not download.")
    args = ap.parse_args()

    if not OUT_JSON.exists():
        log.error("Source JSON missing: %s", OUT_JSON)
        sys.exit(1)

    store = json.loads(OUT_JSON.read_text())
    failed = [r for r in store["records"]
              if (r.get("quantification") or {}).get("status") == "site_outside_crop"]
    log.info("Failed (site_outside_crop) records to retry: %d", len(failed))

    if args.dry_run:
        for r in failed:
            log.info("  %s  product=%s",
                     r.get("acq_key", "?"),
                     r.get("product_name", "?")[:70])
        return

    user, pw = get_credentials()
    cdse = CopernicusClient(username=user, password=pw)
    era5 = ERA5Client()

    log.info("Loading CH4Net v8 weights ...")
    global detector
    detector = CH4NetDetector(WEIGHTS)

    results = {"records": [], "site": SITE_NAME, "tile_id": TILE_ID,
               "crop_sizes_tried": CROP_SIZES}
    if REQUANT_OUT.exists():
        try:
            results = json.loads(REQUANT_OUT.read_text())
        except Exception:
            pass
    already = {r["acq_key"] for r in results.get("records", [])}

    for i, rec in enumerate(failed, 1):
        key = rec.get("acq_key", f"unknown_{i}")
        if key in already:
            log.info("[%d/%d] %s  (already retried, skipping)",
                     i, len(failed), key)
            continue

        product_name = rec.get("product_name", "")
        log.info("=" * 65)
        log.info("[%d/%d] %s", i, len(failed), product_name[:70])

        product, status = search_strict(cdse, product_name)
        if product is None:
            log.warning("  Skipping — %s", status)
            results["records"].append({"acq_key": key,
                                       "original_product": product_name,
                                       "search_status": status,
                                       "outcome": "skipped"})
            REQUANT_OUT.write_text(json.dumps(results, indent=2))
            continue

        # Defense in depth: never accept a non-T34UCB product
        if "T34UCB" not in product.name:
            log.error("  search_strict returned non-T34UCB product: %s",
                      product.name)
            results["records"].append({"acq_key": key,
                                       "original_product": product_name,
                                       "search_status": "non_t34ucb_match",
                                       "matched_product": product.name,
                                       "outcome": "skipped"})
            REQUANT_OUT.write_text(json.dumps(results, indent=2))
            continue

        try:
            log.info("  Downloading %s ...", product.name[:70])
            npy_path, zip_path = download_one(cdse, product)
        except Exception as e:
            log.error("  Download failed: %s", e)
            results["records"].append({"acq_key": key,
                                       "original_product": product_name,
                                       "search_status": "ok",
                                       "matched_product": product.name,
                                       "outcome": "download_failed",
                                       "error": str(e)})
            REQUANT_OUT.write_text(json.dumps(results, indent=2))
            continue

        if "T34UCB" not in npy_path.name:
            log.error("  Saved .npy filename does not contain T34UCB: %s",
                      npy_path.name)
            cleanup(npy_path, zip_path)
            results["records"].append({"acq_key": key,
                                       "original_product": product_name,
                                       "matched_product": product.name,
                                       "outcome": "wrong_tile_saved",
                                       "saved_npy": npy_path.name})
            REQUANT_OUT.write_text(json.dumps(results, indent=2))
            continue

        # Locate or rebuild inference TIF
        out_tif = find_inference_tif(npy_path)
        if out_tif is None:
            log.info("  Running CH4Net inference (no cached TIF)...")
            geo_meta = find_geo_meta(npy_path)
            if geo_meta is None:
                log.error("  No geo metadata; skipping")
                cleanup(npy_path, zip_path)
                results["records"].append({"acq_key": key,
                                           "outcome": "no_geo_meta"})
                REQUANT_OUT.write_text(json.dumps(results, indent=2))
                continue
            site_dir = OUT_DIR / SITE_NAME
            site_dir.mkdir(parents=True, exist_ok=True)
            out_tif = site_dir / f"original_{npy_path.stem}.tif"
            target = np.load(npy_path)
            run_inference(target, detector, geo_meta, out_tif)
            del target
        else:
            log.info("  Using cached inference TIF: %s", out_tif.name)

        sc = compute_sc_ratio(out_tif, LAT, LON)
        sc_rec = {
            "tif":          str(out_tif),
            "sc_ratio":     sc.get("sc_ratio"),
            "sc_cfar":      sc.get("sc_cfar"),
            "cfar_detect":  sc.get("cfar_detect"),
        }
        log.info("  sc_ratio=%.3f  sc_cfar=%.3f  cfar=%s",
                 sc_rec["sc_ratio"] or 0,
                 sc_rec["sc_cfar"] or 0,
                 sc_rec["cfar_detect"])

        # Try crops largest-to-smallest, first success wins
        attempts = []
        winner = None
        log.info("  Trying crop sizes %s ...", CROP_SIZES)
        for crop_px in CROP_SIZES:
            res = quantify_with_crop(npy_path, sc_rec, era5, crop_px)
            attempts.append(res)
            log.info("    crop=%d px  status=%s%s",
                     crop_px, res["status"],
                     f"  Q={res['flow_rate_kgh']:.0f} kg/h"
                     if res["status"] == "quantified" else "")
            if res["status"] == "quantified":
                winner = res
                break

        cleanup(npy_path, zip_path)

        results["records"].append({
            "acq_key":          key,
            "original_product": product_name,
            "matched_product":  product.name,
            "matched_npy":      npy_path.name,
            "sc_ratio":         sc_rec["sc_ratio"],
            "sc_cfar":          sc_rec["sc_cfar"],
            "cfar_detect":      sc_rec["cfar_detect"],
            "attempts":         attempts,
            "winner":           winner,
            "outcome":          "quantified" if winner else "all_crops_failed",
        })
        REQUANT_OUT.write_text(json.dumps(results, indent=2))

    # Summary
    quantified = [r for r in results["records"] if r.get("outcome") == "quantified"]
    by_crop = {}
    for r in quantified:
        c = r["winner"]["crop_px"]
        by_crop[c] = by_crop.get(c, 0) + 1

    log.info("=" * 65)
    log.info("REQUANT FALLBACK V2 — SUMMARY")
    log.info("=" * 65)
    log.info("Total records processed: %d", len(results["records"]))
    log.info("Quantified: %d", len(quantified))
    for crop in CROP_SIZES:
        log.info("  crop=%d px (%.1f km): %d successful",
                 crop, crop / 100.0, by_crop.get(crop, 0))
    log.info("Wrote: %s", REQUANT_OUT)


if __name__ == "__main__":
    main()
