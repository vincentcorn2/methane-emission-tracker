"""
scripts/requant_fallback_crop.py
==================================
Re-quantify the 22 site_outside_crop Belchatow records using a fallback
crop strategy: try 750 → 500 → 250 px, use whichever fits.

Simultaneously calibrates the truncation bias on the 4 records where the
full 750 px crop already works — runs them at 500 and 250 too, so we can
measure how much Q changes with crop size on the same physical plume.

Output
------
results_analysis/requant_fallback.json   per-record per-crop table + summary

Disk discipline
---------------
.npy files are re-downloaded for records missing from the cache, used once,
and deleted immediately after every CEMF+IME pass for that record.
Inference TIFs in results_bitemporal/belchatow/ are reused (not regenerated).

Usage
-----
caffeinate -i python scripts/requant_fallback_crop.py
caffeinate -i python scripts/requant_fallback_crop.py --no-download   # only process cached
caffeinate -i python scripts/requant_fallback_crop.py --crops 750 500 250 100
"""
from __future__ import annotations
import argparse
import getpass
import json
import logging
import os
import re
import shutil
import statistics
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

from apply_bitemporal_diff import B11_IDX, B12_IDX, lonlat_to_pixel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("requant")

ROOT = Path(__file__).resolve().parent.parent
NPY_CACHE = ROOT / "data" / "npy_cache"
DOWNLOAD_DIR = ROOT / "data" / "downloads" / "requant_fallback"
TIMESERIES_JSON = ROOT / "results_analysis" / "belchatow_annual_timeseries.json"
OUT_JSON = ROOT / "results_analysis" / "requant_fallback.json"
TIF_DIR = ROOT / "results_bitemporal" / "belchatow"

LAT, LON = 51.266, 19.315
SITE = "belchatow"
ACQ_DATE_RE = re.compile(r"_(\d{8})T(\d{6})_")

DEFAULT_CROPS = [750, 500, 250]


def get_credentials():
    user = os.environ.get("CDSE_USERNAME")
    pw = os.environ.get("CDSE_PASSWORD")
    if user and pw:
        return user, pw
    log.info("Prompting for CDSE credentials")
    return input("CDSE username: "), getpass.getpass("CDSE password: ")


def find_npy(scene_id_or_stem: str) -> Path | None:
    """Look in npy_cache for a .npy file matching this scene/stem."""
    # Try exact stem
    candidate = NPY_CACHE / f"{scene_id_or_stem}.npy"
    if candidate.exists():
        return candidate
    # Try substring match
    for p in NPY_CACHE.glob("*.npy"):
        if scene_id_or_stem in p.stem or p.stem in scene_id_or_stem:
            return p
    return None


def find_tif(scene_id_or_stem: str) -> Path | None:
    """Look in results_bitemporal/belchatow/ for the inference TIF."""
    candidate = TIF_DIR / f"original_{scene_id_or_stem}.tif"
    if candidate.exists():
        return candidate
    for p in TIF_DIR.glob("original_*.tif"):
        if scene_id_or_stem in p.stem:
            return p
    return None


def quantify_at_crop(
    npy_path: Path,
    tif_path: Path,
    crop_px: int,
    era5_client: ERA5Client,
):
    """Run CEMF+IME at a specific crop size. Returns dict with Q, plume_length, etc.
    or a status dict if the crop doesn't fit."""
    arr = np.load(npy_path)
    H, W, _ = arr.shape

    # lonlat_to_pixel signature is (tif_path, lon, lat) — pass lon first.
    site_row, site_col = lonlat_to_pixel(tif_path, LON, LAT)
    half = crop_px // 2
    r0 = max(0, site_row - half)
    r1 = min(H, site_row + half)
    c0 = max(0, site_col - half)
    c1 = min(W, site_col + half)

    actual_h = r1 - r0
    actual_w = c1 - c0

    # Require at least 50% of the requested crop area to actually fit
    if actual_h < crop_px // 2 or actual_w < crop_px // 2:
        del arr
        return {
            "status": "crop_too_clipped",
            "actual_crop_h": int(actual_h),
            "actual_crop_w": int(actual_w),
            "requested_crop": crop_px,
        }

    b11 = arr[r0:r1, c0:c1, B11_IDX].astype(np.float32) / 255.0
    b12 = arr[r0:r1, c0:c1, B12_IDX].astype(np.float32) / 255.0
    del arr

    with rasterio.open(tif_path) as src:
        prob_full = src.read(1).astype(np.float32)
    prob = prob_full[r0:r1, c0:c1]
    del prob_full

    if prob.size == 0:
        return {"status": "empty_crop_after_bounds", "requested_crop": crop_px}

    mask = (prob >= 0.18).astype(np.float32)
    n_pixels = int(mask.sum())

    m = ACQ_DATE_RE.search(npy_path.name)
    if not m:
        return {"status": "no_acquisition_date", "requested_crop": crop_px}
    acq = m.group(1)
    acq_iso = f"{acq[:4]}-{acq[4:6]}-{acq[6:8]}T{m.group(2)[:2]}:{m.group(2)[2:4]}:00Z"

    cfg = SiteCfg(
        site=SITE,
        scene_id=npy_path.stem,
        acquisition_timestamp=acq_iso,
        lat=LAT, lon=LON,
        b11=b11, b12=b12,
        mask_original=mask, mask_bitemporal=None,
        era5_hour="10:00",
        ch4net_peak_probability=float(prob.max()) if prob.size > 0 else 0.0,
    )

    try:
        record = run_quantification(cfg, dry_run=True, era5_client=era5_client)
    except Exception as e:
        return {"status": "quant_failed", "error": str(e), "requested_crop": crop_px}

    return {
        "status": "ok",
        "requested_crop_px": crop_px,
        "actual_crop_h": int(actual_h),
        "actual_crop_w": int(actual_w),
        "flow_rate_kgh": record.flow_rate_kgh,
        "flow_rate_lower_kgh": record.flow_rate_lower_kgh,
        "flow_rate_upper_kgh": record.flow_rate_upper_kgh,
        "plume_length_m": record.plume_length_m,
        "total_mass_kg": record.total_mass_kg,
        "n_plume_pixels": record.n_plume_pixels,
        "wind_speed_ms": record.wind_speed_ms,
        "wind_dir_deg": record.wind_dir_deg,
        "wind_source": record.wind_source,
    }


def ensure_npy(scene_id: str, client, allow_download: bool) -> Path | None:
    """Return path to .npy for this scene. Re-download if missing and allowed."""
    p = find_npy(scene_id)
    if p:
        return p
    if not allow_download:
        return None
    log.info("    Re-downloading %s ...", scene_id)
    # Search CDSE for this exact product
    try:
        products = client.search_products_by_name(scene_id)
    except AttributeError:
        # Older client doesn't have a by-name method; fall back to a date search
        m = ACQ_DATE_RE.search(scene_id)
        if not m:
            log.error("    Cannot recover acquisition date from %s", scene_id)
            return None
        day = m.group(1)
        start = f"{day[:4]}-{day[4:6]}-{day[6:8]}T00:00:00.000Z"
        end = f"{day[:4]}-{day[4:6]}-{day[6:8]}T23:59:59.999Z"
        products = client.search_products(
            wkt_polygon=f"POLYGON(({LON-0.2} {LAT-0.2},{LON+0.2} {LAT-0.2},"
                        f"{LON+0.2} {LAT+0.2},{LON-0.2} {LAT+0.2},{LON-0.2} {LAT-0.2}))",
            start_date=start, end_date=end,
            collection="SENTINEL-2", max_cloud_cover=80.0,
        )
        products = [p for p in products if scene_id.split("_2")[0] in p.name
                    or p.name.startswith(scene_id[:30])]
    if not products:
        log.error("    No CDSE product matched %s", scene_id)
        return None
    product = products[0]

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = client.download_product(product, str(DOWNLOAD_DIR))
    if zip_path is None:
        log.error("    Download returned None for %s", scene_id)
        return None

    tile_match = re.search(r"_T(\d{2}[A-Z]{3})_", product.name)
    tile_id = "T" + tile_match.group(1) if tile_match else "T34UCB"

    extract_dir = tempfile.mkdtemp(prefix=f"s2_requant_")
    try:
        npy_path, _ = safe_to_npy(
            zip_path=zip_path,
            output_dir=str(NPY_CACHE),
            tile_id=tile_id,
            acquisition_date=product.acquisition_date,
            satellite=product.satellite,
            extract_dir=extract_dir,
        )
    finally:
        shutil.rmtree(extract_dir, ignore_errors=True)
        try:
            Path(zip_path).unlink()
        except OSError:
            pass

    return Path(npy_path)


def process_record(record, client, era5_client, crops, allow_download):
    """Run CEMF+IME at every crop size for one record. Returns list of result dicts."""
    scene_id = record.get("npy", "").replace(".npy", "") or record.get("scene_id") or ""
    if not scene_id:
        return {"scene_id": None, "error": "no scene_id"}

    tif = find_tif(scene_id)
    if not tif:
        return {"scene_id": scene_id, "error": "inference TIF not found"}

    npy = ensure_npy(scene_id, client, allow_download)
    if not npy:
        return {"scene_id": scene_id, "tif": str(tif), "error": "npy unavailable"}

    log.info("  Quantifying at crop sizes %s ...", crops)
    results = []
    for crop in crops:
        log.info("    crop=%d px (%.1f km) ...", crop, crop * 10 / 1000)
        r = quantify_at_crop(npy, tif, crop, era5_client)
        if r.get("status") == "ok":
            log.info("      OK: Q=%.0f kg/h  L=%.0f m  px=%d",
                     r["flow_rate_kgh"], r["plume_length_m"], r["n_plume_pixels"])
        else:
            log.info("      %s", r.get("status"))
        results.append(r)

    # Disk discipline: delete .npy now
    try:
        npy.unlink()
        for geo in NPY_CACHE.glob(f"{npy.stem}*_geo.json"):
            geo.unlink(missing_ok=True)
        log.info("    Cleaned up .npy")
    except OSError:
        pass

    return {
        "scene_id": scene_id,
        "tif": str(tif),
        "month": record.get("month"),
        "detection": record.get("detection"),
        "previous_quant_status": (record.get("quantification") or {}).get("status"),
        "results_by_crop": results,
    }


def compute_bias(all_results, crops):
    """For records where multiple crop sizes produced OK results, compute pairwise ratios."""
    bias = {}
    for crop in crops:
        if crop == max(crops):
            continue
        ratios = []
        for entry in all_results:
            rs = entry.get("results_by_crop") or []
            big = next((r for r in rs if r.get("requested_crop_px") == max(crops)
                        and r.get("status") == "ok"), None)
            small = next((r for r in rs if r.get("requested_crop_px") == crop
                          and r.get("status") == "ok"), None)
            if big and small and big.get("flow_rate_kgh") and small.get("flow_rate_kgh"):
                ratios.append(small["flow_rate_kgh"] / big["flow_rate_kgh"])
        if ratios:
            bias[f"Q_{crop}_over_Q_{max(crops)}"] = {
                "n_records": len(ratios),
                "ratios": [round(r, 4) for r in ratios],
                "mean": round(statistics.mean(ratios), 4),
                "median": round(statistics.median(ratios), 4),
                "sd": round(statistics.stdev(ratios), 4) if len(ratios) > 1 else None,
            }
    return bias


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crops", nargs="+", type=int, default=DEFAULT_CROPS,
                        help="Crop sizes in px to try, largest first")
    parser.add_argument("--no-download", action="store_true",
                        help="Skip records whose .npy is missing instead of re-downloading")
    args = parser.parse_args()

    crops = sorted(args.crops, reverse=True)

    if not TIMESERIES_JSON.exists():
        log.error("Time series not found: %s", TIMESERIES_JSON)
        sys.exit(1)

    store = json.loads(TIMESERIES_JSON.read_text())
    records = store.get("records", []) if isinstance(store, dict) else store

    target = []
    for r in records:
        qstatus = (r.get("quantification") or {}).get("status")
        det = r.get("detection") or {}
        cfar = det.get("cfar_detect")
        sc_cfar = det.get("sc_cfar") or 0
        if cfar and sc_cfar > 4.1052:
            if qstatus in ("site_outside_crop", "quantified", "npy_cleaned", "quant_failed"):
                target.append(r)

    log.info("Found %d production-rule detection records to re-quantify", len(target))
    log.info("Crop sizes: %s", crops)

    # Resume support
    existing = []
    if OUT_JSON.exists():
        existing = json.loads(OUT_JSON.read_text()).get("entries", [])
        done = {e["scene_id"] for e in existing if e.get("scene_id")}
        log.info("Resuming — %d records already processed", len(done))
        target = [r for r in target
                  if (r.get("npy", "").replace(".npy", "") or r.get("scene_id")) not in done]
        log.info("Remaining to process: %d", len(target))

    client = None
    if not args.no_download:
        user, pw = get_credentials()
        client = CopernicusClient(username=user, password=pw)
    era5 = ERA5Client()

    all_entries = list(existing)
    for i, rec in enumerate(target, 1):
        log.info("=" * 65)
        log.info("[%d/%d] %s", i, len(target), rec.get("npy") or rec.get("scene_id"))
        entry = process_record(rec, client, era5, crops, not args.no_download)
        all_entries.append(entry)

        # Save after every record
        bias = compute_bias(all_entries, crops)
        OUT_JSON.write_text(json.dumps({
            "crops_attempted": crops,
            "n_records": len(all_entries),
            "bias_factors": bias,
            "entries": all_entries,
        }, indent=2, default=str))

    # Final summary
    print()
    print("=" * 78)
    print("REQUANT FALLBACK CROP — SUMMARY")
    print("=" * 78)
    print(f"Total records processed: {len(all_entries)}")

    # Recovery counts per crop size
    for crop in crops:
        n_ok = sum(1 for e in all_entries
                   for r in (e.get("results_by_crop") or [])
                   if r.get("requested_crop_px") == crop and r.get("status") == "ok")
        print(f"  Crop {crop:4d} px ({crop*10/1000:.1f} km): {n_ok} successful quantifications")

    # Bias factors
    bias = compute_bias(all_entries, crops)
    if bias:
        print()
        print("Bias factors (Q at smaller crop / Q at largest crop, same record):")
        for key, val in bias.items():
            print(f"  {key}: n={val['n_records']}  mean={val['mean']}  median={val['median']}  "
                  f"sd={val.get('sd')}")
            print(f"    ratios: {val['ratios']}")

    # Combined annualisation using best-available crop per record
    combined_Q = []
    used_crops = []
    for e in all_entries:
        rs = e.get("results_by_crop") or []
        for crop in crops:  # largest first
            ok = next((r for r in rs if r.get("requested_crop_px") == crop and r.get("status") == "ok"), None)
            if ok:
                combined_Q.append(ok["flow_rate_kgh"])
                used_crops.append(crop)
                break

    if combined_Q:
        mean_Q = statistics.mean(combined_Q)
        median_Q = statistics.median(combined_Q)
        print()
        print(f"Combined annualisation (best-available crop per record, n={len(combined_Q)}):")
        print(f"  mean   Q = {mean_Q:.0f} kg/h  →  {mean_Q*8760/1000:.0f} t/yr  "
              f"({100*mean_Q*8760/1000/29636:.0f}% of Climate TRACE)")
        print(f"  median Q = {median_Q:.0f} kg/h  →  {median_Q*8760/1000:.0f} t/yr  "
              f"({100*median_Q*8760/1000/29636:.0f}% of Climate TRACE)")
        print()
        from collections import Counter
        cc = Counter(used_crops)
        print(f"  Crop sizes used: {dict(cc)}")

    print()
    print(f"Wrote: {OUT_JSON}")


if __name__ == "__main__":
    main()
