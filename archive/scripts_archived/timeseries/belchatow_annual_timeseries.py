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

Defaults to 2019–2025. Evaluates all cloud-free acquisitions per month (not just
the lowest-cloud one), giving 3–6× more observations per month. Climate TRACE
reference values for 2025 are model projections.

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

_PROJECT_ROOT = Path(__file__).resolve().parents[2]   # methane-api/
sys.path.insert(0, str(_PROJECT_ROOT))                  # for src.*
sys.path.insert(0, str(_PROJECT_ROOT / "scripts" / "detection"))  # for apply_bitemporal_diff

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
LAT, LON  = 51.242, 19.275   # Climate TRACE KWB Bełchatów coal mine centroid (asset 16168)
                               # Previously 51.266, 19.315 (power station — wrong)
TILE_ID   = "T34UCB"
DETECTION_THRESHOLD = 1.15   # classic S/C threshold for continuous emitter
CONFORMAL_TAU       = 3.5796 # global conformal τ at α=0.10, n_cal=35 (calibrated_threshold.json)
                              # Continental ecoregion Mondrian τ = 4.1052 — not used here;
                              # report uses global guarantee across all n=35 non-emitter sites

# ── S/C control crop offsets (asymmetric — mine spans ~21 km E-W) ─────────────
# N/S: 0.20° (~22 km) clears the ~4.2 km mine height comfortably.
# E:   0.30° (~33 km) clears the eastern mine edge at 19.400°E.
# W:   0.39° (~43 km) clears the western mine edge at 19.097°E with extra margin.
SC_OFFSET_N = 0.20
SC_OFFSET_S = 0.20
SC_OFFSET_E = 0.30
SC_OFFSET_W = 0.39

# ── KWB Bełchatów mine polygon — quantification boundary ─────────────────────
# Exact OSM boundary corners. The quantification crop is taken as the bounding
# box of this polygon; pixels outside the polygon are masked before CEMF+IME.
# Previously a 750 px (7.5 km) square centred on the wrong coords (power station).
MINE_POLYGON_LATLON = [
    (51.257,  19.097),   # NW
    (51.2566, 19.390),   # NE
    (51.219,  19.3996),  # SE
    (51.2185, 19.099),   # SW
]

DOWNLOAD_DIR = Path("data/downloads/annual")
OUT_JSON     = Path("results_analysis/belchatow_annual_timeseries.json")
ACQ_DATE_RE  = re.compile(r"_(\d{8})T")

# Climate TRACE CH4 reference (asset 16168, monthly, t CH4)
CT_CH4_CSV   = Path("data/16168_climate_trace_ch4.csv")


def load_ct_ch4() -> dict:
    """Return {(year, month): t_ch4} from the Climate TRACE monthly CSV."""
    import csv
    ct = {}
    if not CT_CH4_CSV.exists():
        log.warning("Climate TRACE CSV not found at %s — CT comparison disabled", CT_CH4_CSV)
        return ct
    with open(CT_CH4_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("gas", "").strip().lower() != "ch4":
                continue
            # start_time format: "2024-1-1"
            parts = row["start_time"].split("-")
            key = (int(parts[0]), int(parts[1]))
            ct[key] = float(row["emissions_quantity"])
    log.info("Loaded %d monthly CT CH4 records (2021–2025)", len(ct))
    return ct


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
def _zip_is_valid(zip_path):
    """Return True if the file exists and is a readable zip."""
    import zipfile
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.namelist()
        return True
    except Exception:
        return False


def download_one(client, product):
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    NPY_CACHE.mkdir(parents=True, exist_ok=True)
    log.info("  Downloading %s ...", product.name[:70])
    zip_path = client.download_product(product, str(DOWNLOAD_DIR))
    if zip_path is None:
        raise RuntimeError(f"download_product returned None for {product.name}")

    # Guard against corrupt ZIPs from previous interrupted runs
    if not _zip_is_valid(zip_path):
        log.warning("  ZIP appears corrupt — deleting and re-downloading: %s",
                    Path(zip_path).name)
        try:
            Path(zip_path).unlink()
        except Exception:
            pass
        zip_path = client.download_product(product, str(DOWNLOAD_DIR))
        if zip_path is None or not _zip_is_valid(zip_path):
            raise RuntimeError(f"Re-download also produced corrupt ZIP for {product.name}")
        log.info("  Re-download OK: %s", Path(zip_path).name)

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
        # Delete the ZIP immediately — .npy is the working copy
        try:
            Path(zip_path).unlink()
            log.info("  Deleted ZIP: %s", Path(zip_path).name)
        except Exception as e:
            log.warning("  Could not delete ZIP %s: %s", zip_path, e)
        return Path(npy_path)
    finally:
        shutil.rmtree(extract_dir, ignore_errors=True)


def acquire_all_month(year, month, client, max_cloud, dry_run):
    """Return list of {"npy_path", "search"} dicts — one per cloud-free L1C
    acquisition this month.  All candidates are downloaded (or pulled from cache),
    not just the lowest-cloud one.  Provides more observations per month and
    avoids missing emission events that occurred on a non-optimal acquisition day.

    Returns a single-element list with status "no_products" or "search_failed"
    when no usable acquisitions exist.
    """
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
        return [{"npy_path": None,
                 "search":   {"status": "search_failed", "error": str(e)}}]

    l1c = [p for p in products if TILE_ID in getattr(p, "tile_id", "")
           and "MSIL1C" in p.name]
    log.info("  %d L1C candidates", len(l1c))
    if not l1c:
        return [{"npy_path": None,
                 "search":   {"status": "no_products", "month": f"{year}-{month:02d}"}}]

    # Sort by cloud cover ascending; treat None as 100.0
    def _cc(p):
        v = getattr(p, "cloud_cover", None)
        return 100.0 if v is None else v
    l1c.sort(key=_cc)

    results = []
    for product in l1c:
        meta = {
            "product_name":     product.name,
            "cloud_cover":      getattr(product, "cloud_cover", None),
            "acquisition_date": product.acquisition_date,
        }
        cached = cached_for_date(product.acquisition_date[:10])
        if cached:
            log.info("  Cached: %s", cached[0].name)
            meta["status"] = "cached"
            results.append({"npy_path": cached[0], "search": meta})
            continue

        if dry_run:
            meta["status"] = "would_download"
            results.append({"npy_path": None, "search": meta})
            continue

        try:
            npy_path = download_one(client, product)
            meta["status"] = "downloaded"
            results.append({"npy_path": npy_path, "search": meta})
        except Exception as e:
            log.error("  Download failed for %s: %s", product.name[:60], e)
            meta["status"] = "download_failed"
            meta["error"]  = str(e)
            results.append({"npy_path": None, "search": meta})

    return results


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

    sc = compute_sc_ratio(out_tif, LAT, LON,
                          offset_n=SC_OFFSET_N, offset_s=SC_OFFSET_S,
                          offset_e=SC_OFFSET_E, offset_w=SC_OFFSET_W)

    # Uniform-field guard: if site_mean ≈ ctrl_mean to high precision, CH4Net
    # produced a spatially uniform probability field (spectrally homogeneous
    # scene — snow, uniform bright surface).  S/C ratio is floating-point noise.
    # These pass the B11 non-zero swath check, so they need a separate flag.
    sm = sc.get("site_mean") or 0.0
    cm = sc.get("ctrl_mean") or 0.0
    uniform_field = bool(sm > 0 and abs(sm - cm) < 1e-4)

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
        "uniform_field":     uniform_field,
    }
    return sc_record


def quantify(npy_path, sc_record, era5_client):
    """If S/C clears threshold, run CEMF+IME with ERA5 winds."""
    sc = sc_record.get("sc_ratio")
    if sc is None or sc <= DETECTION_THRESHOLD:
        return {"status": "below_threshold", "sc_ratio": sc}

    log.info("  S/C = %.2f >= %.2f -> running quantification", sc, DETECTION_THRESHOLD)

    # ── Polygon-bounded quantification crop ───────────────────────────────────
    # Convert the mine polygon corners to pixel coordinates, take the bounding
    # box of those pixels, then mask out anything outside the polygon.
    # This replaces the old 750 px square (7.5 km) centred on the wrong coords.
    import rasterio
    from rasterio.features import geometry_mask
    from shapely.geometry import Polygon as ShapelyPolygon

    tif_path = Path(sc_record["tif"])
    with rasterio.open(tif_path) as src:
        prob_full = src.read(1).astype(np.float32)
        H, W = prob_full.shape
        transform = src.transform

        # Convert polygon lat/lon → pixel row/col
        poly_rows, poly_cols = [], []
        for plat, plon in MINE_POLYGON_LATLON:
            pr, pc = lonlat_to_pixel(tif_path, plon, plat)
            poly_rows.append(pr)
            poly_cols.append(pc)

        r0 = max(0,  min(poly_rows))
        r1 = min(H,  max(poly_rows))
        c0 = max(0,  min(poly_cols))
        c1 = min(W,  max(poly_cols))
        log.info("  Mine polygon bounding box: rows %d–%d, cols %d–%d "
                 "(%d×%d px @ 10m = %.1f×%.1f km)",
                 r0, r1, c0, c1, r1-r0, c1-c0,
                 (r1-r0)*10/1000, (c1-c0)*10/1000)

        # Build polygon in UTM (same CRS as the raster) and rasterise.
        # geometry_mask expects geometry coords to match the raster's CRS —
        # passing pixel-space coords would produce a completely wrong mask.
        poly_xy = [
            (transform.c + c * transform.a, transform.f + r * transform.e)
            for r, c in zip(poly_rows, poly_cols)
        ]
        poly_crs = ShapelyPolygon(poly_xy)
        window_transform = rasterio.transform.from_bounds(
            transform.c + c0 * transform.a,   # west  (UTM easting)
            transform.f + r1 * transform.e,   # south (UTM northing, e is negative)
            transform.c + c1 * transform.a,   # east
            transform.f + r0 * transform.e,   # north
            c1 - c0, r1 - r0,
        )
        mine_mask = ~geometry_mask(
            [poly_crs.__geo_interface__],
            out_shape=(r1 - r0, c1 - c0),
            transform=window_transform,
            invert=False,
        )   # True inside polygon, False outside

    prob = prob_full[r0:r1, c0:c1]
    del prob_full

    arr = np.load(npy_path)
    b11 = arr[r0:r1, c0:c1, B11_IDX].astype(np.float32) / 255.0
    b12 = arr[r0:r1, c0:c1, B12_IDX].astype(np.float32) / 255.0
    del arr

    # Mask: CH4Net threshold AND inside mine polygon
    mask_original = ((prob >= 0.18) & mine_mask).astype(np.float32)
    log.info("  Polygon mask: %d / %d px active (%.1f%% of bounding box)",
             int(mine_mask.sum()), mine_mask.size,
             100 * mine_mask.sum() / mine_mask.size)

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
                        default=[2019, 2020, 2021, 2022, 2023, 2024, 2025],
                        help="Years to process (default: 2019–2025; CT values for 2025 are model projections)")
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
    ct_ch4 = load_ct_ch4()

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

    # ── Processed-acquisition lookup ──────────────────────────────────────────
    # Records are per-acquisition.  Track by product_name so reruns skip
    # acquisitions already handled, while download_failed ones remain retryable.
    done_products = {
        r.get("search", {}).get("product_name")
        for r in records
        if "detection" in r and r.get("search", {}).get("product_name")
    }
    done_no_products = {
        r.get("month")
        for r in records
        if r.get("search", {}).get("status") == "no_products"
    }

    def _save():
        with open(OUT_JSON, "w") as f:
            json.dump(store, f, indent=2)

    # Process each (year, month) combination
    for year in args.years:
        for month in args.months:
            month_key = f"{year}-{month:02d}"
            if month_key in done_no_products:
                log.info("=" * 65)
                log.info("── %s : no products (confirmed), skipping", month_key)
                continue

            log.info("=" * 65)
            acquisitions = acquire_all_month(
                year, month, cdse, args.max_cloud, args.dry_run
            )
            if (len(acquisitions) == 1
                    and acquisitions[0]["search"].get("status") == "no_products"
                    and args.max_cloud_fallback > args.max_cloud):
                log.info("  Retrying with cloud <= %.0f%% ...", args.max_cloud_fallback)
                acquisitions = acquire_all_month(
                    year, month, cdse, args.max_cloud_fallback, args.dry_run
                )

            first_status = acquisitions[0]["search"].get("status")
            if first_status in ("no_products", "search_failed"):
                if not any(r.get("month") == month_key
                           and r.get("search", {}).get("status") == first_status
                           for r in records):
                    rec = {
                        "month":          month_key,
                        "search":         acquisitions[0]["search"],
                        "ct_ch4_t_month": ct_ch4.get((year, month)),
                    }
                    records.append(rec)
                    _save()
                continue

            for acq in acquisitions:
                search_meta  = acq["search"]
                npy_path     = acq["npy_path"]
                product_name = search_meta.get("product_name", "")
                acq_date     = search_meta.get("acquisition_date", "")[:10]

                if product_name in done_products:
                    log.info("  Already processed: %s", product_name[:60])
                    continue

                rec = {
                    "month":            month_key,
                    "acquisition_date": acq_date,
                    "search":           search_meta,
                    "ct_ch4_t_month":   ct_ch4.get((year, month)),
                }

                if npy_path is None or args.dry_run:
                    records.append(rec)
                    _save()
                    continue

                rec["npy"] = npy_path.name
                sc_rec = inference_and_sc(npy_path, detector)
                rec["detection"] = sc_rec

                uf = sc_rec.get("uniform_field", False)
                if uf:
                    log.info("  %s [%s]  ← UNIFORM FIELD (site_mean≈ctrl_mean=%.5f) — excluded",
                             month_key, acq_date, sc_rec.get("site_mean", 0))

                if sc_rec.get("sc_ratio") is not None and not uf:
                    quant_rec = quantify(npy_path, sc_rec, era5)
                    rec["quantification"] = quant_rec
                    log.info(
                        "  %s [%s]  S/C=%.2f  CFAR=%s  Q=%s kg/h",
                        month_key, acq_date,
                        sc_rec["sc_ratio"],
                        "DETECT" if sc_rec.get("cfar_detect") else "no",
                        f"{quant_rec.get('flow_rate_kgh'):.0f}"
                        if quant_rec.get("flow_rate_kgh") else "—",
                    )
                elif uf:
                    rec["quantification"] = {"status": "uniform_field_excluded"}

                records.append(rec)
                done_products.add(product_name)
                _save()

    # ── Final summary ─────────────────────────────────────────────────────────
    log.info("=" * 65)
    log.info("Annual time series complete. Records: %d", len(records))

    # ── Per-acquisition stats ────────────────────────────────────────────────
    detected = [
        r for r in records
        if r.get("quantification", {}).get("status") == "quantified"
    ]
    above_tau = [
        r for r in records
        if (r.get("detection") or {}).get("sc_ratio") is not None
        and not r.get("detection", {}).get("uniform_field")
        and r["detection"]["sc_ratio"] > CONFORMAL_TAU
    ]
    cfar_pass = [
        r for r in records
        if (r.get("detection") or {}).get("cfar_detect")
        and not r.get("detection", {}).get("uniform_field")
    ]

    # ── Month-level aggregation for annualisation ────────────────────────────
    # Valid acquisition: has detection result AND is not uniform-field.
    # A valid observation month has at least one valid acquisition.
    valid_acq = [
        r for r in records
        if "detection" in r
        and not r.get("detection", {}).get("uniform_field")
    ]
    obs_months = sorted({r["month"] for r in valid_acq})
    det_months = sorted({r["month"] for r in detected})

    summary = {
        "years":                      args.years,
        "site":                       SITE_NAME,
        "acquisitions_processed":     len([r for r in records if "detection" in r]),
        "uniform_field_excluded":     len([r for r in records
                                          if r.get("detection", {}).get("uniform_field")]),
        "valid_acquisitions":         len(valid_acq),
        "n_observed_months":          len(obs_months),
        "detection_acquisitions":     len(detected),
        "detection_months":           len(det_months),
        "detections_above_tau":       len(above_tau),
        "detections_cfar_pass":       len(cfar_pass),
        "detection_rate_by_month":    (len(det_months) / len(obs_months))
                                       if obs_months else 0,
        "quantification_boundary":    "KWB mine polygon (~21km × 4.2km)",
    }

    if detected:
        flows = [r["quantification"]["flow_rate_kgh"] for r in detected
                 if r["quantification"].get("flow_rate_kgh") is not None]
        summary["mean_flow_kg_h"]      = float(np.mean(flows))
        summary["median_flow_kg_h"]    = float(np.median(flows))
        summary["min_flow_kg_h"]       = float(np.min(flows))
        summary["max_flow_kg_h"]       = float(np.max(flows))

        n_det_mo  = len(det_months)
        n_obs_mo  = len(obs_months)
        n_ndet_mo = n_obs_mo - n_det_mo
        mean_q    = float(np.mean(flows))

        # ── Three annualisation framings (month-level) ────────────────────────
        # Non-detection months at a continuous emitter like Bełchatów are NOT
        # zero-emission months (Climate TRACE confirms emissions every month).
        # Non-detections are missing observations, not zero observations.

        # (1) UPPER — assumes every month emits at the mean detected rate.
        summary["annualised_t_per_yr_upper"] = float(mean_q * 8760 / 1000)

        # (2) DETECTION-FLOOR IMPUTATION — non-detection months imputed at the
        # empirical CH4Net + S2 detection floor.  Our smallest resolved Q is
        # 282 kg/h (2024-02); we use 300 kg/h as the floor.  This is consistent
        # with the enhanced sensitivity of CH4Net v8 (Vaughan et al., 2024)
        # relative to MBMP (Varon et al., 2021, min. 2,600–3,500 kg/h).
        DETECTION_FLOOR_KGH = 300.0
        summary["annualised_t_per_yr_floor_imputed"] = float(
            (mean_q * (n_det_mo / n_obs_mo) +
             DETECTION_FLOOR_KGH * (n_ndet_mo / n_obs_mo)) * 8760 / 1000
        )
        summary["detection_floor_kgh_assumed"] = DETECTION_FLOOR_KGH

        # (3) LOWER BOUND — non-detection months treated as Q = 0.
        summary["annualised_t_per_yr_lower_bound"] = float(
            mean_q * 8760 / 1000 * (n_det_mo / n_obs_mo)
        )

        summary["n_detection_months"]     = n_det_mo
        summary["n_non_detection_months"] = n_ndet_mo
        summary["n_observed_months"]      = n_obs_mo

    # ── Climate TRACE annual totals for the run year(s) ──────────────────────
    ct_annual = {}
    for yr in args.years:
        yr_total = sum(v for (y, m), v in ct_ch4.items() if y == yr)
        if yr_total > 0:
            ct_annual[yr] = yr_total
    summary["ct_ch4_annual_t_by_year"] = ct_annual

    # Monthly CT breakdown for every processed record (already tagged above)
    summary["ct_ch4_monthly_reference"] = {
        f"{y}-{m:02d}": round(v, 2)
        for (y, m), v in sorted(ct_ch4.items())
        if y in args.years
    }

    store["summary"] = summary
    with open(OUT_JSON, "w") as f:
        json.dump(store, f, indent=2)

    # Console summary
    print("\n" + "=" * 72)
    years_str = "–".join(str(y) for y in (min(args.years), max(args.years))) \
                if len(args.years) > 1 else str(args.years[0])
    print(f"Bełchatów {years_str} time series  (quant crop: KWB mine polygon ~21km × 4.2km)")
    print("=" * 72)
    print(f"Acquisitions processed:      {summary['acquisitions_processed']}")
    print(f"Uniform-field excluded:      {summary['uniform_field_excluded']}  "
          f"(site_mean≈ctrl_mean — spectrally homogeneous scene)")
    print(f"Valid acquisitions:          {summary['valid_acquisitions']}")
    print(f"Observed months:             {summary['n_observed_months']}")
    print(f"Detection acquisitions:      {summary['detection_acquisitions']}  (S/C > 1.15)")
    print(f"Detection months:            {summary['detection_months']}")
    print(f"Detections above conformal:  {summary['detections_above_tau']}  "
          f"(τ={CONFORMAL_TAU})")
    print(f"Detections CFAR-confirmed:   {summary['detections_cfar_pass']}")
    if detected and flows:
        print(f"Det months / non-det / obs:  "
              f"{summary['n_detection_months']} / {summary['n_non_detection_months']} / {summary['n_observed_months']}")
        print(f"Detection rate (by month):   {summary['detection_rate_by_month']*100:.1f}%")
        print(f"Mean flow rate (detections): {summary['mean_flow_kg_h']:.0f} kg/h")
        print(f"Range:                       {summary['min_flow_kg_h']:.0f}–{summary['max_flow_kg_h']:.0f} kg/h")
        print(f"Annualisation — three framings (non-detections are missing obs, not zero):")
        print(f"  Upper (det-mean × 8760):           {summary['annualised_t_per_yr_upper']:.0f} t/yr")
        print(f"  Floor-imputed (Q_floor={summary['detection_floor_kgh_assumed']:.0f} kg/h):     {summary['annualised_t_per_yr_floor_imputed']:.0f} t/yr")
        print(f"  Lower bound (non-det = 0):         {summary['annualised_t_per_yr_lower_bound']:.0f} t/yr")
    if ct_annual:
        for yr, total in sorted(ct_annual.items()):
            print(f"Climate TRACE {yr} annual CH4:      {total:,.1f} t/yr  (asset 16168, sum of monthly)")
    print(f"Output: {OUT_JSON}")
    print("=" * 72)


if __name__ == "__main__":
    main()
