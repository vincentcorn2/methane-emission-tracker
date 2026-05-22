"""
scripts/timeseries/rybnik_chwalowice_annual_timeseries.py
==========================================================
Annual CH4Net + CEMF+IME time series for KWK ROW Ruch Chwałowice
(Polska Grupa Górnicza S.A.), Rybnik, Upper Silesia, Poland.

Site confirmed by Carbon Mapper (Tanager + EMIT):
  6 detections at 50.0781°N, 18.5451°E, ranging 1,150–2,019 kg CH₄/hr.

For each target month in the configured year:
  1. Search CDSE for cloud-free T34UCA Sentinel-2 L1C acquisitions
  2. Pick the lowest-cloud product not already cached
  3. Download + convert to .npy if needed
  4. Run CH4Net v8 inference, compute S/C against the site crop
  5. If S/C > 1.15 (skip_bitemporal=True — same as rybnik in backfill pipeline):
     extract B11/B12, build a SiteCfg, run CEMF+IME with ERA5 wind
  6. Log every record to results_analysis/rybnik_chwalowice_annual_timeseries.json

Results are tagged with any Carbon Mapper detections in the same calendar month
for direct comparison.

Defaults to 2019–2025. Evaluates all cloud-free acquisitions per month.
CM detections span 2023–2026; pre-2023 years provide baseline context.

Usage:
    conda activate methane
    python scripts/timeseries/rybnik_chwalowice_annual_timeseries.py
    python scripts/timeseries/rybnik_chwalowice_annual_timeseries.py --years 2023 2024 2025
    python scripts/timeseries/rybnik_chwalowice_annual_timeseries.py --dry-run
"""

import argparse
import csv
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

# ── Path setup: must be before any local imports ──────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]   # methane-api/
sys.path.insert(0, str(_PROJECT_ROOT))                          # for src.*
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
        logging.FileHandler("results_analysis/rybnik_chwalowice_annual_timeseries.log"),
    ],
)
log = logging.getLogger("rybnik_chwalowice_timeseries")


# ── Site config ───────────────────────────────────────────────────────────────
SITE_NAME = "rybnik_chwalowice"
LAT, LON  = 50.0781, 18.5451   # Carbon Mapper confirmed source pin
                                 # KWK ROW Ruch Chwałowice (d. "Donnersmarck Grube")
                                 # PGG S.A., Rybnik, Silesian Voivodeship, Poland
TILE_ID   = "T34UCA"
DETECTION_THRESHOLD = 1.15   # classic S/C threshold (skip_bitemporal=True — same
                              # as all Rybnik entries in apply_bitemporal_diff.py)
CONFORMAL_TAU       = 3.5796 # global conformal τ at α=0.10, n_cal=35
                              # (calibrated_threshold.json → global_thresholds.tau_alpha_10.tau)

# ── S/C control crop offsets ──────────────────────────────────────────────────
# Underground mine — symmetric offsets. 0.20° (~22 km) clears the ~7.5 km E-W
# concession boundary on all sides with headroom.
SC_OFFSET_N = 0.20
SC_OFFSET_S = 0.20
SC_OFFSET_E = 0.20
SC_OFFSET_W = 0.20

# ── KWK Chwałowice mine polygon — quantification boundary ────────────────────
# Approximate concession area boundary of KWK ROW Ruch Chwałowice based on
# OSM industrial polygons and the Carbon Mapper plume extents.
# ~7.5 km E-W × ~3.6 km N-S. Pixels outside this polygon are masked before CEMF+IME.
MINE_POLYGON_LATLON = [
    (50.092, 18.508),   # NW
    (50.092, 18.580),   # NE
    (50.060, 18.580),   # SE
    (50.060, 18.508),   # SW
]

DOWNLOAD_DIR = Path("data/downloads/rybnik_chwalowice_annual")
OUT_JSON     = Path("results_analysis/rybnik_chwalowice_annual_timeseries.json")
ACQ_DATE_RE  = re.compile(r"_(\d{8})T")

# Carbon Mapper detections CSV (Tanager + EMIT, all confirmed at this pin)
CM_CSV = Path("data/rybnik_chwalowice_carbon_mapper.csv")

# TROPOMI positive events JSON (from extend_tropomi_rybnik.py)
TROPOMI_JSON = Path("results_analysis/tropomi_positives.json")
TROPOMI_SITE = "silesia_rybnik"   # site key used in tropomi_positives.json


def load_tropomi_events() -> dict:
    """Return {(year, month): [event_dict, ...]} from tropomi_positives.json.

    Only includes events for TROPOMI_SITE.  Each event dict has:
    date, enhancement_ppb, n_near_pixels, validated, product_name.
    Months with no events are absent from the returned dict.
    """
    by_month: dict = {}
    if not TROPOMI_JSON.exists():
        log.warning("TROPOMI JSON not found at %s — TROPOMI tagging disabled",
                    TROPOMI_JSON)
        return by_month

    with open(TROPOMI_JSON) as f:
        events = json.load(f)

    site_events = [e for e in events if e.get("site") == TROPOMI_SITE]
    for e in site_events:
        date_str = e.get("date", "")
        if not date_str:
            continue
        try:
            year, month, _ = date_str.split("-")
            key = (int(year), int(month))
        except ValueError:
            continue
        by_month.setdefault(key, []).append({
            "date":            date_str,
            "enhancement_ppb": e.get("enhancement_ppb"),
            "n_near_pixels":   e.get("n_near_pixels"),
            "validated":       e.get("validated", False),
            "product_name":    e.get("product_name", ""),
        })

    total = sum(len(v) for v in by_month.values())
    log.info("Loaded %d TROPOMI events across %d months (site: %s)",
             total, len(by_month), TROPOMI_SITE)
    return by_month


def load_cm_detections() -> dict:
    """Return {(year, month): [detection_dict, ...]} from the Carbon Mapper CSV.

    Each detection dict has: date, instrument, emission_kgh, wind_dir_deg,
    wind_speed_ms, plume_id.  Months with no detections are absent from the dict.
    """
    by_month: dict = {}
    if not CM_CSV.exists():
        log.warning("Carbon Mapper CSV not found at %s — CM comparison disabled", CM_CSV)
        return by_month

    with open(CM_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            # datetime field: "2025-03-21T10:21:54+00"
            dt_str = row.get("datetime", "").strip()
            if not dt_str:
                continue
            try:
                dt = datetime.fromisoformat(dt_str.replace("+00", "+00:00"))
            except ValueError:
                log.warning("Could not parse CM datetime: %s", dt_str)
                continue

            key = (dt.year, dt.month)

            # emission_auto is blank for some Tanager rows (v3 without auto-emission)
            em_str = row.get("emission_auto", "").strip()
            emission_kgh = float(em_str) if em_str else None

            ws_str = row.get("wind_speed_avg_auto", "").strip()
            wd_str = row.get("wind_direction_avg_auto", "").strip()

            det = {
                "plume_id":      row.get("plume_id", "").strip(),
                "date":          dt.strftime("%Y-%m-%d"),
                "time_utc":      dt.strftime("%H:%M"),
                "instrument":    row.get("instrument", "").strip(),
                "platform":      row.get("platform", "").strip(),
                "mission_phase": row.get("mission_phase", "").strip(),
                "emission_kgh":  emission_kgh,
                "wind_speed_ms": float(ws_str) if ws_str else None,
                "wind_dir_deg":  float(wd_str) if wd_str else None,
            }
            by_month.setdefault(key, []).append(det)

    total_dets = sum(len(v) for v in by_month.values())
    log.info("Loaded %d Carbon Mapper detections across %d months",
             total_dets, len(by_month))
    return by_month


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
        with zipfile.ZipFile(zip_path, "r") as z:
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

    # Guard against corrupt ZIPs from interrupted previous runs
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


def acquire_all_month(year, month, client, max_cloud, dry_run, max_candidates=0):
    """Return list of {"npy_path", "search"} dicts — one per cloud-free L1C
    acquisition this month.  All candidates are downloaded (or pulled from cache),
    not just the lowest-cloud one.  This gives more observations per month,
    which matters especially for episodic emitters like underground coal mines.

    If max_candidates > 0, stop downloading after that many successful results.

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

        # Stop early once we have enough successful acquisitions
        if max_candidates > 0 and len(results) >= max_candidates:
            log.info("  max_candidates=%d reached, stopping early.", max_candidates)
            break

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
    # scene — snow, uniform cloud-free winter landscape).  The S/C ratio is
    # floating-point noise and must not be treated as a real detection.
    # These scenes pass the B11 non-zero swath check, so they need a separate flag.
    sm = sc.get("site_mean") or 0.0
    cm = sc.get("ctrl_mean") or 0.0
    uniform_field = bool(sm > 0 and abs(sm - cm) < 1e-4)

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
        "uniform_field":     uniform_field,
    }


def quantify(npy_path, sc_record, era5_client):
    """If S/C clears threshold, run CEMF+IME with ERA5 winds."""
    sc = sc_record.get("sc_ratio")
    if sc is None or sc <= DETECTION_THRESHOLD:
        return {"status": "below_threshold", "sc_ratio": sc}

    log.info("  S/C = %.2f >= %.2f -> running quantification", sc, DETECTION_THRESHOLD)

    import rasterio
    from rasterio.features import geometry_mask
    from shapely.geometry import Polygon as ShapelyPolygon

    tif_path = Path(sc_record["tif"])
    with rasterio.open(tif_path) as src:
        prob_full = src.read(1).astype(np.float32)
        H, W = prob_full.shape
        transform = src.transform

        # Convert mine polygon lat/lon → pixel row/col
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

        # Build polygon in UTM (same CRS as the raster).
        # geometry_mask expects coords in the raster CRS — passing pixel-space
        # integers would give a completely wrong mask.
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

    # Combined mask: CH4Net threshold AND inside mine polygon
    mask_original = ((prob >= 0.18) & mine_mask).astype(np.float32)
    log.info("  Polygon mask: %d / %d px active (%.1f%% of bounding box)",
             int(mine_mask.sum()), mine_mask.size,
             100 * mine_mask.sum() / mine_mask.size)

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
                        help="Years to process (ignored if --target-months is set)")
    parser.add_argument("--months", nargs="+", type=int,
                        default=list(range(1, 13)),
                        help="Months to process per year (ignored if --target-months is set)")
    parser.add_argument("--target-months", nargs="+", type=str,
                        default=None,
                        metavar="YYYY-MM",
                        help="Explicit list of YYYY-MM months to process. Overrides "
                             "--years/--months. E.g. --target-months 2023-01 2023-08 2025-03")
    parser.add_argument("--max-cloud", type=float, default=20.0)
    parser.add_argument("--max-cloud-fallback", type=float, default=40.0,
                        help="Retry cloud ceiling if first search returns no products")
    parser.add_argument("--max-acq", type=int, default=1,
                        help="Max acquisitions to evaluate per month (default: 1 = lowest-cloud only). "
                             "Use --max-acq 0 for all.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Search catalog only, no downloads or inference")
    args = parser.parse_args()

    # Build (year, month) pairs to process
    if args.target_months:
        try:
            ym_pairs = [(int(s[:4]), int(s[5:7])) for s in args.target_months]
        except (ValueError, IndexError):
            parser.error("--target-months must be YYYY-MM strings, e.g. 2023-01")
        args.years = sorted({y for y, m in ym_pairs})
        log.info("Target months: %s", ", ".join(args.target_months))
    else:
        ym_pairs = [(y, m) for y in args.years for m in args.months]

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    cm_detections   = load_cm_detections()    # {(year, month): [det, ...]}
    tropomi_events  = load_tropomi_events()   # {(year, month): [event, ...]}

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

    # Credentials and clients
    user, pw = get_credentials()
    cdse = CopernicusClient(username=user, password=pw)
    era5 = ERA5Client()

    if not args.dry_run:
        log.info("Loading CH4Net v8 weights ...")
        detector = CH4NetDetector(WEIGHTS)
    else:
        detector = None

    # ── Processed-acquisition lookup ──────────────────────────────────────────
    # Records are now per-acquisition (not per-month).  Track by product_name so
    # reruns skip individual acquisitions we've already handled, while still
    # retrying download_failed acquisitions (those have a product_name but no
    # "detection" key — we exclude them from `done_products`).
    done_products = {
        r.get("search", {}).get("product_name")
        for r in records
        if "detection" in r and r.get("search", {}).get("product_name")
    }
    # Months confirmed to have no available products — skip re-searching these.
    done_no_products = {
        r.get("month")
        for r in records
        if r.get("search", {}).get("status") == "no_products"
    }

    def _save():
        with open(OUT_JSON, "w") as f:
            json.dump(store, f, indent=2)

    for year, month in ym_pairs:
        month_key = f"{year}-{month:02d}"

        if month_key in done_no_products:
            log.info("=" * 65)
            log.info("── %s : no products (confirmed), skipping", month_key)
            continue

        log.info("=" * 65)
        acquisitions = acquire_all_month(
            year, month, cdse, args.max_cloud, args.dry_run,
            max_candidates=args.max_acq,
        )
        # If nothing found at strict cloud threshold, retry looser
        if (len(acquisitions) == 1
                and acquisitions[0]["search"].get("status") == "no_products"
                and args.max_cloud_fallback > args.max_cloud):
            log.info("  Retrying with cloud <= %.0f%% ...", args.max_cloud_fallback)
            acquisitions = acquire_all_month(
                year, month, cdse, args.max_cloud_fallback, args.dry_run,
                max_candidates=args.max_acq,
            )

        # Handle month-level failures (no products / search error)
        first_status = acquisitions[0]["search"].get("status")
        if first_status in ("no_products", "search_failed"):
            if not any(r.get("month") == month_key
                       and r.get("search", {}).get("status") == first_status
                       for r in records):
                rec = {
                    "month":           month_key,
                    "search":          acquisitions[0]["search"],
                    "cm_detections":   cm_detections.get((year, month), []),
                    "tropomi_events":  tropomi_events.get((year, month), []),
                }
                records.append(rec)
                _save()
            continue

        # Process each individual acquisition
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
                "cm_detections":    cm_detections.get((year, month), []),
                "tropomi_events":   tropomi_events.get((year, month), []),
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
                log.info("  %s  ← UNIFORM FIELD (site_mean≈ctrl_mean=%.5f) — excluded",
                         month_key, sc_rec.get("site_mean", 0))

            if sc_rec.get("sc_ratio") is not None and not uf:
                quant_rec = quantify(npy_path, sc_rec, era5)
                rec["quantification"] = quant_rec
                cm_flag = "★CM" if cm_detections.get((year, month)) else ""
                trop_flag = "★TROP" if tropomi_events.get((year, month)) else ""
                log.info(
                    "  %s [%s]  S/C=%.2f  CFAR=%s  Q=%s kg/h  %s%s",
                    month_key, acq_date,
                    sc_rec["sc_ratio"],
                    "DETECT" if sc_rec.get("cfar_detect") else "no",
                    f"{quant_rec.get('flow_rate_kgh'):.0f}"
                    if quant_rec.get("flow_rate_kgh") else "—",
                    cm_flag, trop_flag,
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
    # A "valid observation month" is a calendar month that had at least one
    # acquisition with a real (non-uniform-field, non-partial-swath) detection
    # result — regardless of whether it detected CH4.  Records with only
    # uniform_field or download_failed acquisitions don't count as observed.
    valid_acq = [
        r for r in records
        if "detection" in r
        and not r.get("detection", {}).get("uniform_field")
    ]
    # Unique observed months
    obs_months = sorted({r["month"] for r in valid_acq})
    # Months with any detection
    det_months = sorted({r["month"] for r in detected})

    # Months where Carbon Mapper also detected (independent confirmation)
    cm_confirmed_months = {
        f"{y}-{m:02d}"
        for (y, m) in cm_detections
        if y in args.years
    }
    co_detected_months = sorted(set(det_months) & cm_confirmed_months)

    summary = {
        "years":                      args.years,
        "site":                       SITE_NAME,
        "lat":                        LAT,
        "lon":                        LON,
        "tile":                       TILE_ID,
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
        "quantification_boundary":    "KWK Chwałowice polygon (~7.5km × 3.6km)",
        "cm_co_detected_months":      co_detected_months,
        "n_cm_co_detected_months":    len(co_detected_months),
    }

    if detected:
        flows = [r["quantification"]["flow_rate_kgh"] for r in detected
                 if r["quantification"].get("flow_rate_kgh") is not None]
        if flows:
            summary["mean_flow_kg_h"]   = float(np.mean(flows))
            summary["median_flow_kg_h"] = float(np.median(flows))
            summary["min_flow_kg_h"]    = float(np.min(flows))
            summary["max_flow_kg_h"]    = float(np.max(flows))

            n_det_mo  = len(det_months)
            n_obs_mo  = len(obs_months)
            n_ndet_mo = n_obs_mo - n_det_mo

            # Three annualisation framings — month-level (same framing as Bełchatów).
            # Non-detection months are missing observations, not zero-emission months.
            DETECTION_FLOOR_KGH = 300.0
            mean_q = float(np.mean(flows))
            summary["annualised_t_per_yr_upper"] = float(mean_q * 8760 / 1000)
            summary["annualised_t_per_yr_floor_imputed"] = float(
                (mean_q * (n_det_mo / n_obs_mo) +
                 DETECTION_FLOOR_KGH * (n_ndet_mo / n_obs_mo)) * 8760 / 1000
            )
            summary["annualised_t_per_yr_lower_bound"] = float(
                mean_q * 8760 / 1000 * (n_det_mo / n_obs_mo)
            )
            summary["detection_floor_kgh_assumed"] = DETECTION_FLOOR_KGH
            summary["n_detection_months"]     = n_det_mo
            summary["n_non_detection_months"] = n_ndet_mo
            summary["n_observed_months"]      = n_obs_mo

    # Carbon Mapper reference — all detections in the run years
    cm_ref = {}
    for (y, m), dets in sorted(cm_detections.items()):
        if y in args.years:
            key = f"{y}-{m:02d}"
            cm_ref[key] = [{k: v for k, v in d.items()} for d in dets]
    summary["cm_detections_by_month"] = cm_ref

    # TROPOMI reference — all events in the run years
    tropomi_ref = {}
    for (y, m), evts in sorted(tropomi_events.items()):
        if y in args.years:
            key = f"{y}-{m:02d}"
            tropomi_ref[key] = [{k: v for k, v in e.items()} for e in evts]
    tropomi_validated = {k: [e for e in v if e.get("validated")]
                         for k, v in tropomi_ref.items()}
    tropomi_validated = {k: v for k, v in tropomi_validated.items() if v}
    summary["tropomi_events_by_month"]           = tropomi_ref
    summary["tropomi_validated_months"]          = sorted(tropomi_validated.keys())
    summary["n_tropomi_validated_months"]        = len(tropomi_validated)
    summary["tropomi_mean_enhancement_ppb"]      = (
        float(np.mean([e["enhancement_ppb"]
                       for evts in tropomi_validated.values()
                       for e in evts
                       if e.get("enhancement_ppb") is not None]))
        if tropomi_validated else None
    )

    store["summary"] = summary
    with open(OUT_JSON, "w") as f:
        json.dump(store, f, indent=2)

    # Console summary
    print("\n" + "=" * 72)
    years_str = "–".join(str(y) for y in (min(args.years), max(args.years))) \
                if len(args.years) > 1 else str(args.years[0])
    print(f"KWK Chwałowice (rybnik_chwalowice) {years_str}  "
          f"(quant crop: {summary['quantification_boundary']})")
    print("=" * 72)
    print(f"Acquisitions processed:      {summary['acquisitions_processed']}")
    print(f"Uniform-field excluded:      {summary['uniform_field_excluded']}  "
          f"(site_mean≈ctrl_mean — spectrally homogeneous scene)")
    print(f"Valid acquisitions:          {summary['valid_acquisitions']}")
    print(f"Observed months:             {summary['n_observed_months']}")
    print(f"Detection acquisitions:      {summary['detection_acquisitions']}  "
          f"(S/C > 1.15, not uniform-field)")
    print(f"Detection months:            {summary['detection_months']}")
    print(f"Detections above conformal:  {summary['detections_above_tau']}  "
          f"(τ={CONFORMAL_TAU}, α=0.10, n=35)")
    print(f"Detections CFAR-confirmed:   {summary['detections_cfar_pass']}")
    print(f"CM co-detected months:       {summary['n_cm_co_detected_months']}  "
          f"{summary['cm_co_detected_months']}")
    if detected and flows:
        print(f"Det months / non-det / obs:  "
              f"{summary['n_detection_months']} / {summary['n_non_detection_months']} / {summary['n_observed_months']}")
        print(f"Detection rate (by month):   {summary['detection_rate_by_month']*100:.1f}%")
        print(f"Mean flow rate (detections): {summary['mean_flow_kg_h']:.0f} kg/h")
        print(f"CM range (Tanager+EMIT):     1,150–2,019 kg/h  "
              f"(6 detections 2023-08 / 2025-03 / 2025-08 / 2026-03)")
        print(f"Annualisation — three framings (month-level):")
        print(f"  Upper (det-mean × 8760):           {summary['annualised_t_per_yr_upper']:.0f} t/yr")
        print(f"  Floor-imputed (Q_floor={summary['detection_floor_kgh_assumed']:.0f} kg/h): "
              f"    {summary['annualised_t_per_yr_floor_imputed']:.0f} t/yr")
        print(f"  Lower bound (non-det = 0):         {summary['annualised_t_per_yr_lower_bound']:.0f} t/yr")
    if cm_ref:
        print(f"\nCarbon Mapper detections in run years:")
        for mkey, dets in sorted(cm_ref.items()):
            for d in dets:
                em = f"{d['emission_kgh']:.0f} kg/h" if d["emission_kgh"] else "(no auto-emission)"
                print(f"  {d['date']}  {d['instrument'].upper():5s}  {em}")

    if tropomi_validated:
        print(f"\nTROPOMI validated events ({summary['n_tropomi_validated_months']} months, "
              f"mean enhancement {summary['tropomi_mean_enhancement_ppb']:.1f} ppb):")
        for mkey, evts in sorted(tropomi_validated.items()):
            for e in evts:
                print(f"  {e['date']}  +{e['enhancement_ppb']:.1f} ppb  "
                      f"({e['n_near_pixels']} px)")
    print(f"\nOutput: {OUT_JSON}")
    print("=" * 72)


if __name__ == "__main__":
    main()
