"""
scripts/quantification/requant_mbsp_upgrade.py
===============================================
Re-run CEMF+IME quantification on ALL crop variants (old wrong-site and
new correct-site) for Bełchatów and Rybnik, using the upgraded MBSP
cemf.py (scene-derived c coefficient, Varon Eq. 3).

Processes:
  OLD CROPS (wrong / intermediate site):
    01_belchatow_powerstation_coords_750px_crop_2019-2024.json   (power station, 7.5km box)
    02_belchatow_powerstation_5km_crop_2024.json                 (power station, 5km box)
    03_belchatow_mine_coords_singleacq_2021-2024.json            (mine centroid, single acq)

  NEW CROPS (correct site):
    belchatow_annual_timeseries.json                             (mine polygon, all acq)
    rybnik_chwalowice_annual_timeseries.json                     (CM pin, all acq)

Each old crop is re-quantified using its original spatial extent so the
physics upgrade is isolated from any crop-location change.  This gives
a true apples-to-apples comparison: same MBSP physics, same crop region,
old Q vs new Q.

No new downloads.  No CH4Net re-inference.

Run:
    conda activate methane
    python scripts/quantification/requant_mbsp_upgrade.py
"""

import json
import logging
import re
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import Polygon as ShapelyPolygon

# ── paths ─────────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.quantification.cemf import run_cemf, downsample_mask
from src.quantification.ime import CEMFIntegratedMassEnhancement

NPY_CACHE  = REPO / "data" / "npy_cache"
RESULTS    = REPO / "results_analysis"
RESULTS_BT = REPO / "results_bitemporal"   # root for tif paths stored in JSON

# ── band indices in the .npy cube ─────────────────────────────────────────────
B11_IDX = 10   # ~1610 nm SWIR reference
B12_IDX = 11   # ~2190 nm CH4 absorption

# ── CH4Net probability threshold used to define the mask ─────────────────────
PROB_THRESHOLD = 0.18

# ── acquisition-date regex (matches product name stem) ───────────────────────
ACQ_DATE_RE = re.compile(r"_(\d{8})T\d{6}_")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── crop polygon helpers ──────────────────────────────────────────────────────
def square_polygon(lat, lon, half_m):
    """
    Return a closed 5-point lat/lon polygon representing a square box of
    ±half_m metres centred on (lat, lon).  Used to replicate fixed-pixel
    crop windows from the old timeseries scripts.
    """
    # 1° latitude ≈ 111_320 m everywhere
    d_lat = half_m / 111_320
    # 1° longitude ≈ 111_320 × cos(lat) m
    import math
    d_lon = half_m / (111_320 * math.cos(math.radians(lat)))
    return [
        (lat + d_lat, lon - d_lon),   # NW
        (lat + d_lat, lon + d_lon),   # NE
        (lat - d_lat, lon + d_lon),   # SE
        (lat - d_lat, lon - d_lon),   # SW
        (lat + d_lat, lon - d_lon),   # NW close
    ]


# Power station centre used in the old wrong-site scripts
_PS_LAT, _PS_LON = 51.266, 19.315
# Mine centroid (Climate TRACE, asset 16168) — used in mine-coords scripts
_MINE_LAT, _MINE_LON = 51.242, 19.275

# Mine polygon used in the current (correct) timeseries scripts
_MINE_POLYGON = [
    (51.2750, 19.1510),
    (51.2750, 19.3300),
    (51.1950, 19.3300),
    (51.1950, 19.1510),
    (51.2750, 19.1510),
]

# ── site configs ──────────────────────────────────────────────────────────────
TS_DIR = RESULTS / "timeseries"

SITES = {
    # ── OLD CROPS (wrong / intermediate site) ────────────────────────────────
    "belchatow_01_powerstation_750px": {
        "label":    "Bełchatów — power-station coords, 750 px crop (7.5 km)",
        "json_in":  TS_DIR / "belchatow" / "01_belchatow_powerstation_coords_750px_crop_2019-2024.json",
        "json_out": TS_DIR / "belchatow" / "01_belchatow_powerstation_coords_750px_crop_2019-2024_mbsp.json",
        # 750 px × 10 m / 2 = 3 750 m half-width
        "polygon":  square_polygon(_PS_LAT, _PS_LON, half_m=3_750),
    },
    "belchatow_02_powerstation_5km": {
        "label":    "Bełchatów — power-station coords, 5 km crop",
        "json_in":  TS_DIR / "belchatow" / "02_belchatow_powerstation_5km_crop_2024.json",
        "json_out": TS_DIR / "belchatow" / "02_belchatow_powerstation_5km_crop_2024_mbsp.json",
        # 500 px × 10 m / 2 = 2 500 m half-width
        "polygon":  square_polygon(_PS_LAT, _PS_LON, half_m=2_500),
    },
    "belchatow_03_mine_singleacq": {
        "label":    "Bełchatów — mine centroid, single acq per month",
        "json_in":  TS_DIR / "belchatow" / "03_belchatow_mine_coords_singleacq_2021-2024.json",
        "json_out": TS_DIR / "belchatow" / "03_belchatow_mine_coords_singleacq_2021-2024_mbsp.json",
        # Original script used 750 px crop around the mine centroid
        "polygon":  square_polygon(_MINE_LAT, _MINE_LON, half_m=3_750),
    },
    # ── NEW CROPS (correct site) ─────────────────────────────────────────────
    "belchatow_04_mine_polygon": {
        "label":    "Bełchatów — mine polygon, all acq (new crop) ★",
        "json_in":  RESULTS / "belchatow_annual_timeseries.json",
        "json_out": RESULTS / "belchatow_annual_timeseries_mbsp.json",
        "polygon":  _MINE_POLYGON,
    },
    "rybnik_chwalowice": {
        "label":    "Rybnik/Chwałowice — CM pin, all acq (new crop) ★",
        "json_in":  RESULTS / "rybnik_chwalowice_annual_timeseries.json",
        "json_out": RESULTS / "rybnik_chwalowice_annual_timeseries_mbsp.json",
        "tif_subdir": "rybnik_chwalowice",
        "polygon": [
            (50.1000, 18.5000),
            (50.1000, 18.5900),
            (50.0550, 18.5900),
            (50.0550, 18.5000),
            (50.1000, 18.5000),
        ],
    },
}


def lonlat_to_pixel(tif_path: Path, lon: float, lat: float):
    """Return (row, col) for a given lon/lat in the raster's CRS."""
    import pyproj
    with rasterio.open(tif_path) as src:
        crs = src.crs
        transform = src.transform

    proj = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = proj.transform(lon, lat)
    col = (x - transform.c) / transform.a
    row = (y - transform.f) / transform.e
    return int(row), int(col)


def build_polygon_mask(tif_path: Path, polygon_latlon: list):
    """
    Build a boolean mask (True = inside mine polygon) at the crop bounding
    box of the polygon within the given tif.

    Returns (r0, r1, c0, c1, mine_mask) in pixel coordinates of the full tif.
    """
    poly_rows, poly_cols = [], []
    for plat, plon in polygon_latlon:
        pr, pc = lonlat_to_pixel(tif_path, plon, plat)
        poly_rows.append(pr)
        poly_cols.append(pc)

    with rasterio.open(tif_path) as src:
        H, W = src.shape
        transform = src.transform

    r0 = max(0, min(poly_rows))
    r1 = min(H, max(poly_rows))
    c0 = max(0, min(poly_cols))
    c1 = min(W, max(poly_cols))

    with rasterio.open(tif_path) as src:
        t = src.transform

    poly_xy = [
        (t.c + c * t.a, t.f + r * t.e)
        for r, c in zip(poly_rows, poly_cols)
    ]
    poly_crs = ShapelyPolygon(poly_xy)

    window_transform = rasterio.transform.from_bounds(
        t.c + c0 * t.a,
        t.f + r1 * t.e,
        t.c + c1 * t.a,
        t.f + r0 * t.e,
        c1 - c0, r1 - r0,
    )
    mine_mask = ~geometry_mask(
        [poly_crs.__geo_interface__],
        out_shape=(r1 - r0, c1 - c0),
        transform=window_transform,
        invert=False,
    )
    return r0, r1, c0, c1, mine_mask


def requant_record(record: dict, polygon: list, tif_root: Path) -> dict:
    """
    Re-run CEMF+IME on a single quantified record.

    Returns updated quantification dict, or None if inputs are missing.
    """
    det = record.get("detection", {})
    quant_old = record.get("quantification", {})

    tif_rel = det.get("tif")
    npy_name = record.get("npy")
    if not tif_rel or not npy_name:
        return None

    tif_path = tif_root / tif_rel
    npy_path = NPY_CACHE / npy_name

    if not tif_path.exists():
        log.warning("  Missing tif: %s", tif_path)
        return None
    if not npy_path.exists():
        log.warning("  Missing npy: %s", npy_path)
        return None

    # ── load CH4Net probability crop ─────────────────────────────────────────
    try:
        r0, r1, c0, c1, mine_mask = build_polygon_mask(tif_path, polygon)
    except Exception as e:
        log.warning("  Polygon mask failed: %s", e)
        return None

    with rasterio.open(tif_path) as src:
        prob_full = src.read(1).astype(np.float32)

    prob = prob_full[r0:r1, c0:c1]
    del prob_full

    # ── load B11 / B12 from npy ───────────────────────────────────────────────
    arr = np.load(npy_path)
    b11_full = arr[r0:r1, c0:c1, B11_IDX].astype(np.float32) / 255.0
    b12_full = arr[r0:r1, c0:c1, B12_IDX].astype(np.float32) / 255.0
    del arr

    # ── plume mask: CH4Net threshold AND inside mine polygon ──────────────────
    mask_10m = ((prob >= PROB_THRESHOLD) & mine_mask).astype(np.float32)

    # Downsample 10m → 20m for SWIR band alignment
    mask_20m = downsample_mask(mask_10m)
    b11 = b11_full[::2, ::2]   # downsample to match 20m SWIR resolution
    b12 = b12_full[::2, ::2]
    del b11_full, b12_full

    # ── acquisition timestamp ─────────────────────────────────────────────────
    m = ACQ_DATE_RE.search(npy_name)
    acq = m.group(1) if m else "00000000"
    acq_iso = f"{acq[:4]}-{acq[4:6]}-{acq[6:8]}T10:00:00Z"

    # ── CEMF with upgraded c ──────────────────────────────────────────────────
    cemf_result = run_cemf(
        b11=b11,
        b12=b12,
        mask=mask_20m,
        scene_id=Path(npy_name).stem,
        timestamp=acq_iso,
    )

    if not cemf_result.retrieval_valid:
        log.warning("  CEMF invalid: %s", cemf_result.warning)
        return None

    # ── IME → flow rate, reusing stored ERA5 wind ─────────────────────────────
    wind_ms  = quant_old.get("wind_speed_ms", 3.5)
    wind_src = quant_old.get("wind_source", "ERA5_reanalysis")

    ime = CEMFIntegratedMassEnhancement(pixel_size_m=20.0)
    result = ime.estimate_from_cemf(
        cemf_result=cemf_result,
        wind_speed_ms=wind_ms,
        wind_source=wind_src,
    )

    return {
        "status":                    "quantified",
        "method":                    "CEMF+IME_MBSP_v2",
        "flow_rate_kgh":             result.flow_rate_kgh,
        "flow_rate_lower_kgh":       result.flow_rate_lower_kgh,
        "flow_rate_upper_kgh":       result.flow_rate_upper_kgh,
        "sc_ratio":                  quant_old.get("sc_ratio"),
        "n_plume_pixels":            cemf_result.n_plume_pixels,
        "total_mass_kg":             round(cemf_result.total_mass_kg, 4),
        "wind_speed_ms":             wind_ms,
        "wind_dir_deg":              quant_old.get("wind_dir_deg"),
        "wind_source":               wind_src,
        "uncertainty_pct":           quant_old.get("uncertainty_pct", 30),
        "governance_flags":          quant_old.get("governance_flags", []),
        "annual_tonnes_if_continuous": result.annual_tonnes,
        # provenance: keep old result alongside for comparison
        "flow_rate_kgh_old":         quant_old.get("flow_rate_kgh"),
    }


def process_site(site_name: str, cfg: dict):
    log.info("=" * 65)
    log.info("%s", cfg.get("label", site_name))
    log.info("=" * 65)

    with open(cfg["json_in"]) as f:
        raw = json.load(f)

    # Support both list-format (old JSONs) and dict-with-records (new JSONs)
    if isinstance(raw, list):
        records = raw
        store = {"records": records}
    else:
        store = raw
        records = store["records"]

    polygon = cfg["polygon"]
    tif_root = REPO

    n_updated = 0
    n_skipped = 0

    print(f"\n{'Month':<10} {'Acq Date':<13} {'Old Q (kg/h)':>13} {'New Q (kg/h)':>13} {'Δ%':>7}")
    print("-" * 60)

    for rec in records:
        quant = rec.get("quantification", {})
        if quant.get("status") != "quantified":
            continue

        month    = rec.get("month", "????-??")
        acq_date = rec.get("acquisition_date", rec.get("search", {}).get("acquisition_date", "")[:10])
        old_q    = quant.get("flow_rate_kgh", 0.0)

        log.info("Processing %s [%s]  old Q = %.0f kg/h", month, acq_date, old_q)

        new_quant = requant_record(rec, polygon, tif_root)
        if new_quant is None:
            log.warning("  Skipped (missing files or bad scene)")
            n_skipped += 1
            continue

        new_q = new_quant["flow_rate_kgh"]
        delta_pct = (new_q - old_q) / old_q * 100 if old_q else float("nan")
        print(f"{month:<10} {acq_date:<13} {old_q:>13.0f} {new_q:>13.0f} {delta_pct:>+7.1f}%")

        rec["quantification"] = new_quant
        n_updated += 1

    print()
    log.info("Updated %d records, skipped %d", n_updated, n_skipped)

    out_path = cfg["json_out"]
    with open(out_path, "w") as f:
        json.dump(store, f, indent=2)
    log.info("Saved → %s", out_path)

    old_qs = [r["quantification"]["flow_rate_kgh_old"] for r in records
              if r.get("quantification", {}).get("status") == "quantified"
              and r["quantification"].get("flow_rate_kgh_old") is not None]
    new_qs = [r["quantification"]["flow_rate_kgh"] for r in records
              if r.get("quantification", {}).get("status") == "quantified"
              and r["quantification"].get("flow_rate_kgh_old") is not None]
    mean_old = sum(old_qs) / len(old_qs) if old_qs else 0.0
    mean_new = sum(new_qs) / len(new_qs) if new_qs else 0.0
    return n_updated, mean_old, mean_new


def main():
    summary = []
    for site_name, cfg in SITES.items():
        if not cfg["json_in"].exists():
            log.warning("JSON not found, skipping: %s", cfg["json_in"])
            continue
        n_updated, mean_old, mean_new = process_site(site_name, cfg)
        summary.append((cfg.get("label", site_name), n_updated, mean_old, mean_new))

    print("\n" + "=" * 75)
    print("SUMMARY — MBSP upgrade across all crops")
    print("=" * 75)
    print(f"{'Crop':<52} {'N':>3} {'Mean Q old':>11} {'Mean Q new':>11} {'Δ%':>7}")
    print("-" * 75)
    for label, n, mo, mn in summary:
        if n == 0:
            print(f"{label:<52} {n:>3}  {'—':>11} {'—':>11} {'—':>7}")
        else:
            dp = (mn - mo) / mo * 100 if mo else float("nan")
            print(f"{label:<52} {n:>3} {mo:>11.0f} {mn:>11.0f} {dp:>+7.1f}%")
    print()
    print("_mbsp.json files written alongside originals.")


if __name__ == "__main__":
    main()
