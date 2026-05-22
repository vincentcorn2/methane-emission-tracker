"""
scripts/rybnik_centroid_vs_wind.py
===================================
Compute the CH4Net probability centroid at Rybnik on 2025-03-22 and compare
its displacement direction from the Carbon Mapper source pin to the ERA5
wind direction at S2 overpass time.

Purpose: Section 5.3 paragraph in the EIB report.

Geometric question: is the CH4Net signal located downwind of the
Carbon-Mapper-confirmed source, as a real plume would be?
"""

import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import rasterio
import rasterio.warp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.era5_client import ERA5Client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("rybnik_centroid")

# Anchors
CM_LAT, CM_LON = 50.0781, 18.5451

S2_DATE = "2025-03-22"
S2_HOUR = "09:50"

TIF_PATH = Path(
    "results_bitemporal/rybnik_cm/"
    "original_S2B_MSIL1C_20250322T095029_N0511_R079_T34UCA_20250322T114604.tif"
)

CM_WIND_DATE     = "2025-03-21"
CM_WIND_DIR_DEG  = 215.67
CM_WIND_SPEED_MS = 3.08

SEARCH_RADIUS_M = 3000
PROB_THRESHOLD  = 0.5   # hard probability threshold; only high-confidence pixels

OUT_JSON = Path("results_analysis/rybnik_centroid_vs_wind.json")


# Geodesy helpers
def meters_per_deg(lat):
    m_per_lat = 111320.0
    m_per_lon = 111320.0 * math.cos(math.radians(lat))
    return m_per_lat, m_per_lon


def bearing_deg(lat1, lon1, lat2, lon2):
    m_per_lat, m_per_lon = meters_per_deg((lat1 + lat2) / 2.0)
    dx = (lon2 - lon1) * m_per_lon
    dy = (lat2 - lat1) * m_per_lat
    return (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0


def angular_diff(a, b):
    d = (a - b + 180.0) % 360.0 - 180.0
    return abs(d)


def compass_octant(deg):
    names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return names[int(((deg + 22.5) % 360) // 45)]


# Centroid (CRS-aware)
def compute_centroid(tif_path, lat, lon, radius_m, threshold):
    with rasterio.open(tif_path) as src:
        xs, ys = rasterio.warp.transform(
            "EPSG:4326", src.crs, [lon], [lat]
        )
        pin_row, pin_col = rasterio.transform.rowcol(
            src.transform, xs[0], ys[0]
        )
        pin_row, pin_col = int(pin_row), int(pin_col)

        pixel_size_m = abs(src.transform.a)
        half_px = int(round(radius_m / pixel_size_m))

        H, W = src.height, src.width
        r0 = max(0, pin_row - half_px)
        r1 = min(H, pin_row + half_px + 1)
        c0 = max(0, pin_col - half_px)
        c1 = min(W, pin_col + half_px + 1)
        if r1 <= r0 or c1 <= c0:
            return {
                "error": "pin_outside_tif",
                "pin_pixel": [pin_row, pin_col],
                "tif_shape": [H, W],
            }

        prob = src.read(1, window=((r0, r1), (c0, c1))).astype(np.float32)

        thresh = float(threshold)
        mask = prob >= thresh
        n_active = int(mask.sum())
        if n_active == 0:
            return {"error": "no_pixels_above_threshold",
                    "threshold": thresh,
                    "prob_max_in_window": float(prob.max())}

        rows, cols = np.where(mask)
        weights = prob[rows, cols]
        cr_local = float(np.average(rows, weights=weights))
        cc_local = float(np.average(cols, weights=weights))
        cr = cr_local + r0
        cc = cc_local + c0

        cx_utm, cy_utm = src.xy(cr, cc)
        lon_out, lat_out = rasterio.warp.transform(
            src.crs, "EPSG:4326", [cx_utm], [cy_utm]
        )
        c_lat, c_lon = lat_out[0], lon_out[0]

    return {
        "pin_pixel": [pin_row, pin_col],
        "window_pixels": [int(r0), int(r1), int(c0), int(c1)],
        "pixel_size_m": round(pixel_size_m, 3),
        "prob_threshold": round(thresh, 6),
        "prob_max_in_window": round(float(prob.max()), 6),
        "prob_mean_in_window": round(float(prob.mean()), 6),
        "n_pixels_above": n_active,
        "centroid_pixel": [round(cr, 2), round(cc, 2)],
        "centroid_lat": round(c_lat, 6),
        "centroid_lon": round(c_lon, 6),
    }


def main():
    if not TIF_PATH.exists():
        log.error("TIF not found: %s", TIF_PATH)
        sys.exit(1)

    log.info("Pulling ERA5 wind at (%.4f, %.4f) on %s %s UTC...",
             CM_LAT, CM_LON, S2_DATE, S2_HOUR)
    era5 = ERA5Client().get_wind(CM_LAT, CM_LON, S2_DATE, S2_HOUR)
    dir_str = "{:.1f}".format(era5["wind_dir_deg"]) if era5["wind_dir_deg"] is not None else "-"
    log.info("ERA5: speed=%.2f m/s  dir=%s deg  source=%s",
             era5["wind_speed_ms"], dir_str, era5["wind_source"])

    log.info("Computing CH4Net centroid in %.0f m window around CM pin "
             "(hard threshold prob >= %.2f)...", SEARCH_RADIUS_M, PROB_THRESHOLD)
    cen = compute_centroid(TIF_PATH, CM_LAT, CM_LON,
                           SEARCH_RADIUS_M, PROB_THRESHOLD)
    if "error" in cen:
        log.error("Centroid failed: %s", cen)
        sys.exit(1)
    log.info("Centroid: (%.6f, %.6f)  from %d active pixels  "
             "(prob threshold = %.4f, max in window = %.4f)",
             cen["centroid_lat"], cen["centroid_lon"],
             cen["n_pixels_above"], cen["prob_threshold"],
             cen["prob_max_in_window"])

    m_per_lat, m_per_lon = meters_per_deg(CM_LAT)
    dx_m = (cen["centroid_lon"] - CM_LON) * m_per_lon
    dy_m = (cen["centroid_lat"] - CM_LAT) * m_per_lat
    displacement_m = math.hypot(dx_m, dy_m)
    centroid_bearing = bearing_deg(CM_LAT, CM_LON,
                                   cen["centroid_lat"], cen["centroid_lon"])

    if era5["wind_dir_deg"] is not None:
        expected_plume_bearing = (era5["wind_dir_deg"] + 180.0) % 360.0
        ang_diff_era5 = angular_diff(centroid_bearing, expected_plume_bearing)
    else:
        expected_plume_bearing = None
        ang_diff_era5 = None

    expected_plume_bearing_cm = (CM_WIND_DIR_DEG + 180.0) % 360.0
    ang_diff_cm = angular_diff(centroid_bearing, expected_plume_bearing_cm)

    print("\n" + "=" * 72)
    print("Rybnik centroid vs wind direction -- S2B {} {} UTC".format(S2_DATE, S2_HOUR))
    print("=" * 72)
    print("Carbon Mapper pin:         {:.4f} N  {:.4f} E".format(CM_LAT, CM_LON))
    print("CH4Net centroid:           {:.4f} N  {:.4f} E".format(
        cen["centroid_lat"], cen["centroid_lon"]))
    print("Displacement:              {:>6.0f} m  (dx={:+6.0f} m E, dy={:+6.0f} m N)".format(
        displacement_m, dx_m, dy_m))
    print("Centroid bearing from pin: {:6.1f} deg ({})".format(
        centroid_bearing, compass_octant(centroid_bearing)))
    print("  Active pixels (prob >= {:.4f}): {}".format(
        cen["prob_threshold"], cen["n_pixels_above"]))
    print("  Max prob in window:     {:.4f}".format(cen["prob_max_in_window"]))
    print("-" * 72)

    if era5["wind_dir_deg"] is not None:
        print("ERA5 wind {} {}:  FROM {:6.1f} deg ({}), speed {:.2f} m/s".format(
            S2_DATE, S2_HOUR, era5["wind_dir_deg"],
            compass_octant(era5["wind_dir_deg"]), era5["wind_speed_ms"]))
        print("Expected plume direction:  TO  {:6.1f} deg ({})".format(
            expected_plume_bearing, compass_octant(expected_plume_bearing)))
        print("Angular difference (ERA5): {:6.1f} deg".format(ang_diff_era5))
    else:
        print("ERA5 wind: FALLBACK USED ({})".format(era5.get("_fallback_reason", "?")))

    print("\nCM wind on {} (one day prior, sanity check):".format(CM_WIND_DATE))
    print("  FROM {:6.1f} deg ({}), speed {:.2f} m/s".format(
        CM_WIND_DIR_DEG, compass_octant(CM_WIND_DIR_DEG), CM_WIND_SPEED_MS))
    print("  Expected plume direction: TO {:6.1f} deg ({})".format(
        expected_plume_bearing_cm, compass_octant(expected_plume_bearing_cm)))
    print("  Angular difference (CM):  {:6.1f} deg".format(ang_diff_cm))
    print("=" * 72)

    record = {
        "site": "rybnik_cm",
        "tif": str(TIF_PATH),
        "s2_acquisition": "{}T{}".format(S2_DATE, S2_HOUR),
        "cm_pin": {"lat": CM_LAT, "lon": CM_LON},
        "centroid": cen,
        "displacement_m": round(displacement_m, 1),
        "displacement_dx_m": round(dx_m, 1),
        "displacement_dy_m": round(dy_m, 1),
        "centroid_bearing_deg": round(centroid_bearing, 2),
        "era5_wind": {
            **era5,
            "expected_plume_bearing_deg": (
                round(expected_plume_bearing, 2)
                if expected_plume_bearing is not None else None
            ),
            "angular_diff_deg": (
                round(ang_diff_era5, 2)
                if ang_diff_era5 is not None else None
            ),
        },
        "cm_wind_one_day_prior": {
            "date": CM_WIND_DATE,
            "wind_dir_deg": CM_WIND_DIR_DEG,
            "wind_speed_ms": CM_WIND_SPEED_MS,
            "expected_plume_bearing_deg": round(expected_plume_bearing_cm, 2),
            "angular_diff_deg": round(ang_diff_cm, 2),
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(record, f, indent=2)
    log.info("Wrote %s", OUT_JSON)


if __name__ == "__main__":
    main()
