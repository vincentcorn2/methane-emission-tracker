"""
scripts/belchatow_wind_alignment.py
=====================================
For every Belchatow production-rule detection, compute the angle between
(a) the probability-weighted centroid bearing from the mine source pin
and (b) the contemporaneous ERA5 wind direction.

A static terrain feature would produce a fixed centroid bearing regardless
of wind. A real methane plume drifts downwind, so the centroid bearing
should track the wind vector. Aggregate evidence across N detections is
much harder for a terrain-artifact critique to dismiss than a single date.

Output
------
results_analysis/belchatow_wind_alignment.json
results_analysis/belchatow_wind_alignment.md
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apply_bitemporal_diff import lonlat_to_pixel

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results_analysis"
TIF_DIR = ROOT / "results_bitemporal" / "belchatow"
TIMESERIES_JSON = OUT_DIR / "belchatow_annual_timeseries.json"

# Climate TRACE mine centroid
MINE_LAT = 51.242
MINE_LON = 19.275
SEARCH_RADIUS_M = 5000   # 5 km
PIXEL_M = 10
PROB_FLOOR = 0.18


def bearing_deg(from_pixel, to_pixel, transform):
    """Compute compass bearing FROM from_pixel TO to_pixel (in pixel space).
    Uses the TIF's affine transform to convert to projected coordinates."""
    from rasterio.transform import xy
    x1, y1 = xy(transform, from_pixel[0], from_pixel[1])
    x2, y2 = xy(transform, to_pixel[0], to_pixel[1])
    dx = x2 - x1
    dy = y2 - y1
    # Bearing in compass degrees (0 = north, 90 = east)
    angle_math = math.atan2(dx, dy)  # 0 when dy>0 (north), pi/2 when dx>0 (east)
    bearing = (math.degrees(angle_math)) % 360
    return bearing


def angular_diff(a, b):
    """Smallest angle between two compass bearings (0-180 degrees)."""
    d = abs(a - b) % 360
    return min(d, 360 - d)


def wind_to_bearing(wind_dir_from_deg):
    """ERA5 wind_dir is FROM direction. Downwind plume drifts TO direction = FROM + 180."""
    return (wind_dir_from_deg + 180) % 360


def main():
    if not TIMESERIES_JSON.exists():
        print(f"Missing {TIMESERIES_JSON}")
        return

    store = json.loads(TIMESERIES_JSON.read_text())
    records = store.get("records", []) if isinstance(store, dict) else store

    results = []
    for r in records:
        det = r.get("detection") or {}
        if not (det.get("cfar_detect") and (det.get("sc_cfar") or 0) > 4.1052):
            continue

        quant = r.get("quantification") or {}
        wind_dir = quant.get("wind_dir_deg")
        wind_ms = quant.get("wind_speed_ms")
        if wind_dir is None:
            continue

        scene_id = (r.get("npy") or "").replace(".npy", "") or r.get("scene_id", "")
        tif = TIF_DIR / f"original_{scene_id}.tif"
        if not tif.exists():
            continue

        with rasterio.open(tif) as src:
            prob = src.read(1).astype(np.float32)
            transform = src.transform
            crs = src.crs

        # lonlat_to_pixel signature is (tif_path, lon, lat) — pass lon first.
        mine_row, mine_col = lonlat_to_pixel(tif, MINE_LON, MINE_LAT)

        # Define search box around the mine
        half_px = SEARCH_RADIUS_M // PIXEL_M
        H, W = prob.shape
        r0 = max(0, mine_row - half_px)
        r1 = min(H, mine_row + half_px)
        c0 = max(0, mine_col - half_px)
        c1 = min(W, mine_col + half_px)
        prob_window = prob[r0:r1, c0:c1]

        # Find probability-weighted centroid above PROB_FLOOR
        mask = prob_window >= PROB_FLOOR
        if mask.sum() < 10:
            continue
        rows, cols = np.where(mask)
        weights = prob_window[rows, cols]
        cy_local = float(np.average(rows, weights=weights))
        cx_local = float(np.average(cols, weights=weights))
        cy = r0 + cy_local
        cx = c0 + cx_local

        # Compute bearing from mine to centroid
        centroid_bearing = bearing_deg((mine_row, mine_col), (cy, cx), transform)
        downwind_bearing = wind_to_bearing(wind_dir)
        delta = angular_diff(centroid_bearing, downwind_bearing)

        # Distance from mine to centroid (m)
        d_row = (cy - mine_row) * PIXEL_M
        d_col = (cx - mine_col) * PIXEL_M
        dist_m = math.hypot(d_row, d_col)

        results.append({
            "date":              r.get("month"),
            "scene":             scene_id,
            "sc_cfar":           det.get("sc_cfar"),
            "centroid_bearing":  round(centroid_bearing, 1),
            "wind_dir_from":     round(wind_dir, 1),
            "wind_to_bearing":   round(downwind_bearing, 1),
            "angular_diff_deg":  round(delta, 1),
            "wind_aligned":      delta < 45,         # within one compass octant
            "wind_ms":           wind_ms,
            "dist_to_centroid_m": round(dist_m, 0),
            "n_plume_pixels":    int(mask.sum()),
        })

    # Summarise
    if not results:
        print("No detections with wind metadata found.")
        return

    aligned = sum(1 for r in results if r["wind_aligned"])
    deltas = [r["angular_diff_deg"] for r in results]
    median_delta = float(np.median(deltas))
    mean_delta = float(np.mean(deltas))

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "belchatow_wind_alignment.json").write_text(json.dumps({
        "n_detections":    len(results),
        "n_wind_aligned":  aligned,
        "fraction_aligned": aligned / len(results),
        "median_angular_diff_deg": median_delta,
        "mean_angular_diff_deg":   mean_delta,
        "results":         results,
    }, indent=2))

    md = []
    md.append("# Bełchatów per-detection wind alignment\n")
    md.append(f"For each production-rule detection at Bełchatów, this checks whether the "
              f"probability-weighted centroid of the plume sits in the direction the "
              f"contemporaneous ERA5 wind would carry a plume from the mine source pin.\n")
    md.append(f"**N detections analysed:** {len(results)}")
    md.append(f"**N wind-aligned (angular diff < 45°):** {aligned} ({aligned/len(results):.1%})")
    md.append(f"**Median angular difference:** {median_delta:.1f}°")
    md.append(f"**Mean angular difference:** {mean_delta:.1f}°\n")
    md.append("If the model were firing on a static terrain feature, the centroid bearing "
              "would be roughly constant across acquisitions regardless of wind direction; "
              "the expected angular_diff_deg distribution would be uniform on [0, 180] with "
              "mean 90°. A mean substantially below 90° indicates the centroid systematically "
              "tracks the wind vector, which is the expected behaviour of a real downwind "
              "plume from the mine source.\n")
    md.append("## Per-detection table\n")
    md.append("| Date | sc_cfar | Wind from (°) | Centroid bearing (°) | Δ (°) | Aligned | Dist (m) |")
    md.append("|---|---|---|---|---|---|---|")
    for r in sorted(results, key=lambda x: x["date"] or ""):
        md.append(f"| {r['date']} | {r['sc_cfar']:.1f} | {r['wind_dir_from']} | "
                  f"{r['centroid_bearing']} | {r['angular_diff_deg']} | "
                  f"{'✓' if r['wind_aligned'] else '·'} | {r['dist_to_centroid_m']:.0f} |")

    (OUT_DIR / "belchatow_wind_alignment.md").write_text("\n".join(md))

    print()
    print("=" * 70)
    print("BEŁCHATÓW PER-DETECTION WIND ALIGNMENT")
    print("=" * 70)
    print(f"N detections analysed: {len(results)}")
    print(f"N wind-aligned (Δ < 45°): {aligned} ({aligned/len(results):.0%})")
    print(f"Median angular diff: {median_delta:.1f}°")
    print(f"Mean angular diff:   {mean_delta:.1f}°")
    print()
    print("If random (terrain artifact): mean ≈ 90°.")
    print("If wind-aligned (real plume): mean ≪ 90°.")
    print()
    print(f"Wrote: {OUT_DIR / 'belchatow_wind_alignment.md'}")
    print(f"Wrote: {OUT_DIR / 'belchatow_wind_alignment.json'}")


if __name__ == "__main__":
    main()
