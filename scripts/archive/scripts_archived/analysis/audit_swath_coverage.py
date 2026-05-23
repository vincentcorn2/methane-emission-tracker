"""
scripts/analysis/audit_swath_coverage.py
==========================================
Post-processing swath audit for belchatow_annual_timeseries.json.

For each processed record, checks whether the site crop actually fell
inside the Sentinel-2 acquisition swath by measuring the fraction of
non-zero pixels in the site crop of the raw .npy.

Partial-swath tiles (site outside swath) produce a uniform CH4Net
probability field and a spurious sc_ratio = 1.0 exactly. These months
must be excluded from the n_obs denominator in annualisation.

Outputs:
  - Annotates every record with "swath_valid_frac" and "partial_swath" flag
  - Recomputes annualisation using only valid observations
  - Saves updated JSON (backup of original kept as .json.bak2)

Run:
    conda activate methane
    python scripts/analysis/audit_swath_coverage.py
"""

import json
import sys
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "scripts" / "detection"))

from apply_bitemporal_diff import SC_CROP_PX, lonlat_to_pixel

# ── Site config (must match belchatow_annual_timeseries.py) ───────────────────
LAT, LON        = 51.242, 19.275
NPY_CACHE       = _root / "data" / "npy_cache"
OUT_JSON        = _root / "results_analysis" / "belchatow_annual_timeseries.json"
DETECTION_FLOOR_KGH = 300.0
MIN_VALID_FRAC  = 0.50   # same threshold used in historical_backfill_timeseries.py


def site_valid_fraction(npy_path: Path, tif_path: Path) -> float:
    """Return fraction of non-zero pixels in the 100×100 site crop of the npy.

    Near-zero means the site fell outside the actual S2 swath boundary.
    Uses any one band (B11, index 10) as a proxy for valid data.
    """
    import rasterio

    arr = np.load(npy_path)            # (H, W, 12) uint8
    with rasterio.open(tif_path) as src:
        row, col = lonlat_to_pixel(tif_path, LON, LAT)

    half = SC_CROP_PX // 2
    H, W = arr.shape[:2]
    r0 = max(0, row - half)
    r1 = min(H, row + half)
    c0 = max(0, col - half)
    c1 = min(W, col + half)

    crop = arr[r0:r1, c0:c1, 10]      # B11 band — 0 outside swath
    if crop.size == 0:
        return 0.0
    return float((crop > 0).sum()) / crop.size


def main():
    with open(OUT_JSON) as f:
        store = json.load(f)

    # Backup before modifying
    bak2 = OUT_JSON.with_suffix(".json.bak2")
    with open(bak2, "w") as f:
        json.dump(store, f, indent=2)
    print(f"Backup → {bak2.name}")

    records = store["records"]
    n_checked = 0
    n_partial  = 0

    for rec in records:
        if "detection" not in rec:
            # no_products or download_failed — not an observation at all
            rec["partial_swath"] = None
            rec["swath_valid_frac"] = None
            continue

        npy_name = rec.get("npy")
        tif_path = Path(rec["detection"].get("tif", ""))

        if not npy_name or not tif_path.exists():
            rec["partial_swath"] = None
            rec["swath_valid_frac"] = None
            print(f"  {rec['month']}  — TIF or NPY not found, cannot check")
            continue

        npy_path = NPY_CACHE / npy_name
        if not npy_path.exists():
            rec["partial_swath"] = None
            rec["swath_valid_frac"] = None
            print(f"  {rec['month']}  — NPY not in cache, cannot check")
            continue

        vf = site_valid_fraction(npy_path, tif_path)
        is_partial = vf < MIN_VALID_FRAC
        rec["swath_valid_frac"] = round(vf, 4)
        rec["partial_swath"]    = is_partial
        n_checked += 1
        if is_partial:
            n_partial += 1
            print(f"  {rec['month']}  valid_frac={vf:.2f}  ← PARTIAL SWATH  "
                  f"(sc={rec['detection'].get('sc_ratio', '?'):.3f})")
        else:
            print(f"  {rec['month']}  valid_frac={vf:.2f}  OK  "
                  f"sc={rec['detection'].get('sc_ratio', '?'):.3f}")

    print(f"\nChecked {n_checked} tiles — {n_partial} partial swath")

    # ── Recompute annualisation excluding partial-swath months ────────────────
    valid_obs = [
        r for r in records
        if r.get("partial_swath") is False   # explicitly False, not None
    ]
    detected = [
        r for r in valid_obs
        if r.get("quantification", {}).get("status") == "quantified"
    ]

    flows = [r["quantification"]["flow_rate_kgh"] for r in detected
             if r["quantification"].get("flow_rate_kgh") is not None]

    n_obs    = len(valid_obs)
    n_det    = len(flows)
    n_nondet = n_obs - n_det

    print(f"\n{'='*65}")
    print("Swath-corrected annualisation")
    print(f"{'='*65}")
    print(f"Total records:               {len(records)}")
    print(f"Partial-swath excluded:      {n_partial}")
    print(f"No-products excluded:        {sum(1 for r in records if r.get('search',{}).get('status')=='no_products')}")
    print(f"Valid observations:           {n_obs}")
    print(f"Detections (S/C > 1.15):     {n_det}")
    print(f"Non-detections:              {n_nondet}")
    print(f"Detection rate:              {n_det/n_obs*100:.1f}%")

    if flows:
        mean_q  = float(np.mean(flows))
        upper   = mean_q * 8760 / 1000
        floor_i = (mean_q * (n_det / n_obs) +
                   DETECTION_FLOOR_KGH * (n_nondet / n_obs)) * 8760 / 1000
        lower   = mean_q * 8760 / 1000 * (n_det / n_obs)
        print(f"Mean flow (detections):      {mean_q:.0f} kg/h")
        print(f"Range:                       {min(flows):.0f}–{max(flows):.0f} kg/h")
        print(f"\nAnnualisation (swath-corrected):")
        print(f"  Upper  (det-mean × 8760):            {upper:.0f} t/yr")
        print(f"  Floor-imputed (Q_floor={DETECTION_FLOOR_KGH:.0f} kg/h):      {floor_i:.0f} t/yr")
        print(f"  Lower  (non-det = 0):                {lower:.0f} t/yr")

        # Update summary in store
        store["summary"]["swath_corrected"] = {
            "n_valid_observations":   n_obs,
            "n_partial_swath_excluded": n_partial,
            "n_detections":           n_det,
            "n_non_detections":       n_nondet,
            "detection_rate":         round(n_det / n_obs, 4),
            "mean_flow_kg_h":         round(mean_q, 1),
            "min_flow_kg_h":          round(min(flows), 1),
            "max_flow_kg_h":          round(max(flows), 1),
            "annualised_t_per_yr_upper":         round(upper, 1),
            "annualised_t_per_yr_floor_imputed": round(floor_i, 1),
            "annualised_t_per_yr_lower_bound":   round(lower, 1),
            "detection_floor_kgh_assumed":       DETECTION_FLOOR_KGH,
        }

    with open(OUT_JSON, "w") as f:
        json.dump(store, f, indent=2)
    print(f"\nUpdated JSON saved → {OUT_JSON.name}")


if __name__ == "__main__":
    main()
