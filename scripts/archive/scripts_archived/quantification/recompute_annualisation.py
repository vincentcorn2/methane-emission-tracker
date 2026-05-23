"""
scripts/recompute_annualisation.py
===================================
Post-process the Bełchatów time series and report every annualisation framing
side-by-side, with sampling-error confidence intervals derived from the
detection distribution. Read-only — does not modify the source JSON.

Usage:
    python scripts/recompute_annualisation.py
    python scripts/recompute_annualisation.py --input results_analysis/belchatow_max_data.json
    python scripts/recompute_annualisation.py --floor 200 --floor 300 --floor 500

Why three framings?
  At a continuous emitter, non-detection months are missing observations,
  not zero observations. Climate TRACE confirms Bełchatów emits 1,700–2,300
  t CH4 every month including January and February. The three framings
  bracket the true annual:
    upper          = mean(detections) × 8760
    floor_imputed  = (mean × det_rate + Q_floor × non_det_rate) × 8760
    lower_bound    = mean(detections) × 8760 × det_rate
"""

import argparse
import json
import statistics
from pathlib import Path

DEFAULT_INPUT = Path("results_analysis/belchatow_annual_timeseries.json")
CLIMATE_TRACE_2024_ANNUAL_T = 29636.0
DETECTION_FLOORS_KGH = [200.0, 300.0, 500.0]   # Varon 2021 / Sherwin 2024 range


def t95_critical(n: int) -> float:
    """Two-tailed t-critical at alpha=0.05 for n-1 degrees of freedom.
    Hardcoded table for small n (statsmodels would be overkill here)."""
    table = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
             8: 2.365, 9: 2.306, 10: 2.262, 15: 2.145, 20: 2.093,
             25: 2.064, 30: 2.045, 40: 2.021, 60: 2.000, 120: 1.980}
    keys = sorted(table.keys())
    for k in keys:
        if n <= k:
            return table[k]
    return 1.96  # z-critical for large n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="Path to time series JSON")
    parser.add_argument("--floor", type=float, action="append",
                        help="Detection floor (kg/h) for imputation (repeatable)")
    args = parser.parse_args()

    floors = args.floor or DETECTION_FLOORS_KGH

    store = json.loads(args.input.read_text())
    records = store.get("records", [])

    # Partial-swath fingerprint: sc_ratio == 1.0 exactly with cv_ctrl == 0 means
    # CH4Net inference ran on a degenerate (partial-swath) tile. These records
    # are MISSING OBSERVATIONS, not non-detections, and must be excluded from
    # the detection-rate denominator. Same logic as repair_backfill_coverage.py.
    def is_degenerate(detection):
        if not detection:
            return False
        sc = detection.get("sc_ratio")
        cv = detection.get("cv_ctrl")
        if sc is None:
            return False
        # Exact S/C = 1.0 with cv_ctrl = 0 → uniform probability map → partial swath
        if abs(sc - 1.0) < 1e-6 and (cv is None or abs(cv) < 1e-6):
            return True
        # Alternate fingerprint: site_mean == ctrl_mean within 1e-6
        sm = detection.get("site_mean")
        cm = detection.get("ctrl_mean")
        if sm is not None and cm is not None and abs(sm - cm) < 1e-6:
            return True
        return False

    flows = []
    n_attempted = 0
    n_degenerate = 0
    n_real_nondet = 0

    for r in records:
        detection = r.get("detection")
        if detection is None:
            continue
        n_attempted += 1
        if is_degenerate(detection):
            n_degenerate += 1
            continue
        q = r.get("quantification") or {}
        if q.get("status") == "quantified" and q.get("flow_rate_kgh") is not None:
            flows.append(float(q["flow_rate_kgh"]))
        else:
            n_real_nondet += 1

    n_det = len(flows)
    n_obs = n_attempted - n_degenerate           # valid observations only
    n_nondet = n_real_nondet                     # real non-detections only

    if n_det == 0:
        print("No detections in time series — nothing to annualise.")
        return

    mean = statistics.mean(flows)
    sd = statistics.stdev(flows) if n_det > 1 else 0.0
    sem = sd / (n_det ** 0.5) if n_det > 1 else 0.0
    t95 = t95_critical(n_det)
    ci_lo_mean = mean - t95 * sem
    ci_hi_mean = mean + t95 * sem

    det_rate = n_det / n_obs
    nondet_rate = n_nondet / n_obs

    print("=" * 78)
    print(f"Time series source: {args.input}")
    print(f"Total acquisitions attempted: {n_attempted}")
    print(f"  Valid observations:         {n_obs}  ({n_det} detections, {n_nondet} real non-detections)")
    print(f"  Excluded (partial-swath / degenerate): {n_degenerate}")
    print(f"Detection rate (det / valid obs): {det_rate:.2%}")
    print()
    print(f"Per-detection Q (kg/h)")
    print(f"  min / median / max:  {min(flows):.0f} / {statistics.median(flows):.0f} / {max(flows):.0f}")
    print(f"  mean ± sd:           {mean:.0f} ± {sd:.0f}")
    print(f"  SEM:                 {sem:.0f}")
    print(f"  95% CI on mean:      [{ci_lo_mean:.0f}, {ci_hi_mean:.0f}] kg/h  (t-dist, df={n_det-1})")
    print()
    print("=" * 78)
    print("Annualisation framings (t/yr)")
    print("=" * 78)

    def t_per_yr(kg_per_h):
        return kg_per_h * 8760 / 1000

    # (1) Upper — non-detections imputed at the detection-day mean
    upper_point = t_per_yr(mean)
    upper_lo = t_per_yr(ci_lo_mean)
    upper_hi = t_per_yr(ci_hi_mean)
    print(f"\nUpper (non-det = detection-day mean):")
    print(f"  Point estimate:  {upper_point:>7.0f} t/yr")
    print(f"  95% sampling CI: [{upper_lo:>7.0f}, {upper_hi:>7.0f}] t/yr")
    print(f"  ±30% IME bounds: [{0.7*upper_point:>7.0f}, {1.3*upper_point:>7.0f}] t/yr")

    # (2) Floor-imputed (multiple flavours)
    print(f"\nFloor-imputed (non-det = Q_floor):")
    for floor in floors:
        weighted_mean = mean * det_rate + floor * nondet_rate
        weighted_mean_lo = ci_lo_mean * det_rate + floor * nondet_rate
        weighted_mean_hi = ci_hi_mean * det_rate + floor * nondet_rate
        point = t_per_yr(weighted_mean)
        ci_lo = t_per_yr(weighted_mean_lo)
        ci_hi = t_per_yr(weighted_mean_hi)
        print(f"  Q_floor = {floor:>4.0f} kg/h: point {point:>7.0f} t/yr   "
              f"95% sampling CI [{ci_lo:>7.0f}, {ci_hi:>7.0f}]")

    # (3) Lower bound — non-detections treated as zero (strawman)
    lower_point = t_per_yr(mean) * det_rate
    lower_lo = t_per_yr(ci_lo_mean) * det_rate
    lower_hi = t_per_yr(ci_hi_mean) * det_rate
    print(f"\nLower bound (non-det = 0, strawman):")
    print(f"  Point estimate:  {lower_point:>7.0f} t/yr")
    print(f"  95% sampling CI: [{lower_lo:>7.0f}, {lower_hi:>7.0f}] t/yr")

    print()
    print("=" * 78)
    print(f"Climate TRACE 2024 reported total: {CLIMATE_TRACE_2024_ANNUAL_T:.0f} t/yr")
    print("=" * 78)

    # Check whether Climate TRACE falls inside the upper-framing sampling CI
    inside = upper_lo <= CLIMATE_TRACE_2024_ANNUAL_T <= upper_hi
    print(f"\nClimate TRACE total inside upper-framing 95% sampling CI: {inside}")
    if inside:
        print("  → The inventory total is within the satellite-derived sampling")
        print("    uncertainty. Report Section 4.1 should lead with this.")
    else:
        if CLIMATE_TRACE_2024_ANNUAL_T > upper_hi:
            print("  → Inventory exceeds the upper bound of the satellite CI.")
            print("    Consistent with the published S2 IME under-recovery range")
            print("    (Varon 2021, Sherwin 2024 report ~30–60% recovery on")
            print("    coal-mine fugitives). Cite this in Section 4.")
        else:
            print("  → Inventory below the lower bound of the satellite CI.")
            print("    Investigate: possible over-quantification or inventory dispute.")


if __name__ == "__main__":
    main()
