"""
scripts/audit_detections.py
============================
Read the Belchatow time series and surface every detection record with its
wind speed and flow rate, sorted by Q. Identifies outliers, reports both
mean-based and median-based annualisations, and flags which framing is
defensible against a sceptical reviewer.

Why median may be the right headline
------------------------------------
The detection-day distribution is skewed (long tail). With n=6 detections,
the mean is sensitive to single observations (especially low-wind plumes
where Q = M·u/L blows up). Median × 8760 is more robust and reads as
"typical plume magnitude annualised" rather than "extrapolated upper bound."

Usage
-----
  python scripts/audit_detections.py
  python scripts/audit_detections.py --input results_analysis/belchatow_annual_timeseries.json
"""
import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "results_analysis" / "belchatow_annual_timeseries.json"
CLIMATE_TRACE_2024 = 29636.0


def t95(n):
    table = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
             8: 2.365, 9: 2.306, 10: 2.262, 15: 2.145, 20: 2.093}
    for k in sorted(table):
        if n <= k:
            return table[k]
    return 1.96


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()

    store = json.loads(args.input.read_text())
    records = store.get("records", []) if isinstance(store, dict) else store

    detections = []
    for r in records:
        q = r.get("quantification") or {}
        if q.get("status") != "quantified":
            continue
        flow = q.get("flow_rate_kgh")
        if flow is None:
            continue
        det = r.get("detection") or {}
        detections.append({
            "month":     r.get("month") or r.get("acquisition_date"),
            "scene":     r.get("npy") or r.get("scene_id"),
            "sc_ratio":  det.get("sc_ratio"),
            "cfar":      det.get("cfar_detect"),
            "Q_kgh":     flow,
            "Q_lo":      q.get("flow_rate_lower_kgh"),
            "Q_hi":      q.get("flow_rate_upper_kgh"),
            "wind_ms":   q.get("wind_speed_ms"),
            "wind_src":  q.get("wind_source"),
            "n_pixels":  q.get("n_plume_pixels"),
        })

    if not detections:
        print("No quantified detections found.")
        return

    # Sort by Q ascending
    sorted_by_Q = sorted(detections, key=lambda d: d["Q_kgh"])

    print("=" * 95)
    print(f"Belchatow detection audit ({len(detections)} detections)")
    print("=" * 95)
    print(f"{'Date':<12}{'S/C':>10}{'CFAR':>6}{'Q (kg/h)':>12}{'wind m/s':>11}{'wind src':>20}{'plume px':>10}")
    print("-" * 95)
    for d in sorted_by_Q:
        sc = f"{d['sc_ratio']:.2f}" if d['sc_ratio'] is not None else "—"
        w  = f"{d['wind_ms']:.2f}"  if d['wind_ms']  is not None else "—"
        ws = (d['wind_src'] or "")[:18]
        print(f"{d['month']:<12}{sc:>10}{('✓' if d['cfar'] else '·'):>6}"
              f"{d['Q_kgh']:>12.0f}{w:>11}{ws:>20}"
              f"{(d['n_pixels'] if d['n_pixels'] is not None else 0):>10}")

    flows = [d["Q_kgh"] for d in detections]
    winds = [d["wind_ms"] for d in detections if d["wind_ms"] is not None]
    n = len(flows)

    print()
    print("=" * 95)
    print("Distribution of detected flow rates")
    print("=" * 95)
    print(f"  n:              {n}")
    print(f"  min:            {min(flows):>8.0f} kg/h")
    print(f"  Q1:             {statistics.quantiles(flows, n=4)[0] if n >= 4 else flows[0]:>8.0f} kg/h")
    print(f"  median:         {statistics.median(flows):>8.0f} kg/h")
    print(f"  Q3:             {statistics.quantiles(flows, n=4)[2] if n >= 4 else flows[-1]:>8.0f} kg/h")
    print(f"  max:            {max(flows):>8.0f} kg/h")
    print(f"  mean:           {statistics.mean(flows):>8.0f} kg/h")
    print(f"  sd:             {statistics.stdev(flows) if n > 1 else 0:>8.0f} kg/h")

    if winds:
        print()
        print(f"  wind speed min / median / max:  "
              f"{min(winds):.2f} / {statistics.median(winds):.2f} / {max(winds):.2f} m/s")

    # Outlier detection: > 2 × median is suspicious
    median_Q = statistics.median(flows)
    outliers = [d for d in detections if d["Q_kgh"] > 2.5 * median_Q]
    print()
    print("=" * 95)
    print("Outlier analysis (Q > 2.5 × median)")
    print("=" * 95)
    if outliers:
        for d in outliers:
            ratio = d["Q_kgh"] / median_Q
            print(f"  {d['month']} : Q = {d['Q_kgh']:.0f} kg/h  ({ratio:.1f}× median)  "
                  f"wind = {d['wind_ms']:.2f} m/s")
            if d["wind_ms"] and d["wind_ms"] < 1.5:
                print(f"     ↑ LOW-WIND outlier — Q = M·u/L inflates at low wind")
    else:
        print("  None — distribution is reasonably symmetric.")

    # Annualisation framings
    mean_Q = statistics.mean(flows)
    median_Q = statistics.median(flows)
    sd = statistics.stdev(flows) if n > 1 else 0
    sem = sd / (n ** 0.5) if n > 1 else 0
    t_crit = t95(n)
    ci_lo_mean = mean_Q - t_crit * sem
    ci_hi_mean = mean_Q + t_crit * sem

    def t_per_yr(kg): return kg * 8760 / 1000
    def pct(t):       return 100 * t / CLIMATE_TRACE_2024

    print()
    print("=" * 95)
    print(f"Annualisation framings  (Climate TRACE 2024: {CLIMATE_TRACE_2024:.0f} t/yr)")
    print("=" * 95)
    print(f"  Mean-based   : Q={mean_Q:.0f} kg/h  →  {t_per_yr(mean_Q):>7.0f} t/yr  "
          f"({pct(t_per_yr(mean_Q)):.0f}% of inventory)")
    print(f"    95% CI:    : [{max(0, t_per_yr(ci_lo_mean)):>7.0f}, {t_per_yr(ci_hi_mean):>7.0f}] t/yr")
    print(f"  Median-based : Q={median_Q:.0f} kg/h  →  {t_per_yr(median_Q):>7.0f} t/yr  "
          f"({pct(t_per_yr(median_Q)):.0f}% of inventory)")
    print()

    # Outlier-removed mean
    if outliers:
        clean_flows = [d["Q_kgh"] for d in detections if d not in outliers]
        if clean_flows:
            clean_mean = statistics.mean(clean_flows)
            print(f"  Outlier-removed mean (n={len(clean_flows)}):")
            print(f"    Q={clean_mean:.0f} kg/h  →  {t_per_yr(clean_mean):>7.0f} t/yr  "
                  f"({pct(t_per_yr(clean_mean)):.0f}% of inventory)")
            print()

    # Recommendation
    print("=" * 95)
    print("Recommendation")
    print("=" * 95)
    skew_ratio = mean_Q / median_Q if median_Q > 0 else 1.0
    if skew_ratio > 1.5:
        print(f"  Mean / median = {skew_ratio:.2f} — distribution is right-skewed.")
        print(f"  Median-based annualisation is more robust to single outliers and reads")
        print(f"  as 'typical detected magnitude × 8760' rather than 'extrapolated upper bound.'")
        print()
        print(f"  → Recommended report headline: median × 8760 = {t_per_yr(median_Q):.0f} t/yr "
              f"({pct(t_per_yr(median_Q)):.0f}% of Climate TRACE).")
        if outliers and outliers[0]["wind_ms"] and outliers[0]["wind_ms"] < 1.5:
            print(f"  → Flag the low-wind outlier in Section 5 as a documented IME edge case.")
    elif skew_ratio > 1.15:
        print(f"  Mean / median = {skew_ratio:.2f} — mild right skew.")
        print(f"  Either framing is defensible; report both for transparency.")
    else:
        print(f"  Mean / median = {skew_ratio:.2f} — distribution is approximately symmetric.")
        print(f"  Mean-based annualisation is fine to report as the headline.")


if __name__ == "__main__":
    main()
