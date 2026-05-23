"""
scripts/analysis/annualise_belchatow.py
========================================
Compute defensible annualised CH4 emission estimate from the
belchatow_annual_timeseries.json, applying QC filters to exclude
spurious detections before annualisation.

Defensible detection criteria (all three must pass):
  - site_mean >= 0.05  (minimum real CH4Net signal above background noise)
  - flow_rate_kgh >= 200  (CH4Net+S2 empirical detection floor)
  - sc_ratio < 20  (exclude implausibly high ratios from surface artefacts)
  - not uniform_field  (spectrally homogeneous scene)
  - not partial_swath  (site outside S2 acquisition swath)

Annualisation uses month-level aggregation: a calendar month is a
"detection month" if any single qualifying acquisition detected CH4.
Non-detection months are treated as missing observations (Q < floor),
not zero-emission months.

Run:
    conda activate methane
    python scripts/analysis/annualise_belchatow.py
"""

import json
from pathlib import Path
from collections import defaultdict

import numpy as np

OUT_JSON = Path("results_analysis/belchatow_annual_timeseries.json")

# ── QC thresholds ─────────────────────────────────────────────────────────────
MIN_SITE_MEAN   = 0.05    # minimum absolute CH4Net probability at site crop
MIN_Q_KGH       = 200.0   # empirical detection floor (smallest resolved Q = 282 kg/h)
MAX_SC_RATIO    = 20.0    # S/C > 20 with no corresponding site_mean → surface artefact
DETECTION_FLOOR = 300.0   # kg/h — imputed for non-detection months


def main():
    with open(OUT_JSON) as f:
        store = json.load(f)

    records = store["records"]
    print(f"Total records in JSON:  {len(records)}")

    # ── Flag every acquisition ────────────────────────────────────────────────
    n_uf = n_partial = n_low_sm = n_low_q = n_high_sc = 0
    for r in records:
        det  = r.get("detection", {})
        q_rec = r.get("quantification", {})

        if not det:
            continue  # no_products / download_failed

        uf       = det.get("uniform_field", False)
        partial  = r.get("partial_swath", None)   # from swath audit if run
        sm       = det.get("site_mean") or 0.0
        sc       = det.get("sc_ratio")  or 0.0
        q        = q_rec.get("flow_rate_kgh") or 0.0
        status   = q_rec.get("status", "")

        exclude_reason = None
        if uf:
            exclude_reason = "uniform_field"
            n_uf += 1
        elif partial is True:
            exclude_reason = "partial_swath"
            n_partial += 1
        elif sm < MIN_SITE_MEAN and status == "quantified":
            exclude_reason = f"low_site_mean ({sm:.4f})"
            n_low_sm += 1
        elif status == "quantified" and q < MIN_Q_KGH:
            exclude_reason = f"below_q_floor ({q:.0f} kg/h)"
            n_low_q += 1
        elif status == "quantified" and sc > MAX_SC_RATIO:
            exclude_reason = f"high_sc ({sc:.1f})"
            n_high_sc += 1

        r["_qc_exclude"] = exclude_reason  # temp flag for reporting

    # ── Per-month aggregation ─────────────────────────────────────────────────
    # Valid observation month: at least one acquisition with a real detection result
    # (not uniform_field, not partial_swath).
    # Detection month: at least one acquisition passed ALL QC filters and quantified.

    month_valid   = defaultdict(list)   # month → list of valid acq records
    month_det     = defaultdict(list)   # month → list of qualifying detections

    for r in records:
        det  = r.get("detection", {})
        q_rec = r.get("quantification", {})
        if not det:
            continue
        month = r.get("month", "????")

        uf      = det.get("uniform_field", False)
        partial = r.get("partial_swath", None)

        if not uf and partial is not True:
            month_valid[month].append(r)

        if (r.get("_qc_exclude") is None
                and q_rec.get("status") == "quantified"
                and q_rec.get("flow_rate_kgh") is not None):
            month_det[month].append(r)

    obs_months = sorted(month_valid.keys())
    det_months = sorted(month_det.keys())
    n_obs      = len(obs_months)
    n_det      = len(det_months)
    n_ndet     = n_obs - n_det

    # ── Flow rates ────────────────────────────────────────────────────────────
    # For months with multiple detections, take the acquisition with highest Q.
    best_flows = []
    for m in det_months:
        qs = [r["quantification"]["flow_rate_kgh"] for r in month_det[m]]
        best_flows.append(max(qs))

    mean_q  = float(np.mean(best_flows)) if best_flows else 0.0
    upper   = mean_q * 8760 / 1000
    floor_i = (mean_q * (n_det / n_obs) +
               DETECTION_FLOOR * (n_ndet / n_obs)) * 8760 / 1000
    lower   = mean_q * 8760 / 1000 * (n_det / n_obs)

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("Bełchatów QC-filtered annualisation")
    print(f"QC filters: site_mean≥{MIN_SITE_MEAN}, Q≥{MIN_Q_KGH:.0f} kg/h, "
          f"S/C<{MAX_SC_RATIO:.0f}, not uniform_field, not partial_swath")
    print(f"{'='*65}")
    print(f"Acquisitions with detection result:  "
          f"{sum(1 for r in records if r.get('detection'))}")
    print(f"  Excluded — uniform field:          {n_uf}")
    print(f"  Excluded — partial swath:          {n_partial}")
    print(f"  Excluded — site_mean < {MIN_SITE_MEAN}:       {n_low_sm}")
    print(f"  Excluded — Q < {MIN_Q_KGH:.0f} kg/h:          {n_low_q}")
    print(f"  Excluded — S/C > {MAX_SC_RATIO:.0f}:              {n_high_sc}")
    print(f"Valid observation months:            {n_obs}  {obs_months[0]}–{obs_months[-1]}")
    print(f"Detection months:                    {n_det}")
    print(f"Non-detection months:                {n_ndet}")
    print(f"Detection rate (by month):           {n_det/n_obs*100:.1f}%")

    print(f"\nQualifying detections:")
    for m in det_months:
        for r in month_det[m]:
            q   = r["quantification"]["flow_rate_kgh"]
            sc  = r["detection"]["sc_ratio"]
            sm  = r["detection"]["site_mean"]
            acq = r.get("acquisition_date", "")
            print(f"  {m} [{acq}]  S/C={sc:.2f}  site_mean={sm:.3f}  Q={q:.0f} kg/h")

    print(f"\nMean flow (detection months, best Q per month):  {mean_q:.0f} kg/h")
    print(f"Range: {min(best_flows):.0f}–{max(best_flows):.0f} kg/h")

    print(f"\nAnnualisation — three framings:")
    print(f"  Upper  (det-mean × 8760):                    {upper:.0f} t/yr")
    print(f"  Floor-imputed (Q_floor={DETECTION_FLOOR:.0f} kg/h):        {floor_i:.0f} t/yr")
    print(f"  Lower  (non-det = 0):                        {lower:.0f} t/yr")

    # Clean up temp flags
    for r in records:
        r.pop("_qc_exclude", None)


if __name__ == "__main__":
    main()
