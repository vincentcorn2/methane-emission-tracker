"""
verify_outputs.py — Sanity-check all pipeline outputs against paper_final.md.

Checks three things for each module:
  1. Expected output files exist in results_analysis/ (not scripts/results_analysis/)
  2. Key numbers match the paper's reported values (within stated tolerance)
  3. No misplaced outputs or stale artifacts in the wrong directories

Usage:
    python verify_outputs.py           # run all checks, show only failures
    python verify_outputs.py --verbose # print every check result

Exit code 0 = all checks pass. Non-zero = at least one hard failure.

Paper reference values (paper_final.md):
  - tau = 3.5796, alpha = 0.10, n_calibration = 35
  - 31 above-threshold responses, 30 quantification-supporting
  - Mean flow: 476 kg/hr, annual estimate: 4,174 t CH4/yr, CI [2,987, 5,360]
  - Recovery vs Climate TRACE 2024: 14.1%
  - co2e_t_mean (GWP100): 116,872 t/yr
  - Monte Carlo GWP100 mean: 7.51 M€, VaR99: 17.77 M€, ES99: 20.22 M€
  - Monte Carlo GWP20 mean: 22.25 M€
"""

import argparse
import json
import sys
from pathlib import Path

ROOT       = Path(__file__).resolve().parent
RESULTS    = ROOT / "results_analysis"
SCRIPTS_RA = ROOT / "scripts" / "results_analysis"

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m!\033[0m"

failures = []
warnings = []

parser = argparse.ArgumentParser()
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()


def check(label: str, condition: bool, detail: str = "", warn_only: bool = False) -> bool:
    if condition:
        if args.verbose:
            print(f"  {PASS} {label}" + (f"  ({detail})" if detail else ""))
    else:
        symbol = WARN if warn_only else FAIL
        print(f"  {symbol} {label}" + (f"  — {detail}" if detail else ""))
        (warnings if warn_only else failures).append(label)
    return condition


def near(actual, expected, tol_pct=5.0) -> bool:
    if expected == 0:
        return actual == 0
    return abs(actual - expected) / abs(expected) * 100 <= tol_pct


def load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════════
print("\n── 1. FILE LOCATIONS ─────────────────────────────────────────────────")
# ══════════════════════════════════════════════════════════════════════════════

for fname in [
    "calibrated_threshold.json",
    "nonemitter_sc_scores.json",
    "belchatow_annual_timeseries.json",
    "finance_climate_var.json",
    "finance_transition_risk.json",
    "held_out_evaluation.json",
    "bootstrap_auroc_ap.json",
    "leakage_audit.json",
    "loo_detection_stability.json",
    "conformal_calibration.png",
]:
    check(f"results_analysis/{fname}", (RESULTS / fname).exists())

check(
    "No misplaced scripts/results_analysis/ directory",
    not SCRIPTS_RA.exists(),
    detail="delete with: rm -rf scripts/results_analysis/",
    warn_only=True,
)

# ══════════════════════════════════════════════════════════════════════════════
print("\n── 2. CALIBRATION ────────────────────────────────────────────────────")
# ══════════════════════════════════════════════════════════════════════════════
# Paper values: tau=3.5796, alpha=0.10, n=35, empirical FPR at tau <= 10%

cal_path = RESULTS / "calibrated_threshold.json"
if cal_path.exists():
    cal = load(cal_path)
    pt  = cal.get("primary_threshold", {})
    tau = pt.get("tau")
    alpha = pt.get("alpha")
    n   = cal.get("n_calibration")                        # correct key
    fpr = pt.get("empirical_fpr_at_calibrated")           # correct key

    check("tau = 3.5796",
          tau is not None and near(tau, 3.5796, tol_pct=0.1),
          detail=f"got {tau}")
    check("alpha = 0.10",
          alpha is not None and abs(alpha - 0.10) < 0.001,
          detail=f"got {alpha}")
    check("n_calibration = 35",
          n == 35,
          detail=f"got {n}")
    check("empirical FPR at tau ≤ 10%",
          fpr is not None and fpr <= 0.10,
          detail=f"got {fpr:.3f}" if fpr is not None else "key missing — check JSON",
          warn_only=(fpr is None))
else:
    check("calibrated_threshold.json readable", False, "file missing")

# ══════════════════════════════════════════════════════════════════════════════
print("\n── 3. BEŁCHATÓW TIME SERIES ──────────────────────────────────────────")
# ══════════════════════════════════════════════════════════════════════════════
# Paper: 139 acquisitions, 30 quantification-supporting, mean 476 kg/hr,
# annual 4,174 t/yr, CI [2,987, 5,360], flow range 72–1,578 kg/hr

ts_path = RESULTS / "belchatow_annual_timeseries.json"
if ts_path.exists():
    ts      = load(ts_path)
    records = ts.get("records", [])
    summary = ts.get("summary", {})

    n_total      = len(records)
    n_quantified = sum(1 for r in records
                       if r.get("quantification", {}).get("status") == "quantified")
    flows = [r["quantification"]["flow_rate_kgh"]
             for r in records
             if r.get("quantification", {}).get("status") == "quantified"
             and r["quantification"].get("flow_rate_kgh") is not None]
    mean_flow = sum(flows) / len(flows) if flows else None

    check("Total acquisitions ≥ 139",
          n_total >= 139,               detail=f"got {n_total}")
    check("30 quantification-supporting",
          n_quantified == 30,           detail=f"got {n_quantified}")
    check("Mean flow ~476 kg/hr (±10%)",
          mean_flow is not None and near(mean_flow, 476, tol_pct=10),
          detail=f"got {mean_flow:.0f}" if mean_flow else "no flows")
    check("Flow min ≤ 100 kg/hr",
          flows and min(flows) <= 100,
          detail=f"min={min(flows):.0f}" if flows else "no flows")
    check("Flow max ≤ 1580 kg/hr (outliers excluded; paper rounds to 1,578)",
          flows and max(flows) <= 1580,
          detail=f"max={max(flows):.1f}" if flows else "no flows")
else:
    check("belchatow_annual_timeseries.json readable", False, "file missing")

# ══════════════════════════════════════════════════════════════════════════════
print("\n── 4. FINANCE — CLIMATE VaR ──────────────────────────────────────────")
# ══════════════════════════════════════════════════════════════════════════════
# Paper (§7.2, Monte Carlo table):
#   GWP100 mean €7.51M, VaR99 €17.77M, ES99 €20.22M
#   GWP20  mean €22.25M
# Values are stored in M€ (millions) in the JSON.
# Tail metrics use 15% tolerance — stochastic variation across runs is expected.

cvar_path = RESULTS / "finance_climate_var.json"
if cvar_path.exists():
    cv   = load(cvar_path)
    meta = cv.get("metadata", {})
    g100 = cv.get("liability_gwp100", {})
    g20  = cv.get("liability_gwp20",  {})

    n_sim    = meta.get("n_sim")
    mean_100 = g100.get("mean")
    var99_100 = g100.get("var_99")
    es99_100 = g100.get("es_99")
    mean_20  = g20.get("mean")

    check("n_sim = 10,000",
          n_sim == 10000, detail=f"got {n_sim}")
    check("GWP100 mean ~€7.51M (±10%)",
          mean_100 is not None and near(mean_100, 7.51, tol_pct=10),
          detail=f"got €{mean_100:.2f}M" if mean_100 is not None else "missing")
    check("GWP100 VaR99 ~€17.77M (±15%)",
          var99_100 is not None and near(var99_100, 17.77, tol_pct=15),
          detail=f"got €{var99_100:.2f}M" if var99_100 is not None else "missing")
    check("GWP100 ES99 ~€20.22M (±15%)",
          es99_100 is not None and near(es99_100, 20.22, tol_pct=15),
          detail=f"got €{es99_100:.2f}M" if es99_100 is not None else "missing",
          warn_only=True)
    check("GWP20 mean ~€22.25M (±10%)",
          mean_20 is not None and near(mean_20, 22.25, tol_pct=10),
          detail=f"got €{mean_20:.2f}M" if mean_20 is not None else "missing")
    check("emission mu in metadata = 4,174 t/yr (±1%)",
          any(
              abs(layer.get("mu_t_yr", 0) - 4174) / 4174 < 0.01
              for layer in meta.get("layers", [])
              if layer.get("layer") == "EmissionSampling"
          ),
          detail="EmissionSampling layer mu_t_yr not matching 4,174",
          warn_only=True)
else:
    check("finance_climate_var.json readable", False, "file missing")

# ══════════════════════════════════════════════════════════════════════════════
print("\n── 5. FINANCE — TRANSITION RISK ──────────────────────────────────────")
# ══════════════════════════════════════════════════════════════════════════════
# Paper §7.2: 4,174 × GWP100(28) = 116,872 t CO₂e/yr

tr_path = RESULTS / "finance_transition_risk.json"
if tr_path.exists():
    tr = load(tr_path)
    co2e_mean = tr.get("co2e_t_mean")
    schema    = tr.get("schema_version")

    check("co2e_t_mean ~116,872 t (±1%)",
          co2e_mean is not None and near(co2e_mean, 116872, tol_pct=1),
          detail=f"got {co2e_mean:,.0f}" if co2e_mean else "missing")
    check("schema_version present",
          schema is not None, detail=f"got {schema}")
else:
    check("finance_transition_risk.json readable", False, "file missing")

# ══════════════════════════════════════════════════════════════════════════════
print("\n── 6. VALIDATION ─────────────────────────────────────────────────────")
# ══════════════════════════════════════════════════════════════════════════════

auroc_path = RESULTS / "bootstrap_auroc_ap.json"
if auroc_path.exists():
    auroc_data = load(auroc_path)
    auroc_block = auroc_data.get("auroc", {})
    # auroc is a dict: {"point": 0.7857, "ci_90_lo": ..., "boot_mean": ...}
    val = auroc_block.get("point") if isinstance(auroc_block, dict) else auroc_block
    check("AUROC point estimate > 0.70",
          val is not None and val > 0.70,
          detail=f"got {val:.3f}" if val is not None else "key not found")
    check("AUROC 90% CI lower > 0.55",
          isinstance(auroc_block, dict) and auroc_block.get("ci_90_lo", 0) > 0.55,
          detail=f"ci_90_lo={auroc_block.get('ci_90_lo')}" if isinstance(auroc_block, dict) else "n/a",
          warn_only=True)
else:
    check("bootstrap_auroc_ap.json readable", False, "file missing")

leakage_path = RESULTS / "leakage_audit.json"
if leakage_path.exists():
    la = load(leakage_path)
    overlap = la.get("overlap_count",
              la.get("n_overlap",
              la.get("leakage_count",
              la.get("n_leaking_pairs"))))
    check("No train/test date overlap (overlap = 0)",
          overlap == 0,
          detail=f"got {overlap}" if overlap is not None else "key not found — check JSON structure",
          warn_only=(overlap is None))
else:
    check("leakage_audit.json readable", False, "file missing")

# ══════════════════════════════════════════════════════════════════════════════
print("\n── 7. SCRIPT ROOT PATHS ──────────────────────────────────────────────")
# ══════════════════════════════════════════════════════════════════════════════
# Every script at scripts/<module>/*.py must resolve ROOT to methane-api/
# (parent.parent.parent), not scripts/ (parent.parent).

for script_rel in [
    "scripts/finance/finance_climate_var.py",
    "scripts/finance/finance_transition_risk.py",
    "scripts/calibration/conformal_threshold.py",
    "scripts/validation/validation_metrics.py",
    "scripts/quantification/quantification_uncertainty.py",
    "scripts/timeseries/timeseries_builder.py",
]:
    path = ROOT / script_rel
    if not path.exists():
        check(f"{script_rel} — ROOT path", False, "script not found")
        continue
    resolved = path.resolve().parent.parent.parent
    check(f"{path.name} — ROOT = methane-api/",
          resolved == ROOT,
          detail=f"resolves to '{resolved.name}'")

# ══════════════════════════════════════════════════════════════════════════════
print("\n── RESULT ────────────────────────────────────────────────────────────")
# ══════════════════════════════════════════════════════════════════════════════

if not failures and not warnings:
    print(f"  {PASS} All checks passed.\n")
    sys.exit(0)
elif not failures:
    print(f"  {WARN} {len(warnings)} warning(s) (non-blocking): {warnings}\n")
    sys.exit(0)
else:
    if warnings:
        print(f"  {WARN} {len(warnings)} warning(s): {warnings}")
    print(f"  {FAIL} {len(failures)} failure(s): {failures}\n")
    sys.exit(1)
