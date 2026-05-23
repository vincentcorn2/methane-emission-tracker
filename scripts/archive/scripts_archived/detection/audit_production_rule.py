"""
scripts/audit_production_rule.py
==================================
Apply the EXPLICIT production detection rule to every record in the candidate
backfill and the Belchatow time series, and produce a single source-of-truth
table that the report cites everywhere.

Why this exists
---------------
The current draft has internal inconsistency. Section 3.2 lists Belchatow
2024-08-24 (S/C=27.30) as a detection that "clears global conformal threshold τ
but not the per-scene CFAR margin." Section 3.4 says Boxberg's 1202.83 spike
is "flagged as terrain artifact, manual audit" — but the conformal rule alone
would flag it as the largest detection in the candidate set. A sophisticated
reviewer reads both passages and asks "what is the production detection rule,
exactly?" This script answers that question with a definitive table.

Production detection rule (single source of truth)
--------------------------------------------------
A record is classified as a DETECTION if and only if ALL of the following hold:

  (1) status == "ok" (record is a valid observation, not a partial-swath
      degeneracy or a download failure)
  (2) sc_cfar > τ (the conformal threshold from results_analysis/
      calibrated_threshold.json, currently τ = 3.5796 at α = 0.10)
  (3) cfar_detect == True (the per-scene CFAR margin is positive — the site
      mean exceeds the local heterogeneity-adjusted threshold)

Any record that fails (2) or (3) is NOT a detection. We classify it as:

  - SUB_TAU         : sc_cfar ≤ τ (below the conformal threshold)
  - CFAR_SUPPRESSED : sc_cfar > τ AND cfar_detect == False
  - NO_COVERAGE     : partial-swath or invalid-status record
  - NOT_OK          : status set to an error code

Outputs
-------
results_analysis/production_rule_audit.json  — full per-record table
results_analysis/production_rule_audit.md    — report-ready markdown summary

Usage
-----
  python scripts/audit_production_rule.py
  python scripts/audit_production_rule.py --tau 3.5796   # override threshold (reads from JSON by default)
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results_analysis"
CONFORMAL_THRESHOLD_JSON = OUT_DIR / "calibrated_threshold.json"
BACKFILL_JSON = OUT_DIR / "historical_backfill_timeseries.json"
BELCHATOW_TS_JSON = OUT_DIR / "belchatow_annual_timeseries.json"


CLASSIFICATION_DESCRIPTION = {
    "DETECTION":
        "passes the full production rule: status=ok AND sc_cfar > τ AND cfar_detect=True",
    "CFAR_SUPPRESSED":
        "sc_cfar > τ but cfar_detect=False — high control heterogeneity inflated the local "
        "CFAR threshold above the observed S/C. NOT a production detection.",
    "SUB_TAU":
        "sc_cfar ≤ τ — does not clear the conformal threshold. NOT a detection.",
    "NO_COVERAGE":
        "partial-swath fingerprint (sc_ratio = 1.0 exactly with cv_ctrl = 0) or "
        "missing inference. Excluded as a missing observation rather than a non-detection.",
    "NOT_OK":
        "record status flagged as an error (download failure, no_geo_meta, etc.).",
}


def load_tau(default=3.5796):
    if CONFORMAL_THRESHOLD_JSON.exists():
        d = json.loads(CONFORMAL_THRESHOLD_JSON.read_text())
        # Try multiple schema versions
        global_thresholds = d.get("global_thresholds") or {}
        alpha_10 = global_thresholds.get("tau_alpha_10") or {}
        tau = alpha_10.get("tau")
        if tau:
            return float(tau)
    return float(default)


def is_partial_swath(rec):
    sc = rec.get("sc_ratio")
    cv = rec.get("cv_ctrl")
    sm = rec.get("site_mean")
    cm = rec.get("ctrl_mean")
    if sc is None:
        return False
    if abs(sc - 1.0) < 1e-6 and (cv is None or abs(cv) < 1e-6):
        return True
    if sm is not None and cm is not None and abs(sm - cm) < 1e-6:
        return True
    return False


def classify(rec, tau):
    status = rec.get("status")
    if status not in (None, "ok"):
        return "NOT_OK"
    if is_partial_swath(rec):
        return "NO_COVERAGE"

    sc_cfar = rec.get("sc_cfar") if rec.get("sc_cfar") is not None else rec.get("sc_ratio")
    if sc_cfar is None:
        return "NO_COVERAGE"

    if sc_cfar <= tau:
        return "SUB_TAU"

    # Above tau — now check CFAR
    cfar = rec.get("cfar_detect")
    if cfar is True:
        return "DETECTION"
    return "CFAR_SUPPRESSED"


def normalise_backfill_record(site, rec):
    """Bring a backfill record into a flat dict with date / sc_ratio / sc_cfar / cv_ctrl / cfar_detect / status."""
    return {
        "site":         site,
        "date":         rec.get("date") or rec.get("acquisition_date"),
        "tile":         rec.get("tile_id"),
        "sc_ratio":     rec.get("sc_ratio"),
        "sc_cfar":      rec.get("sc_cfar"),
        "cv_ctrl":      rec.get("cv_ctrl"),
        "cfar_detect":  rec.get("cfar_detect"),
        "cfar_margin":  rec.get("cfar_margin"),
        "site_mean":    rec.get("site_mean"),
        "ctrl_mean":    rec.get("ctrl_mean"),
        "status":       rec.get("status"),
        "_source":      "historical_backfill",
    }


def normalise_timeseries_record(rec):
    """Bring a max-data time series record into the same flat dict."""
    det = rec.get("detection") or {}
    quant = rec.get("quantification") or {}
    return {
        "site":         "belchatow",
        "date":         rec.get("month") or rec.get("acquisition_date"),
        "tile":         "T34UCB",
        "sc_ratio":     det.get("sc_ratio"),
        "sc_cfar":      det.get("sc_cfar"),
        "cv_ctrl":      det.get("cv_ctrl"),
        "cfar_detect":  det.get("cfar_detect"),
        "cfar_margin":  det.get("cfar_margin"),
        "site_mean":    det.get("site_mean"),
        "ctrl_mean":    det.get("ctrl_mean"),
        "status":       "ok" if det.get("sc_ratio") is not None else None,
        "Q_kgh":        quant.get("flow_rate_kgh"),
        "_source":      "belchatow_annual_timeseries",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau", type=float, default=None,
                        help="Override the conformal threshold (default: read from "
                             "calibrated_threshold.json)")
    args = parser.parse_args()

    tau = args.tau if args.tau is not None else load_tau()
    print(f"Conformal threshold τ = {tau}")
    print()

    records = []

    # Load backfill
    if BACKFILL_JSON.exists():
        bf = json.loads(BACKFILL_JSON.read_text())
        if isinstance(bf, dict):
            for site, site_records in bf.items():
                if not isinstance(site_records, list):
                    continue
                for r in site_records:
                    records.append(normalise_backfill_record(site.lower(), r))
        else:
            print(f"WARNING: backfill at {BACKFILL_JSON} not a dict; skipping")

    # Load max-data Belchatow time series
    if BELCHATOW_TS_JSON.exists():
        ts = json.loads(BELCHATOW_TS_JSON.read_text())
        ts_records = ts.get("records", []) if isinstance(ts, dict) else ts
        for r in ts_records:
            records.append(normalise_timeseries_record(r))

    if not records:
        print("No records found. Run the backfill and time series first.")
        return

    # Classify
    for r in records:
        r["classification"] = classify(r, tau)

    # Summarise
    by_site_class = defaultdict(lambda: defaultdict(int))
    for r in records:
        site = r["site"] or "unknown"
        by_site_class[site][r["classification"]] += 1

    classes = ["DETECTION", "CFAR_SUPPRESSED", "SUB_TAU", "NO_COVERAGE", "NOT_OK"]

    md = []
    md.append("# Production detection rule audit\n")
    md.append("**Production rule (single source of truth):**  \n")
    md.append("A record is a DETECTION iff status=`ok` AND `sc_cfar > τ` AND `cfar_detect = True`.\n")
    md.append(f"- τ (α=0.10) = **{tau}** (read from `results_analysis/calibrated_threshold.json`)")
    md.append("- All other above-threshold outcomes are classified as below.\n")
    md.append("## Classification key\n")
    for cls, desc in CLASSIFICATION_DESCRIPTION.items():
        md.append(f"- **`{cls}`** — {desc}")
    md.append("")

    md.append("## Per-site outcome counts\n")
    md.append("| Site | DETECTION | CFAR_SUPPRESSED | SUB_TAU | NO_COVERAGE | NOT_OK | Total |")
    md.append("|---|---|---|---|---|---|---|")
    for site in sorted(by_site_class):
        row = by_site_class[site]
        total = sum(row.values())
        md.append(
            f"| {site} | {row.get('DETECTION', 0)} | "
            f"{row.get('CFAR_SUPPRESSED', 0)} | {row.get('SUB_TAU', 0)} | "
            f"{row.get('NO_COVERAGE', 0)} | {row.get('NOT_OK', 0)} | {total} |"
        )
    md.append("")

    # Per-site detection detail
    md.append("## Detection-grade records (the only records that count as 'detections')\n")
    md.append("| Site | Date | Tile | sc_cfar | cv_ctrl | cfar_margin | source |")
    md.append("|---|---|---|---|---|---|---|")
    for r in records:
        if r["classification"] == "DETECTION":
            md.append(
                f"| {r['site']} | {r['date']} | {r.get('tile') or '—'} | "
                f"{r.get('sc_cfar') or r.get('sc_ratio')} | "
                f"{r.get('cv_ctrl')} | {r.get('cfar_margin')} | "
                f"{r.get('_source')} |"
            )
    md.append("")

    md.append("## CFAR-suppressed records (above τ but failing CFAR — NOT detections)\n")
    md.append("These records would be falsely flagged as detections under a τ-only rule. The "
              "production rule's CFAR gate correctly rejects them.\n")
    md.append("| Site | Date | Tile | sc_cfar | cv_ctrl | cfar_margin | source |")
    md.append("|---|---|---|---|---|---|---|")
    for r in records:
        if r["classification"] == "CFAR_SUPPRESSED":
            md.append(
                f"| {r['site']} | {r['date']} | {r.get('tile') or '—'} | "
                f"{r.get('sc_cfar') or r.get('sc_ratio')} | "
                f"{r.get('cv_ctrl')} | {r.get('cfar_margin')} | "
                f"{r.get('_source')} |"
            )
    md.append("")

    # Save outputs
    out_md = OUT_DIR / "production_rule_audit.md"
    out_json = OUT_DIR / "production_rule_audit.json"
    out_md.write_text("\n".join(md))
    out_json.write_text(json.dumps({
        "tau": tau,
        "records": records,
        "by_site_class": {k: dict(v) for k, v in by_site_class.items()},
    }, indent=2, default=str))

    # Console output
    print("=" * 78)
    print("PRODUCTION RULE AUDIT  (τ =", tau, ")")
    print("=" * 78)
    print(f"{'Site':<14}{'DET':>5}{'CFAR_SUPP':>11}{'SUB_TAU':>9}{'NO_COV':>8}{'NOT_OK':>8}{'TOTAL':>7}")
    print("-" * 78)
    for site in sorted(by_site_class):
        row = by_site_class[site]
        total = sum(row.values())
        print(f"{site:<14}{row.get('DETECTION', 0):>5}{row.get('CFAR_SUPPRESSED', 0):>11}"
              f"{row.get('SUB_TAU', 0):>9}{row.get('NO_COVERAGE', 0):>8}"
              f"{row.get('NOT_OK', 0):>8}{total:>7}")
    print()

    n_det = sum(1 for r in records if r["classification"] == "DETECTION")
    n_supp = sum(1 for r in records if r["classification"] == "CFAR_SUPPRESSED")
    print(f"Total DETECTIONS across all sites:        {n_det}")
    print(f"Total CFAR_SUPPRESSED (rescued by CFAR):  {n_supp}")
    print()
    print(f"Wrote: {out_md}")
    print(f"Wrote: {out_json}")
    print()
    print("If any 'detection' currently cited in the report does not appear in the")
    print("DETECTION table above, the report needs to be reconciled before submission.")


if __name__ == "__main__":
    main()
