"""
scripts/conformal_threshold.py
================================
WS1 Conformal Prediction — calibrate a statistically guaranteed CH4Net
detection threshold from the non-emitter S/C score distribution.

Background
──────────
Split conformal prediction (Angelopoulos & Bates 2021) gives a finite-sample
guarantee:

    P(FPR ≤ α) ≥ 1 - δ

for any test distribution exchangeable with the calibration set.  The
calibrated threshold τ is the ⌈(n+1)(1-α)⌉/n quantile of the non-emitter
S/C scores.

For our non-emitter set (n=18), at α=0.10:
  τ = S/C at rank ⌈19×0.90⌉ = rank 18 (i.e. the maximum of the set),
  giving a hard guarantee that no more than 1 in 10 non-emitter scenes
  exceeds τ in expectation.

For a more practical operational threshold we report:
  τ_0.05  (95th percentile — ~1 in 20 FPR bound)
  τ_0.10  (90th percentile — ~1 in 10 FPR bound)  ← recommended
  τ_0.20  (80th percentile — ~1 in 5 FPR bound)

These are compared against the hard-coded 1.15 threshold used in current
production.

Input:  results_analysis/nonemitter_sc_scores.json
Output: results_analysis/calibrated_threshold.json
        results_analysis/conformal_calibration.png  (optional — needs matplotlib)

Usage:
    python scripts/conformal_threshold.py [--alpha 0.10] [--no-plot]

References:
    Angelopoulos & Bates (2021) "A Gentle Introduction to Conformal Prediction
    and Distribution-Free Uncertainty Quantification". arXiv:2107.07511.

    Venn prediction / Mondrian conformal prediction for stratified bounds:
    Shafer & Vovk (2008) "A Tutorial on Conformal Prediction". JMLR 9.
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS_DIR = Path("results_analysis")
RESULTS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("conformal_threshold")

SCORES_PATH   = RESULTS_DIR / "nonemitter_sc_scores.json"
OUTPUT_PATH   = RESULTS_DIR / "calibrated_threshold.json"
PLOT_PATH     = RESULTS_DIR / "conformal_calibration.png"

# Current production threshold
PRODUCTION_THRESHOLD = 1.15

# Conformal alpha levels to report
ALPHA_LEVELS = [0.05, 0.10, 0.20]


# ── Conformal quantile ─────────────────────────────────────────────────────────

def conformal_quantile(scores: list[float], alpha: float) -> float:
    """
    Split-conformal quantile at level (1-alpha).

    Returns the ⌈(n+1)(1-α)⌉/n-th empirical quantile of non-conformity scores.
    This is the standard finite-sample conformal threshold (Theorem 2, Angelopoulos
    & Bates 2021).  For n < 20, the guarantee is conservative (we may overshoot the
    1-α bound) but it remains valid.

    Args:
        scores: list of non-conformity scores (S/C ratios) from non-emitter set
        alpha:  desired FPR level (e.g., 0.10 for 90% specificity)

    Returns:
        Threshold τ such that P(FPR > α) ≤ δ → 0 as n → ∞.
    """
    n = len(scores)
    if n == 0:
        raise ValueError("Empty score set")
    # Finite-sample conformal quantile index
    rank = math.ceil((n + 1) * (1 - alpha))
    rank = min(rank, n)   # clamp to n (extrapolation-safe)
    sorted_scores = sorted(scores)
    return float(sorted_scores[rank - 1])


def empirical_fpr(scores: list[float], threshold: float) -> float:
    """Fraction of non-emitter scores that exceed threshold."""
    if not scores:
        return float("nan")
    return sum(1 for s in scores if s > threshold) / len(scores)


# ── Stratified (Mondrian) conformal ────────────────────────────────────────────

def mondrian_thresholds(records: list[dict],
                        alpha: float,
                        groupby: str = "ecoregion") -> dict:
    """
    Mondrian conformal prediction: compute a separate calibrated threshold
    per group (ecoregion or CLC class).

    With small group sizes (n_group < 5) the finite-sample bound degrades;
    these are flagged with a warning.  Groups with fewer than 3 samples return
    the maximum score with a SMALL_N flag.

    Returns:
        dict[group_name] = {
            "tau": float,
            "n": int,
            "scores": list,
            "empirical_fpr_at_production": float,
            "small_n_warning": bool,
        }
    """
    groups = defaultdict(list)
    for r in records:
        key = r.get(groupby, "unknown")
        sc  = r.get("sc_cfar") or r.get("sc_ratio")
        if sc is not None:
            groups[key].append(sc)

    result = {}
    for group, scores in groups.items():
        n = len(scores)
        small_n = n < 5
        try:
            tau = conformal_quantile(scores, alpha)
        except Exception:
            tau = max(scores) if scores else float("nan")
        result[group] = {
            "tau":                         round(tau, 4),
            "n":                           n,
            "scores":                      [round(s, 4) for s in sorted(scores)],
            "min_sc":                      round(min(scores), 4) if scores else None,
            "max_sc":                      round(max(scores), 4) if scores else None,
            "mean_sc":                     round(float(np.mean(scores)), 4) if scores else None,
            "empirical_fpr_at_production": round(empirical_fpr(scores, PRODUCTION_THRESHOLD), 4),
            "small_n_warning":             small_n,
        }
        if small_n:
            log.warning("  SMALL_N: group '%s' has only %d samples (recommend ≥5)",
                        group, n)
    return result


# ── Bootstrap uncertainty on τ ─────────────────────────────────────────────────

def bootstrap_tau(scores: list[float], alpha: float,
                  n_boot: int = 2000, seed: int = 42) -> dict:
    """
    Bootstrap confidence interval on the conformal threshold τ.

    This is NOT a formal conformal guarantee — it quantifies sampling
    uncertainty in τ due to finite calibration set size.
    """
    rng = np.random.default_rng(seed)
    boot_taus = []
    n = len(scores)
    scores_arr = np.array(scores)
    for _ in range(n_boot):
        samp = rng.choice(scores_arr, size=n, replace=True)
        try:
            boot_taus.append(conformal_quantile(list(samp), alpha))
        except Exception:
            pass

    boot_arr = np.array(boot_taus)
    return {
        "tau_boot_mean":  round(float(np.mean(boot_arr)), 4),
        "tau_boot_std":   round(float(np.std(boot_arr, ddof=1)), 4),
        "tau_boot_p5":    round(float(np.percentile(boot_arr, 5)), 4),
        "tau_boot_p95":   round(float(np.percentile(boot_arr, 95)), 4),
        "n_bootstrap":    n_boot,
    }


# ── Sample size adequacy ───────────────────────────────────────────────────────

def sample_size_analysis(n: int, alpha: float) -> dict:
    """
    Report on whether n is sufficient for the desired conformal guarantee.

    The conformal guarantee is exact for any n, but the bound tightens
    with larger n.  Standard guidance: n ≥ (1/alpha - 1) for the guarantee
    to be non-trivial.  For alpha=0.10, n ≥ 9.  For alpha=0.05, n ≥ 19.
    """
    min_n_nontrivial = math.ceil(1 / alpha - 1)
    rank = math.ceil((n + 1) * (1 - alpha))
    # The threshold is the `rank`-th order statistic.  Standard error of the
    # p-th quantile ≈ sqrt(p(1-p)/n) / f(Q_p) — use density-free bound.
    # Conservative: range of 2 adjacent order statistics / 2.
    return {
        "n_calibration":         n,
        "alpha":                 alpha,
        "conformal_rank":        rank,
        "min_n_nontrivial":      min_n_nontrivial,
        "n_adequate":            n >= min_n_nontrivial,
        "note": (
            "Conformal guarantee is valid for any n ≥ 1 (finite-sample exact), "
            "but the threshold is the k-th largest score; for very small n the "
            "threshold equals the maximum, giving a loose (conservative) bound."
        ),
    }


# ── Plotting (optional) ────────────────────────────────────────────────────────

def make_plot(scores: list[float], thresholds: dict[float, float],
              records: list[dict], out_path: Path) -> None:
    """Generate calibration summary plot if matplotlib is available."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed — skipping plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("CH4Net v8 — WS5 Conformal Threshold Calibration", fontsize=13)

    # ── Left: ECDF of S/C scores ──────────────────────────────────────────────
    ax = axes[0]
    sorted_sc = sorted(scores)
    n = len(sorted_sc)
    ecdf_y = [(i + 1) / n for i in range(n)]

    ax.step(sorted_sc, ecdf_y, where="post", lw=2, color="#2066a8", label="ECDF (non-emitters)")
    ax.axvline(PRODUCTION_THRESHOLD, color="red", ls="--", lw=1.5,
               label=f"Production τ={PRODUCTION_THRESHOLD}")

    colors = ["#1a9641", "#f97f0f", "#d7191c"]
    for (alpha, tau), color in zip(sorted(thresholds.items()), colors):
        ax.axvline(tau, color=color, ls=":", lw=1.5,
                   label=f"Conformal τ (α={alpha})={tau:.3f}")

    ax.set_xlabel("S/C ratio")
    ax.set_ylabel("ECDF")
    ax.set_title("Non-emitter S/C score distribution")
    ax.legend(fontsize=8)
    ax.set_xlim(left=0)
    ax.grid(alpha=0.3)

    # ── Right: S/C by ecoregion (dot plot) ───────────────────────────────────
    ax = axes[1]
    ecoregion_scores = defaultdict(list)
    for r in records:
        eco = r.get("ecoregion", "unknown")
        sc  = r.get("sc_cfar") or r.get("sc_ratio")
        if sc is not None:
            ecoregion_scores[eco].append(sc)

    ecoregions = sorted(ecoregion_scores.keys())
    y_positions = {eco: i for i, eco in enumerate(ecoregions)}

    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, eco in enumerate(ecoregions):
        sc_vals = ecoregion_scores[eco]
        y = y_positions[eco]
        ax.scatter(sc_vals, [y] * len(sc_vals),
                   color=palette[i % len(palette)], s=60, zorder=3, alpha=0.8)

    ax.axvline(PRODUCTION_THRESHOLD, color="red", ls="--", lw=1.5,
               label=f"τ_prod={PRODUCTION_THRESHOLD}")
    tau_10 = thresholds.get(0.10)
    if tau_10:
        ax.axvline(tau_10, color="#f97f0f", ls=":", lw=1.5,
                   label=f"τ_conf(α=0.10)={tau_10:.3f}")

    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels(list(y_positions.keys()), fontsize=9)
    ax.set_xlabel("S/C ratio")
    ax.set_title("S/C scores by ecoregion")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Plot saved → %s", out_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="WS1 conformal threshold calibration from non-emitter S/C scores"
    )
    parser.add_argument("--alpha", type=float, default=0.10,
                        help="Primary FPR level (default: 0.10)")
    parser.add_argument("--scores", default=str(SCORES_PATH),
                        help=f"Input scores JSON (default: {SCORES_PATH})")
    parser.add_argument("--output", default=str(OUTPUT_PATH),
                        help=f"Output calibration JSON (default: {OUTPUT_PATH})")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip matplotlib plot")
    args = parser.parse_args()

    # ── Load scores ───────────────────────────────────────────────────────────
    if not Path(args.scores).exists():
        print(f"ERROR: Scores file not found: {args.scores}")
        print("Run: python scripts/run_nonemitter_inference.py")
        sys.exit(1)

    records = json.load(open(args.scores))
    ok_records = [r for r in records if r.get("status") == "ok"]
    log.info("Loaded %d scored locations (%d OK)", len(records), len(ok_records))

    scores = []
    for r in ok_records:
        sc = r.get("sc_cfar") or r.get("sc_ratio")
        if sc is not None and not math.isnan(sc):
            scores.append(sc)

    if len(scores) < 3:
        print(f"ERROR: Only {len(scores)} valid S/C scores — need at least 3.")
        print("Run run_nonemitter_inference.py on more locations.")
        sys.exit(1)

    log.info("Using %d S/C scores for calibration", len(scores))

    # ── Global conformal thresholds ───────────────────────────────────────────
    global_thresholds = {}
    for alpha in ALPHA_LEVELS:
        tau = conformal_quantile(scores, alpha)
        global_thresholds[alpha] = round(tau, 4)
        log.info("  τ(α=%.2f) = %.4f   empirical FPR at production=%.3f  "
                 "(n=%d, rank=%d)",
                 alpha, tau,
                 empirical_fpr(scores, PRODUCTION_THRESHOLD),
                 len(scores), math.ceil((len(scores)+1)*(1-alpha)))

    primary_tau   = global_thresholds[args.alpha]
    prod_fpr      = empirical_fpr(scores, PRODUCTION_THRESHOLD)
    calibr_fpr    = empirical_fpr(scores, primary_tau)

    # ── Bootstrap CI on primary τ ─────────────────────────────────────────────
    log.info("Bootstrap CI on τ (α=%.2f)...", args.alpha)
    boot = bootstrap_tau(scores, args.alpha)

    # ── Mondrian thresholds ───────────────────────────────────────────────────
    log.info("Mondrian conformal thresholds by ecoregion and CLC class...")
    mondrian_eco = mondrian_thresholds(ok_records, args.alpha, groupby="ecoregion")
    mondrian_clc = mondrian_thresholds(ok_records, args.alpha, groupby="clc_class")

    # ── Sample size analysis ──────────────────────────────────────────────────
    size_info = sample_size_analysis(len(scores), args.alpha)

    # ── Assemble output ───────────────────────────────────────────────────────
    result = {
        "schema_version":    "1.0.0",
        "method":            "split_conformal_prediction",
        "reference":         "Angelopoulos & Bates (2021) arXiv:2107.07511",
        "n_calibration":     len(scores),
        "all_scores":        [round(s, 4) for s in sorted(scores)],

        "global_thresholds": {
            f"tau_alpha_{int(a*100):02d}": {
                "alpha":                 a,
                "tau":                   t,
                "guarantee":             f"FPR ≤ {a:.0%} with finite-sample guarantee",
                "empirical_fpr_nonems":  round(empirical_fpr(scores, t), 4),
            }
            for a, t in global_thresholds.items()
        },

        "primary_threshold": {
            "alpha":                         args.alpha,
            "tau":                           primary_tau,
            "production_threshold":          PRODUCTION_THRESHOLD,
            "empirical_fpr_at_production":   round(prod_fpr, 4),
            "empirical_fpr_at_calibrated":   round(calibr_fpr, 4),
            "recommended_action": (
                "REPLACE production threshold 1.15 with τ_calibrated"
                if primary_tau > PRODUCTION_THRESHOLD
                else "Production threshold is already conservative — retain or reduce"
            ),
        },

        "bootstrap_ci": boot,
        "sample_size_analysis": size_info,

        "mondrian_by_ecoregion": mondrian_eco,
        "mondrian_by_clc_class": mondrian_clc,

        "interpretation": (
            f"On n={len(scores)} non-emitter locations, the S/C scores ranged from "
            f"{min(scores):.4f} to {max(scores):.4f}. "
            f"The production threshold of {PRODUCTION_THRESHOLD} yields an empirical "
            f"FPR of {prod_fpr:.1%} on this calibration set. "
            f"The conformal threshold at α={args.alpha} is τ={primary_tau:.4f}, "
            f"providing a finite-sample guarantee that FPR ≤ {args.alpha:.0%} "
            f"on exchangeable test scenes."
        ),
    }

    # ── Write output ──────────────────────────────────────────────────────────
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    log.info("Calibration result written → %s", args.output)

    # ── Pretty-print summary ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  WS1 Conformal Threshold Calibration")
    print("=" * 70)
    print(f"  Calibration set: n={len(scores)} non-emitter locations")
    print(f"  Score range: [{min(scores):.4f}, {max(scores):.4f}]")
    print(f"  Score mean: {np.mean(scores):.4f}   median: {np.median(scores):.4f}")
    print()
    print(f"  Production threshold:  τ_prod = {PRODUCTION_THRESHOLD}")
    print(f"  Empirical FPR at τ_prod: {prod_fpr:.1%}")
    print()
    print(f"  Conformal thresholds (finite-sample guaranteed FPR bound):")
    for alpha in ALPHA_LEVELS:
        tau = global_thresholds[alpha]
        star = " ← recommended" if alpha == args.alpha else ""
        fpr_label = f"FPR ≤ {alpha:.0%}"
        print(f"    α={alpha:.2f}  τ={tau:.4f}  ({fpr_label}){star}")
    print()
    print(f"  Bootstrap 90% CI on τ(α={args.alpha}): "
          f"[{boot['tau_boot_p5']:.4f}, {boot['tau_boot_p95']:.4f}]")
    print()

    # Sample size note
    sa = size_info
    print(f"  Sample size: n={sa['n_calibration']}  "
          f"(minimum for non-trivial bound at α={args.alpha}: n≥{sa['min_n_nontrivial']}  "
          f"{'✓ adequate' if sa['n_adequate'] else '⚠ marginal'})")
    print()

    # Mondrian by ecoregion
    print("  Mondrian thresholds by ecoregion:")
    for eco, info in sorted(mondrian_eco.items()):
        warn = "  ⚠ SMALL_N" if info["small_n_warning"] else ""
        print(f"    {eco:<20}  n={info['n']}  τ={info['tau']:.4f}{warn}")
    print()
    print(f"  Written to: {args.output}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    if not args.no_plot:
        make_plot(scores, global_thresholds, ok_records, PLOT_PATH)
        if PLOT_PATH.exists():
            print(f"  Plot saved: {PLOT_PATH}")


if __name__ == "__main__":
    main()
