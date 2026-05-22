"""
scripts/bootstrap_auroc_ap.py
==============================
Bootstrap confidence intervals on scene-level AUROC and Average Precision,
restricted to the 14 real-positive crops and 22 real-negative crops only
(synthetics excluded — addresses Reviewer 2's concern about tiny real dataset).

Methodology:
  - Score: prob_mean (scene-level mean CH4Net probability)
  - Label: 1 = confirmed methane, 0 = verified non-emitter
  - Resampling: stratified bootstrap (sample n_pos with replacement from
    positives, n_neg with replacement from negatives) to preserve class ratio
  - n_bootstrap: 2000
  - Degenerate resamples (all-same-class) are skipped, not counted

Outputs:
  results_analysis/bootstrap_auroc_ap.json  — full bootstrap distribution
  (also prints a summary table)

Usage:
    cd ~/Downloads/methane-api
    conda activate methane
    python scripts/bootstrap_auroc_ap.py
"""

import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent.parent
METRICS_PATH = ROOT / "results_analysis" / "ml_metrics.json"
OUT_PATH = ROOT / "results_analysis" / "bootstrap_auroc_ap.json"

N_BOOTSTRAP = 2000
RANDOM_SEED = 42


# ── sklearn-free AUROC and AP ─────────────────────────────────────────────────

def auroc(labels, scores):
    """
    Compute AUROC via trapezoidal rule over the ROC curve.
    Returns NaN if only one class present.
    """
    pos = [(s, l) for s, l in zip(scores, labels) if l == 1]
    neg = [(s, l) for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    n_pos, n_neg = len(pos), len(neg)
    # Wilcoxon-Mann-Whitney statistic = AUROC
    n_concordant = sum(1 for p in pos for n in neg if p[0] > n[0])
    n_tied = sum(0.5 for p in pos for n in neg if p[0] == n[0])
    return (n_concordant + n_tied) / (n_pos * n_neg)


def average_precision(labels, scores):
    """
    Compute Average Precision (area under precision-recall curve).
    Returns NaN if no positives.
    """
    if sum(labels) == 0:
        return float("nan")
    # Sort by descending score
    paired = sorted(zip(scores, labels), reverse=True)
    tp = fp = 0
    prev_recall = 0.0
    ap = 0.0
    n_pos = sum(labels)
    for _, label in paired:
        if label == 1:
            tp += 1
        else:
            fp += 1
        precision = tp / (tp + fp)
        recall = tp / n_pos
        ap += precision * (recall - prev_recall)
        prev_recall = recall
    return ap


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rng = random.Random(RANDOM_SEED)
    np_rng = np.random.default_rng(RANDOM_SEED)

    with open(METRICS_PATH) as f:
        data = json.load(f)

    crops = data["per_crop"]

    # Filter to real crops only (exclude synthetic)
    real_pos = [c for c in crops if c["source"] == "positive"]
    real_neg = [c for c in crops if c["source"] == "negative"]

    n_pos = len(real_pos)
    n_neg = len(real_neg)

    print(f"Real positives: {n_pos}")
    print(f"Real negatives: {n_neg}")
    print(f"Total real crops: {n_pos + n_neg}")
    print(f"Bootstrap iterations: {N_BOOTSTRAP}")
    print()

    # Point estimates on the real-only subset
    all_real = real_pos + real_neg
    scores_real = [c["prob_mean"] for c in all_real]
    labels_real = [c["label"] for c in all_real]

    point_auroc = auroc(labels_real, scores_real)
    point_ap = average_precision(labels_real, scores_real)

    print(f"Point estimates (real crops only, n={n_pos + n_neg}):")
    print(f"  AUROC: {point_auroc:.4f}")
    print(f"  AP:    {point_ap:.4f}")
    print()

    # Bootstrap — stratified: resample within each class separately
    boot_auroc = []
    boot_ap = []
    skipped = 0

    pos_scores = [c["prob_mean"] for c in real_pos]
    pos_labels = [1] * n_pos
    neg_scores = [c["prob_mean"] for c in real_neg]
    neg_labels = [0] * n_neg

    for _ in range(N_BOOTSTRAP):
        # Sample with replacement within each class
        idx_pos = np_rng.integers(0, n_pos, size=n_pos)
        idx_neg = np_rng.integers(0, n_neg, size=n_neg)

        s = [pos_scores[i] for i in idx_pos] + [neg_scores[i] for i in idx_neg]
        l = [1] * n_pos + [0] * n_neg  # labels preserved by construction

        a = auroc(l, s)
        p = average_precision(l, s)

        if a != a or p != p:  # NaN check — degenerate resample
            skipped += 1
            continue

        boot_auroc.append(a)
        boot_ap.append(p)

    n_valid = len(boot_auroc)
    print(f"Valid bootstrap resamples: {n_valid} / {N_BOOTSTRAP}  (skipped: {skipped})")
    print()

    # CIs
    def ci(values, level):
        alpha = (1 - level) / 2
        lo = float(np.quantile(values, alpha))
        hi = float(np.quantile(values, 1 - alpha))
        return lo, hi

    auroc_90 = ci(boot_auroc, 0.90)
    auroc_95 = ci(boot_auroc, 0.95)
    ap_90    = ci(boot_ap,    0.90)
    ap_95    = ci(boot_ap,    0.95)

    print("Bootstrap confidence intervals (stratified, 2000 resamples):")
    print(f"  AUROC  point = {point_auroc:.4f}")
    print(f"         90% CI: [{auroc_90[0]:.4f}, {auroc_90[1]:.4f}]")
    print(f"         95% CI: [{auroc_95[0]:.4f}, {auroc_95[1]:.4f}]")
    print()
    print(f"  AP     point = {point_ap:.4f}")
    print(f"         90% CI: [{ap_90[0]:.4f}, {ap_90[1]:.4f}]")
    print(f"         95% CI: [{ap_95[0]:.4f}, {ap_95[1]:.4f}]")
    print()

    # Save
    result = {
        "method": "stratified_bootstrap",
        "n_bootstrap": N_BOOTSTRAP,
        "n_valid": n_valid,
        "n_skipped": skipped,
        "n_real_positives": n_pos,
        "n_real_negatives": n_neg,
        "score_field": "prob_mean",
        "note": (
            "Restricted to 14 TROPOMI-confirmed real positive crops and 22 "
            "verified-negative crops. Synthetic crops excluded. Stratified "
            "bootstrap: resampling within each class preserves class ratio. "
            "Addresses Reviewer 2 concern: 'effective positive real dataset "
            "is still tiny' — reporting CIs rather than point estimates only."
        ),
        "point_estimates": {
            "auroc": round(point_auroc, 6),
            "average_precision": round(point_ap, 6),
        },
        "auroc": {
            "point": round(point_auroc, 4),
            "ci_90_lo": round(auroc_90[0], 4),
            "ci_90_hi": round(auroc_90[1], 4),
            "ci_95_lo": round(auroc_95[0], 4),
            "ci_95_hi": round(auroc_95[1], 4),
            "boot_mean": round(float(np.mean(boot_auroc)), 4),
            "boot_std":  round(float(np.std(boot_auroc)),  4),
        },
        "average_precision": {
            "point": round(point_ap, 4),
            "ci_90_lo": round(ap_90[0], 4),
            "ci_90_hi": round(ap_90[1], 4),
            "ci_95_lo": round(ap_95[0], 4),
            "ci_95_hi": round(ap_95[1], 4),
            "boot_mean": round(float(np.mean(boot_ap)), 4),
            "boot_std":  round(float(np.std(boot_ap)),  4),
        },
        "distribution": {
            "auroc_p5":  round(float(np.percentile(boot_auroc,  5)), 4),
            "auroc_p25": round(float(np.percentile(boot_auroc, 25)), 4),
            "auroc_p50": round(float(np.percentile(boot_auroc, 50)), 4),
            "auroc_p75": round(float(np.percentile(boot_auroc, 75)), 4),
            "auroc_p95": round(float(np.percentile(boot_auroc, 95)), 4),
            "ap_p5":     round(float(np.percentile(boot_ap,  5)), 4),
            "ap_p25":    round(float(np.percentile(boot_ap, 25)), 4),
            "ap_p50":    round(float(np.percentile(boot_ap, 50)), 4),
            "ap_p75":    round(float(np.percentile(boot_ap, 75)), 4),
            "ap_p95":    round(float(np.percentile(boot_ap, 95)), 4),
        },
    }

    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Results saved → {OUT_PATH.relative_to(ROOT)}")
    print()
    print("For paper: update ml_metrics table to read:")
    print(f"  AUROC {point_auroc:.3f} (90% bootstrap CI [{auroc_90[0]:.3f}, {auroc_90[1]:.3f}])")
    print(f"  AP    {point_ap:.3f}   (90% bootstrap CI [{ap_90[0]:.3f}, {ap_90[1]:.3f}])")
    print("  (real crops only, n=14 positives + n=22 negatives, stratified bootstrap n=2000)")


if __name__ == "__main__":
    main()
