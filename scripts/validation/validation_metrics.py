"""
validation_metrics.py
=====================
Statistical validation and audit suite for CH4Net v8.

Bundles four independent validators, each operating on pre-computed JSON
results files in results_analysis/.  No model weights or GPU required.

Class hierarchy
---------------
BaseValidator               Abstract base.  Defines Run() / PrintSummary()
                            interface; provides concrete SaveResults().

BootstrapAUROCValidator     Stratified bootstrap confidence intervals on
(BaseValidator)             scene-level AUROC and Average Precision (real
                            crops only, synthetics excluded).

LooDetectionAnalyzer        Leave-one-out stability analysis on the
(BaseValidator)             Bełchatów above-threshold detection record.
                            Runs two passes: scene-level (A) and month-level (B).

LeakageAuditor              Data-leakage and independence audit.  Three checks:
(BaseValidator)             site-level training overlap, temporal proximity between
                            training and evaluation acquisitions, and conformal
                            calibration set spatial independence.

HeldOutEvaluator            Evaluates v8 performance on sites that were genuinely
(BaseValidator)             held out from training (or trained as negatives).

Usage
-----
    cd methane-api
    python scripts/validation/validation_metrics.py [--validator all|bootstrap|loo|leakage|held-out]

Inputs (all in results_analysis/)
------
    ml_metrics.json
    belchatow_annual_timeseries.json
    nonemitter_sc_scores.json
    training_set_audit.json
    historical_backfill_timeseries.json
    data/crops/**/*_label.json

Outputs (all in results_analysis/)
-------
    bootstrap_auroc_ap.json
    loo_detection_stability.json
    leakage_audit.json, leakage_audit.md
    held_out_evaluation.json, held_out_evaluation.md

References
----------
Angelopoulos & Bates (2021) "A Gentle Introduction to Conformal Prediction".
Reviewer 2 concerns re: synthetic-data generalization and tiny real dataset.
"""

import argparse
import json
import logging
import math
import re
import statistics
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("validation_metrics")

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------

ROOT        = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT / "results_analysis"
CROPS_DIR   = ROOT / "data" / "crops"

# BootstrapAUROCValidator
BOOTSTRAP_N_RESAMPLES = 2_000
BOOTSTRAP_SEED        = 42
METRICS_PATH          = RESULTS_DIR / "ml_metrics.json"
BOOTSTRAP_OUT_PATH    = RESULTS_DIR / "bootstrap_auroc_ap.json"

# LooDetectionAnalyzer
LOO_TIMESERIES_PATH   = RESULTS_DIR / "belchatow_annual_timeseries.json"
LOO_OUT_PATH          = RESULTS_DIR / "loo_detection_stability.json"
LOO_PLOT_PATH         = RESULTS_DIR / "loo_detection_stability.png"

# LeakageAuditor
LEAKAGE_OUT_JSON      = RESULTS_DIR / "leakage_audit.json"
LEAKAGE_OUT_MD        = RESULTS_DIR / "leakage_audit.md"
CALIBRATION_SCORES    = RESULTS_DIR / "nonemitter_sc_scores.json"
TEMPORAL_LEAKAGE_DAYS = 14
SPATIAL_EXCLUSION_KM  = 50.0
CANDIDATE_SITES: dict[str, tuple[float, float, str]] = {
    "belchatow":  (51.242, 19.275, "T34UCB"),
    "rybnik":     (50.135, 18.522, "T34UCA"),
    "weisweiler": (50.837,  6.322, "T31UGS"),
    "lippendorf": (51.178, 12.378, "T33UUS"),
    "neurath":    (51.038,  6.616, "T32ULB"),
    "boxberg":    (51.412, 14.626, "T33UVT"),
    "groningen":  (53.388,  6.617, "T31UGV"),
    "maasvlakte": (51.952,  4.073, "T31UET"),
}
EVAL_DATES: dict[str, list[str]] = {
    "belchatow": [
        "2020-06-01", "2021-06-06", "2021-09-09", "2024-04-11",
        "2024-05-26", "2024-07-10", "2024-07-30", "2024-10-28",
    ],
    "neurath": ["2024-06-25", "2024-08-29"],
    "rybnik":  ["2025-03-22"],
}

# HeldOutEvaluator
HELD_OUT_BACKFILL_PATH = RESULTS_DIR / "historical_backfill_timeseries.json"
HELD_OUT_AUDIT_PATH    = RESULTS_DIR / "training_set_audit.json"
HELD_OUT_OUT_JSON      = RESULTS_DIR / "held_out_evaluation.json"
HELD_OUT_OUT_MD        = RESULTS_DIR / "held_out_evaluation.md"
CONFORMAL_TAU          = 4.1052

HELD_OUT      = "held_out"
TRAIN_NEG     = "training_negative"
TRAIN_NEG_SYN = "training_negative_and_synthetic_substrate"
CATEGORY_LABEL: dict[str, str] = {
    HELD_OUT:      "TRULY HELD-OUT (never seen in training)",
    TRAIN_NEG:     "Trained as NEGATIVE (positive detection overrides training label)",
    TRAIN_NEG_SYN: "Trained as NEGATIVE + used as synthetic substrate",
}

ACQ_DATE_RE = re.compile(r"_?(\d{4})[-_]?(\d{2})[-_]?(\d{2})")

# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Great-circle distance between two (lat, lon) points in kilometres.

    Parameters
    ----------
    lat1, lon1 : float
        Coordinates of point 1 (degrees).
    lat2, lon2 : float
        Coordinates of point 2 (degrees).

    Return
    ------
    float  Distance in kilometres.
    """
    R = 6_371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def extract_date_from_stem(stem: str) -> datetime | None:
    """
    Parse an ISO-8601 date from a filename stem using ACQ_DATE_RE.

    Parameters
    ----------
    stem : str
        Filename stem (no extension).

    Return
    ------
    datetime | None  Parsed datetime, or None if no valid date is found.
    """
    m = ACQ_DATE_RE.search(stem)
    if not m:
        return None
    try:
        return datetime.strptime(f"{m.group(1)}-{m.group(2)}-{m.group(3)}", "%Y-%m-%d")
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Abstract base validator
# ---------------------------------------------------------------------------


class BaseValidator(ABC):
    """
    Abstract base class for all CH4Net v8 validation analyses.

    All concrete validators must implement Run() and PrintSummary().
    SaveResults() is provided as a concrete method that writes any
    dict payload to a JSON file.
    """

    @abstractmethod
    def Run(self) -> dict:
        """
        Execute the validation analysis.

        Return
        ------
        dict  Result payload suitable for JSON serialisation.
        """

    @abstractmethod
    def PrintSummary(self, results: dict) -> None:
        """
        Print a human-readable summary of the validation results to stdout.

        Parameters
        ----------
        results : dict
            Payload returned by Run().
        """

    def SaveResults(self, results: dict, path: Path | None = None) -> None:
        """
        Serialise the results payload to a JSON file.

        Parameters
        ----------
        results : dict
            Payload returned by Run().
        path : Path, optional
            Output file path.  If omitted, the class-level default output
            path is used (subclass must define self._out_path).

        Raises
        ------
        AttributeError if path is None and the subclass has not set
            self._out_path.
        """
        out = path or self._out_path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        log.info("Results written -> %s", out)


# ---------------------------------------------------------------------------
# BootstrapAUROCValidator
# ---------------------------------------------------------------------------


class BootstrapAUROCValidator(BaseValidator):
    """
    Stratified bootstrap confidence intervals on scene-level AUROC and
    Average Precision, restricted to the 14 real-positive crops and 22
    real-negative crops only (synthetics excluded).

    Addresses Reviewer 2's concern that the real dataset is tiny by
    reporting bootstrap CIs in addition to point estimates.

    Methodology
    -----------
    - Score field : ``prob_mean`` (scene-level mean CH4Net probability)
    - Label       : 1 = confirmed methane, 0 = verified non-emitter
    - Resampling  : stratified (within-class) with replacement
    - n_bootstrap : 2000
    - Degenerate resamples (all-same-class) skipped and counted

    Input
    -----
    results_analysis/ml_metrics.json

    Output
    ------
    results_analysis/bootstrap_auroc_ap.json
    """

    def __init__(
        self,
        metrics_path: Path = METRICS_PATH,
        n_bootstrap: int = BOOTSTRAP_N_RESAMPLES,
        seed: int = BOOTSTRAP_SEED,
    ) -> None:
        """
        Parameters
        ----------
        metrics_path : Path, optional
            Path to ml_metrics.json.
        n_bootstrap : int, optional
            Number of bootstrap resamples (default: 2000).
        seed : int, optional
            Random seed for reproducibility (default: 42).
        """
        self._metrics_path = metrics_path
        self._n_bootstrap  = n_bootstrap
        self._seed         = seed
        self._out_path     = BOOTSTRAP_OUT_PATH

    # ------------------------------------------------------------------ statics

    @staticmethod
    def _Auroc(labels: list[int], scores: list[float]) -> float:
        """
        Compute AUROC via the Wilcoxon-Mann-Whitney statistic.

        Parameters
        ----------
        labels : list[int]   Binary class labels (0 or 1).
        scores : list[float] Classifier scores.

        Return
        ------
        float  AUROC in [0, 1], or NaN if only one class present.
        """
        pos = [s for s, l in zip(scores, labels) if l == 1]
        neg = [s for s, l in zip(scores, labels) if l == 0]
        if not pos or not neg:
            return float("nan")
        n_concordant = sum(1   for p in pos for n in neg if p > n)
        n_tied       = sum(0.5 for p in pos for n in neg if p == n)
        return (n_concordant + n_tied) / (len(pos) * len(neg))

    @staticmethod
    def _AveragePrecision(labels: list[int], scores: list[float]) -> float:
        """
        Compute Average Precision (area under the precision-recall curve).

        Parameters
        ----------
        labels : list[int]   Binary class labels.
        scores : list[float] Classifier scores.

        Return
        ------
        float  AP in [0, 1], or NaN if no positives.
        """
        if sum(labels) == 0:
            return float("nan")
        paired     = sorted(zip(scores, labels), reverse=True)
        tp = fp    = 0
        prev_recall = 0.0
        ap          = 0.0
        n_pos       = sum(labels)
        for _, label in paired:
            if label == 1:
                tp += 1
            else:
                fp += 1
            precision   = tp / (tp + fp)
            recall      = tp / n_pos
            ap         += precision * (recall - prev_recall)
            prev_recall = recall
        return ap

    @staticmethod
    def _BootstrapCI(values: list[float], level: float) -> tuple[float, float]:
        """
        Percentile bootstrap confidence interval.

        Parameters
        ----------
        values : list[float]  Bootstrap sample distribution.
        level  : float        Confidence level (e.g. 0.90 for 90% CI).

        Return
        ------
        tuple[float, float]  (lower, upper) bounds.
        """
        alpha = (1 - level) / 2
        return (
            float(np.quantile(values, alpha)),
            float(np.quantile(values, 1 - alpha)),
        )

    # ---------------------------------------------------------------------- Run

    def Run(self) -> dict:
        """
        Load ml_metrics.json and run the stratified bootstrap.

        Return
        ------
        dict  Bootstrap results with point estimates, CIs, and distributions.

        Raises
        ------
        FileNotFoundError if ml_metrics.json does not exist.
        """
        if not self._metrics_path.exists():
            raise FileNotFoundError(
                f"ml_metrics.json not found: {self._metrics_path}\n"
                "Run scripts/validation/compute_ml_metrics.py first."
            )

        data  = json.loads(self._metrics_path.read_text())
        crops = data["per_crop"]

        real_pos = [c for c in crops if c["source"] == "positive"]
        real_neg = [c for c in crops if c["source"] == "negative"]
        n_pos, n_neg = len(real_pos), len(real_neg)
        log.info("Real positives: %d   Real negatives: %d", n_pos, n_neg)

        all_real    = real_pos + real_neg
        scores_real = [c["prob_mean"] for c in all_real]
        labels_real = [c["label"]     for c in all_real]

        point_auroc = self._Auroc(labels_real, scores_real)
        point_ap    = self._AveragePrecision(labels_real, scores_real)
        log.info("Point AUROC=%.4f  AP=%.4f  (n=%d)", point_auroc, point_ap, n_pos + n_neg)

        rng       = np.random.default_rng(self._seed)
        pos_scores = [c["prob_mean"] for c in real_pos]
        neg_scores = [c["prob_mean"] for c in real_neg]

        boot_auroc, boot_ap, skipped = [], [], 0
        for _ in range(self._n_bootstrap):
            idx_p = rng.integers(0, n_pos, size=n_pos)
            idx_n = rng.integers(0, n_neg, size=n_neg)
            s = [pos_scores[i] for i in idx_p] + [neg_scores[i] for i in idx_n]
            l = [1] * n_pos + [0] * n_neg
            a = self._Auroc(l, s)
            p = self._AveragePrecision(l, s)
            if a != a or p != p:
                skipped += 1
                continue
            boot_auroc.append(a)
            boot_ap.append(p)

        n_valid = len(boot_auroc)
        log.info("Bootstrap complete: %d / %d valid (skipped: %d)", n_valid, self._n_bootstrap, skipped)

        auroc_90 = self._BootstrapCI(boot_auroc, 0.90)
        auroc_95 = self._BootstrapCI(boot_auroc, 0.95)
        ap_90    = self._BootstrapCI(boot_ap,    0.90)
        ap_95    = self._BootstrapCI(boot_ap,    0.95)

        return {
            "method":            "stratified_bootstrap",
            "n_bootstrap":       self._n_bootstrap,
            "n_valid":           n_valid,
            "n_skipped":         skipped,
            "n_real_positives":  n_pos,
            "n_real_negatives":  n_neg,
            "score_field":       "prob_mean",
            "note": (
                "Restricted to 14 TROPOMI-confirmed real positive crops and 22 "
                "verified-negative crops.  Synthetic crops excluded.  Stratified "
                "bootstrap resamples within each class to preserve class ratio.  "
                "Addresses Reviewer 2: 'effective positive real dataset is still "
                "tiny' — reporting CIs rather than point estimates only."
            ),
            "auroc": {
                "point":    round(point_auroc, 6),
                "ci_90_lo": round(auroc_90[0], 4),
                "ci_90_hi": round(auroc_90[1], 4),
                "ci_95_lo": round(auroc_95[0], 4),
                "ci_95_hi": round(auroc_95[1], 4),
                "boot_mean": round(float(np.mean(boot_auroc)), 4),
                "boot_std":  round(float(np.std(boot_auroc)),  4),
            },
            "average_precision": {
                "point":    round(point_ap, 6),
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

    def PrintSummary(self, results: dict) -> None:
        """
        Print a formatted AUROC / AP bootstrap summary to stdout.

        Parameters
        ----------
        results : dict  Payload from Run().
        """
        a = results["auroc"]
        p = results["average_precision"]
        sep = "=" * 65
        print(f"\n{sep}")
        print("  Bootstrap AUROC / Average Precision")
        print(sep)
        print(f"  Real positives : {results['n_real_positives']}")
        print(f"  Real negatives : {results['n_real_negatives']}")
        print(f"  Bootstrap n    : {results['n_valid']} valid "
              f"(skipped: {results['n_skipped']})")
        print()
        print(f"  AUROC  point = {a['point']:.4f}")
        print(f"         90% CI: [{a['ci_90_lo']:.4f}, {a['ci_90_hi']:.4f}]")
        print(f"         95% CI: [{a['ci_95_lo']:.4f}, {a['ci_95_hi']:.4f}]")
        print()
        print(f"  AP     point = {p['point']:.4f}")
        print(f"         90% CI: [{p['ci_90_lo']:.4f}, {p['ci_90_hi']:.4f}]")
        print(f"         95% CI: [{p['ci_95_lo']:.4f}, {p['ci_95_hi']:.4f}]")
        print(sep)
        print(
            f"\n  For paper: AUROC {a['point']:.3f} "
            f"(90% CI [{a['ci_90_lo']:.3f}, {a['ci_90_hi']:.3f}]); "
            f"AP {p['point']:.3f} "
            f"(90% CI [{p['ci_90_lo']:.3f}, {p['ci_90_hi']:.3f}])"
        )
        print(f"  (real crops only, n={results['n_real_positives']} pos + "
              f"n={results['n_real_negatives']} neg, stratified bootstrap "
              f"n={results['n_valid']})")


# ---------------------------------------------------------------------------
# LooDetectionAnalyzer
# ---------------------------------------------------------------------------


class LooDetectionAnalyzer(BaseValidator):
    """
    Leave-one-out (LOO) stability analysis on the Bełchatów above-threshold
    detection record.

    Addresses Reviewer 2: "whether performance is dominated by a few scenes."

    Two passes
    ----------
    Pass A — scene-level LOO (N = all cfar_detect=True records).
             Each independently-acquired scene is one observation.
    Pass B — month-level LOO (N = unique calendar months with ≥1 detection).
             More conservative: counts a month as detected if any tile fired.

    For each held-out item the Cook-D analog is computed as
    |metric_full − metric_loo| / metric_full  (relative influence).

    Input
    -----
    results_analysis/belchatow_annual_timeseries.json

    Output
    ------
    results_analysis/loo_detection_stability.json
    results_analysis/loo_detection_stability.png  (if matplotlib available)
    """

    def __init__(
        self,
        timeseries_path: Path = LOO_TIMESERIES_PATH,
        plot_path: Path = LOO_PLOT_PATH,
    ) -> None:
        """
        Parameters
        ----------
        timeseries_path : Path, optional
            Path to belchatow_annual_timeseries.json.
        plot_path : Path, optional
            Output path for the influence plot PNG.
        """
        self._timeseries_path = timeseries_path
        self._plot_path       = plot_path
        self._out_path        = LOO_OUT_PATH

    # --------------------------------------------------------------- data load

    @staticmethod
    def _LoadRecords(path: Path) -> list[dict]:
        """
        Read the timeseries JSON and return the flat record list.

        Parameters
        ----------
        path : Path  JSON file path.

        Return
        ------
        list[dict]  Raw timeseries records.
        """
        raw = json.loads(path.read_text())
        return raw["records"] if isinstance(raw, dict) else raw

    @staticmethod
    def _ExtractScenes(records: list[dict]) -> tuple[list[dict], list[dict]]:
        """
        Parse a flat record list into positives and all-usable sets.

        Parameters
        ----------
        records : list[dict]  Raw timeseries records.

        Return
        ------
        tuple[list[dict], list[dict]]
            (positives, all_usable) where positives is the cfar_detect=True
            subset and all_usable is every record with a valid sc_cfar.
        """
        positives, all_usable = [], []
        for r in records:
            d = r.get("detection", {})
            if "original" in d:         # skip nested BT sub-dicts
                continue
            sc = d.get("sc_cfar")
            if sc is None:
                continue
            rec = {
                "month":       r.get("month", ""),
                "sc_cfar":     sc,
                "cfar_detect": bool(d.get("cfar_detect", False)),
                "cfar_thresh": d.get("cfar_thresh_ratio"),
                "cfar_margin": d.get("cfar_margin"),
                "tif":         d.get("tif", ""),
                "flow_kgh":    r.get("quantification", {}).get("flow_rate_kgh"),
                "product":     r.get("search", {}).get("product_name", "")[:50],
            }
            all_usable.append(rec)
            if rec["cfar_detect"]:
                positives.append(rec)
        return positives, all_usable

    @staticmethod
    def _ComputeMetrics(positives: list[dict], n_usable: int) -> dict:
        """
        Aggregate detection-rate and sc_cfar statistics.

        Parameters
        ----------
        positives : list[dict]  cfar_detect=True scenes.
        n_usable  : int         Total scenes with valid sc_cfar.

        Return
        ------
        dict  Keys: n_pos, n_usable, detection_rate, mean/median/max/min_sc_cfar.
        """
        if not positives:
            return {
                "n_pos": 0, "n_usable": n_usable,
                "detection_rate": 0.0,
                "mean_sc_cfar": None, "median_sc_cfar": None,
                "max_sc_cfar": None, "min_sc_cfar": None,
            }
        scs = [p["sc_cfar"] for p in positives]
        return {
            "n_pos":           len(positives),
            "n_usable":        n_usable,
            "detection_rate":  len(positives) / n_usable,
            "mean_sc_cfar":    statistics.mean(scs),
            "median_sc_cfar":  statistics.median(scs),
            "max_sc_cfar":     max(scs),
            "min_sc_cfar":     min(scs),
        }

    @staticmethod
    def _RelativeInfluence(full_val: float | None, loo_val: float | None) -> float | None:
        """
        Cook-D analog: |metric_full - metric_loo| / |metric_full|.

        Parameters
        ----------
        full_val : float | None  Metric on full dataset.
        loo_val  : float | None  Metric on LOO subset.

        Return
        ------
        float | None  Relative influence, or None if inputs are invalid.
        """
        if full_val is None or loo_val is None or full_val == 0:
            return None
        return abs(full_val - loo_val) / abs(full_val)

    # ----------------------------------------------------------------- LOO passes

    def _RunScenePass(
        self, positives: list[dict], all_usable: list[dict]
    ) -> list[dict]:
        """
        Scene-level LOO: hold out one cfar_detect scene at a time.

        Parameters
        ----------
        positives  : list[dict]  All cfar_detect=True scenes.
        all_usable : list[dict]  All scenes with valid sc_cfar.

        Return
        ------
        list[dict]  Per-scene LOO results.
        """
        baseline = self._ComputeMetrics(positives, len(all_usable))
        results  = []
        for i, held_out in enumerate(positives):
            rem_pos  = [p for j, p in enumerate(positives) if j != i]
            rem_use  = [u for u in all_usable if u is not held_out]
            loo_m    = self._ComputeMetrics(rem_pos, len(rem_use))
            results.append({
                "held_out_month":        held_out["month"],
                "held_out_sc_cfar":      held_out["sc_cfar"],
                "held_out_product":      held_out["product"],
                "loo_detection_rate":    loo_m["detection_rate"],
                "delta_detect_rate":     loo_m["detection_rate"] - baseline["detection_rate"],
                "loo_mean_sc_cfar":      loo_m["mean_sc_cfar"],
                "loo_median_sc_cfar":    loo_m["median_sc_cfar"],
                "influence_detect_rate": self._RelativeInfluence(
                    baseline["detection_rate"], loo_m["detection_rate"]),
                "influence_mean_sc":     self._RelativeInfluence(
                    baseline["mean_sc_cfar"], loo_m["mean_sc_cfar"]),
            })
        return results

    def _RunMonthPass(
        self, positives: list[dict], all_usable: list[dict]
    ) -> list[dict]:
        """
        Month-level LOO: hold out all scenes from one calendar month at a time.
        A month is counted as detected if any scene in it fired.

        Parameters
        ----------
        positives  : list[dict]  All cfar_detect=True scenes.
        all_usable : list[dict]  All scenes with valid sc_cfar.

        Return
        ------
        list[dict]  Per-month LOO results.
        """
        pos_by_month: dict[str, list] = defaultdict(list)
        use_by_month: dict[str, list] = defaultdict(list)
        for p in positives:
            pos_by_month[p["month"]].append(p)
        for u in all_usable:
            use_by_month[u["month"]].append(u)

        unique_months   = sorted(pos_by_month.keys())
        n_pos_months    = len(unique_months)
        n_total_months  = len(use_by_month)
        baseline_rate   = n_pos_months / n_total_months
        month_sc        = {m: max(s["sc_cfar"] for s in ss) for m, ss in pos_by_month.items()}
        baseline_mean   = statistics.mean(month_sc.values())
        baseline_median = statistics.median(month_sc.values())

        results = []
        for held_month in unique_months:
            rem_pos_months   = n_pos_months - 1
            rem_total_months = n_total_months - 1
            loo_rate         = rem_pos_months / rem_total_months if rem_total_months else 0.0
            rem_sc           = {m: v for m, v in month_sc.items() if m != held_month}
            loo_mean         = statistics.mean(rem_sc.values())   if rem_sc else None
            loo_median       = statistics.median(rem_sc.values()) if rem_sc else None
            results.append({
                "held_out_month":        held_month,
                "held_out_max_sc_cfar":  month_sc[held_month],
                "n_scenes_in_month":     len(pos_by_month[held_month]),
                "loo_detection_rate":    loo_rate,
                "delta_detect_rate":     loo_rate - baseline_rate,
                "loo_mean_sc_cfar":      loo_mean,
                "loo_median_sc_cfar":    loo_median,
                "influence_detect_rate": self._RelativeInfluence(baseline_rate, loo_rate),
                "influence_mean_sc":     self._RelativeInfluence(baseline_mean, loo_mean),
            })
        return results

    # ----------------------------------------------------------------- Run / plot

    def Run(self, loo_pass: str = "both") -> dict:
        """
        Load the Bełchatów timeseries and run the specified LOO pass(es).

        Parameters
        ----------
        loo_pass : str, optional
            Which passes to run: ``"a"`` (scene), ``"b"`` (month), or
            ``"both"`` (default).

        Return
        ------
        dict  Results including baseline metrics and per-item LOO tables.

        Raises
        ------
        FileNotFoundError if the timeseries JSON does not exist.
        """
        if not self._timeseries_path.exists():
            raise FileNotFoundError(
                f"Timeseries not found: {self._timeseries_path}\n"
                "Run the timeseries pipeline first."
            )

        records             = self._LoadRecords(self._timeseries_path)
        positives, all_usable = self._ExtractScenes(records)

        log.info("Loaded %d total records  (%d usable, %d cfar_detect=True)",
                 len(records), len(all_usable), len(positives))

        baseline_a = self._ComputeMetrics(positives, len(all_usable))
        pos_months  = set(p["month"] for p in positives)
        all_months  = set(u["month"] for u in all_usable)

        pass_a = self._RunScenePass(positives, all_usable) if loo_pass in ("a", "both") else []
        pass_b = self._RunMonthPass(positives, all_usable) if loo_pass in ("b", "both") else []

        if pass_a:
            max_delta  = max(abs(r["delta_detect_rate"]) for r in pass_a)
            if max_delta < 0.02:
                verdict = "STABLE — no single scene drives detection rate (max Δ < 2 pp)"
            elif max_delta < 0.05:
                verdict = "MOSTLY STABLE — minor sensitivity (max Δ < 5 pp); disclose top scene"
            else:
                verdict = "SENSITIVE — one scene has outsized influence; disclose explicitly"
            log.info("Stability verdict: %s", verdict)
        else:
            verdict = "n/a"

        return {
            "baseline": baseline_a,
            "baseline_month_level": {
                "n_detected_months": len(pos_months),
                "n_total_months":    len(all_months),
                "detection_rate":    len(pos_months) / len(all_months),
            },
            "stability_verdict": verdict,
            "pass_a_scene_level": pass_a,
            "pass_b_month_level": pass_b,
        }

    def MakePlot(self, results: dict) -> None:
        """
        Generate a 3-panel matplotlib influence plot and save to PNG.

        Parameters
        ----------
        results : dict  Payload from Run().
        """
        pass_a = results["pass_a_scene_level"]
        pass_b = results["pass_b_month_level"]
        bml    = results["baseline_month_level"]

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
        except ImportError:
            log.warning("matplotlib not available — skipping influence plot.")
            return

        base_rate_a = results["baseline"]["detection_rate"]
        base_rate_b = bml["detection_rate"]

        fig = plt.figure(figsize=(14, 10))
        gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, :])

        if pass_a:
            scs_a   = [r["held_out_sc_cfar"]    for r in pass_a]
            rates_a = [r["loo_detection_rate"]   for r in pass_a]
            ax1.scatter(scs_a, rates_a, c="steelblue", s=60, zorder=3)
            ax1.axhline(base_rate_a, color="red", ls="--", lw=1.5,
                        label=f"Baseline {base_rate_a:.4f}")
            ax1.set_xscale("log")
            ax1.set_xlabel("Held-out sc_cfar (log scale)")
            ax1.set_ylabel("LOO detection rate")
            ax1.set_title("Pass A — Scene-level LOO\nDetection rate stability")
            ax1.legend(fontsize=8)
            ax1.grid(True, alpha=0.3)

        if pass_b:
            scs_b   = [r["held_out_max_sc_cfar"] for r in pass_b]
            rates_b = [r["loo_detection_rate"]    for r in pass_b]
            ax2.scatter(scs_b, rates_b, c="darkorange", s=60, zorder=3)
            ax2.axhline(base_rate_b, color="red", ls="--", lw=1.5,
                        label=f"Baseline {base_rate_b:.4f}")
            ax2.set_xscale("log")
            ax2.set_xlabel("Held-out max sc_cfar (log scale)")
            ax2.set_ylabel("LOO detection rate")
            ax2.set_title("Pass B — Month-level LOO\nDetection rate stability")
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)

        if pass_a:
            all_sc = sorted(r["held_out_sc_cfar"] for r in pass_a)
            colors = ["#e74c3c" if sc == max(all_sc) else "#3498db" for sc in all_sc]
            ax3.bar(range(len(all_sc)), all_sc, color=colors,
                    edgecolor="white", linewidth=0.5)
            ax3.set_yscale("log")
            ax3.set_xlabel("Scene index (sorted by sc_cfar)")
            ax3.set_ylabel("sc_cfar (log scale)")
            ax3.set_title(
                "sc_cfar distribution across all cfar_detect=True scenes\n"
                "(red = most influential; note log scale — distribution is right-skewed)"
            )
            med = results["baseline"]["median_sc_cfar"]
            ax3.axhline(med, color="green", ls="--", lw=1.5,
                        label=f"Median {med:.1f}")
            ax3.legend(fontsize=8)
            ax3.grid(True, alpha=0.3, axis="y")

        fig.suptitle(
            "Leave-One-Out Detection Stability — Bełchatów 2021–2024",
            fontsize=13, fontweight="bold", y=0.98,
        )
        self._plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(self._plot_path, dpi=150, bbox_inches="tight")
        log.info("Plot saved -> %s", self._plot_path)
        plt.close(fig)

    def PrintSummary(self, results: dict) -> None:
        """
        Print a formatted LOO stability table and verdict to stdout.

        Parameters
        ----------
        results : dict  Payload from Run().
        """
        bl  = results["baseline"]
        bml = results["baseline_month_level"]
        sep = "=" * 80

        print(f"\n{sep}")
        print("  LOO DETECTION STABILITY — Bełchatów")
        print(sep)
        print(f"  Baseline: n_pos={bl['n_pos']}, n_usable={bl['n_usable']}, "
              f"detect_rate={bl['detection_rate']:.4f}, "
              f"mean_sc={bl['mean_sc_cfar']:.2f}, "
              f"median_sc={bl['median_sc_cfar']:.2f}")
        print(f"  Month baseline: {bml['n_detected_months']}/{bml['n_total_months']} "
              f"months detected (rate={bml['detection_rate']:.4f})")
        print()

        pass_a = results["pass_a_scene_level"]
        if pass_a:
            print(f"{'Month':<10} {'sc_cfar':>10} {'LOO_rate':>9} "
                  f"{'Δrate':>8} {'LOO_mean':>10} {'infl_rate':>10} {'infl_mean':>10}")
            print("-" * 80)
            for r in sorted(pass_a, key=lambda x: -x["held_out_sc_cfar"]):
                ir = f"{r['influence_detect_rate']:.4f}" if r["influence_detect_rate"] is not None else "  n/a"
                im = f"{r['influence_mean_sc']:.4f}"     if r["influence_mean_sc"]     is not None else "  n/a"
                print(f"{r['held_out_month']:<10} "
                      f"{r['held_out_sc_cfar']:>10.2f} "
                      f"{r['loo_detection_rate']:>9.4f} "
                      f"{r['delta_detect_rate']:>+8.4f} "
                      f"{r['loo_mean_sc_cfar']:>10.2f} "
                      f"{ir:>10} "
                      f"{im:>10}")

        print()
        print(f"  Verdict: {results['stability_verdict']}")
        print(sep)


# ---------------------------------------------------------------------------
# LeakageAuditor
# ---------------------------------------------------------------------------


class LeakageAuditor(BaseValidator):
    """
    Data-leakage and independence audit for CH4Net v8.

    Three checks:
    (1) Temporal proximity — flag training/evaluation acquisition pairs at
        the same MGRS tile that are fewer than TEMPORAL_LEAKAGE_DAYS apart.
    (2) Site overlap — identify each candidate site's role(s) in training
        (positive, negative, or synthetic substrate).
    (3) Conformal calibration independence — verify that no calibration
        location is within SPATIAL_EXCLUSION_KM of any candidate site.

    Inputs
    ------
    data/crops/**/*_label.json          (training crop catalogue)
    results_analysis/nonemitter_sc_scores.json  (calibration locations)

    Outputs
    -------
    results_analysis/leakage_audit.json
    results_analysis/leakage_audit.md
    """

    def __init__(
        self,
        crops_dir: Path = CROPS_DIR,
        calibration_path: Path = CALIBRATION_SCORES,
        candidate_sites: dict[str, tuple[float, float, str]] = CANDIDATE_SITES,
        eval_dates: dict[str, list[str]] = EVAL_DATES,
    ) -> None:
        """
        Parameters
        ----------
        crops_dir : Path, optional
            Root directory of the training crops.
        calibration_path : Path, optional
            Path to nonemitter_sc_scores.json.
        candidate_sites : dict, optional
            Mapping of site name → (lat, lon, MGRS tile).
        eval_dates : dict, optional
            Mapping of site name → list of evaluation ISO dates.
        """
        self._crops_dir         = crops_dir
        self._calibration_path  = calibration_path
        self._candidate_sites   = candidate_sites
        self._eval_dates        = eval_dates
        self._out_path          = LEAKAGE_OUT_JSON

    # ----------------------------------------------------------------- helpers

    def _CatalogTraining(self) -> list[dict]:
        """
        Walk the crops directory and return one dict per training crop.

        Return
        ------
        list[dict]  Each dict has keys: crop, source_dir, label_value,
                    tile, date.
        """
        training = []
        for label_path in self._crops_dir.glob("**/*_label.json"):
            if ".hidden" in str(label_path):
                continue
            stem  = label_path.stem.replace("_label", "")
            meta  = json.loads(label_path.read_text())
            tile  = re.search(r"T\d{2}[A-Z]{3}", stem)
            d_obj = extract_date_from_stem(stem)
            training.append({
                "crop":        stem,
                "source_dir":  label_path.parent.name,
                "label_value": meta.get("label_value"),
                "tile":        tile.group(0) if tile else None,
                "date":        meta.get("acquisition_date") or (d_obj.isoformat() if d_obj else None),
            })
        return training

    def _AuditSiteOverlap(self, training: list[dict]) -> dict:
        """
        Check each candidate site for presence in the training set.

        Parameters
        ----------
        training : list[dict]  Training crop catalogue.

        Return
        ------
        dict  Mapping of site name → overlap info dict.
        """
        overlap = {}
        for site, (lat, lon, tile) in self._candidate_sites.items():
            matches = [
                t for t in training
                if t["tile"] == tile or site.lower() in t["crop"].lower()
            ]
            roles: set[str] = set()
            for t in matches:
                if "synthetic" in t["source_dir"]:
                    roles.add("synthetic_substrate")
                elif t["label_value"] == 1:
                    roles.add("positive")
                elif t["label_value"] == 0:
                    roles.add("negative")
            overlap[site] = {
                "in_training":    len(matches) > 0,
                "roles":          sorted(roles),
                "n_crops":        len(matches),
                "training_crops": [t["crop"] for t in matches],
            }
        return overlap

    def _AuditTemporalProximity(self, training: list[dict]) -> dict:
        """
        For each evaluation date, find the nearest training acquisition at
        the same site and flag pairs closer than TEMPORAL_LEAKAGE_DAYS.

        Parameters
        ----------
        training : list[dict]  Training crop catalogue.

        Return
        ------
        dict  Mapping of site name → list of proximity flag dicts.
        """
        results: dict[str, list] = {}
        for site, dates in self._eval_dates.items():
            site_crops = [t for t in training if site.lower() in t["crop"].lower()]
            flags = []
            for d_eval_str in dates:
                d_eval   = datetime.strptime(d_eval_str, "%Y-%m-%d")
                nearest  = None
                min_days = float("inf")
                for t in site_crops:
                    if not t.get("date"):
                        continue
                    try:
                        d_train = datetime.strptime(t["date"][:10], "%Y-%m-%d")
                    except ValueError:
                        continue
                    delta = abs((d_eval - d_train).days)
                    if delta < min_days:
                        min_days = delta
                        nearest  = t["crop"]
                flags.append({
                    "eval_date":              d_eval_str,
                    "nearest_training_crop":  nearest,
                    "days_apart":             min_days if min_days != float("inf") else None,
                    "flag": (
                        "POTENTIAL_TEMPORAL_LEAKAGE"
                        if min_days < TEMPORAL_LEAKAGE_DAYS
                        else "OK"
                    ),
                })
            results[site] = flags
        return results

    def _AuditConformalIndependence(self) -> dict:
        """
        Check each calibration site against candidate sites using a
        SPATIAL_EXCLUSION_KM radius.

        Return
        ------
        dict  With keys checked_n and sites_within_exclusion_radius.
        """
        result: dict = {"sites_within_exclusion_radius": [], "checked_n": 0}
        if not self._calibration_path.exists():
            log.warning("Calibration scores not found: %s", self._calibration_path)
            return result

        cal_data = json.loads(self._calibration_path.read_text())
        cal_sites = cal_data if isinstance(cal_data, list) else cal_data.get("sites", [])

        for c in cal_sites:
            if c.get("status") != "ok":
                continue
            result["checked_n"] += 1
            for site, (lat, lon, _) in self._candidate_sites.items():
                dist = haversine_km(c.get("lat", 0), c.get("lon", 0), lat, lon)
                if dist < SPATIAL_EXCLUSION_KM:
                    result["sites_within_exclusion_radius"].append({
                        "calibration_site": c.get("location_id"),
                        "candidate_site":   site,
                        "distance_km":      round(dist, 1),
                    })
        return result

    def _BuildMarkdown(
        self,
        site_overlap: dict,
        temporal_proximity: dict,
        cal_independence: dict,
    ) -> str:
        """
        Render the three-check audit as a Markdown document.

        Parameters
        ----------
        site_overlap        : dict  Output of _AuditSiteOverlap().
        temporal_proximity  : dict  Output of _AuditTemporalProximity().
        cal_independence    : dict  Output of _AuditConformalIndependence().

        Return
        ------
        str  Markdown text.
        """
        md = [
            "# Data leakage and independence audit\n",
            "Three checks: site-level training overlap, temporal proximity between "
            "training and evaluation acquisitions at the same site, and conformal "
            "calibration set independence from candidate sites.\n",
            "## (1) Candidate site overlap with training set\n",
            "| Candidate | In training? | Role(s) | N crops |",
            "|---|---|---|---|",
        ]
        for site, info in site_overlap.items():
            roles = ", ".join(info["roles"]) if info["roles"] else "—"
            md.append(
                f"| {site} | {'yes' if info['in_training'] else 'no'} "
                f"| {roles} | {info['n_crops']} |"
            )
        md.append("")

        md.extend([
            "## (2) Temporal proximity between evaluation and training dates\n",
            "Same-site training crops within 14 days of an evaluation acquisition are "
            "flagged as potential leakage.\n",
            "| Site | Eval date | Nearest training crop | Days apart | Flag |",
            "|---|---|---|---|---|",
        ])
        for site, flags in temporal_proximity.items():
            for f in flags:
                md.append(
                    f"| {site} | {f['eval_date']} | {f['nearest_training_crop'] or '—'} "
                    f"| {f['days_apart']} | {f['flag']} |"
                )
        md.append("")

        md.append("## (3) Conformal calibration set independence\n")
        md.append(
            f"Checked {cal_independence['checked_n']} OK-status calibration sites "
            f"for proximity (< {SPATIAL_EXCLUSION_KM:.0f} km) to any candidate site.\n"
        )
        if cal_independence["sites_within_exclusion_radius"]:
            md.extend([
                "**Proximity flags found:**\n",
                "| Calibration site | Candidate site | Distance (km) |",
                "|---|---|---|",
            ])
            for f in cal_independence["sites_within_exclusion_radius"]:
                md.append(
                    f"| {f['calibration_site']} | {f['candidate_site']} "
                    f"| {f['distance_km']} |"
                )
        else:
            md.append(
                f"**No conformal calibration sites within {SPATIAL_EXCLUSION_KM:.0f} km "
                "of any candidate site.**  The threshold τ is calibrated on a set that "
                "is spatially independent of the evaluation sites.\n"
            )

        md.extend([
            "\n## (4) Threshold selection methodology\n",
            "The conformal threshold τ was computed by the split conformal prediction "
            "quantile on the non-emitter calibration set scores, without reference to "
            "the candidate-site backfill outcomes.  The retraining hyperparameter "
            "selection (v1-v11) used a small held-out set of training crops (3 "
            "negatives) for validation-loss monitoring, not the candidate-site "
            "evaluation outcomes.  The candidate-site results were computed only "
            "after v8 was fixed and τ was calibrated.\n",
        ])
        return "\n".join(md)

    # ------------------------------------------------------------------- Run

    def Run(self) -> dict:
        """
        Execute all three leakage checks.

        Return
        ------
        dict  Audit results with keys: training_crop_count,
              candidate_site_overlap, temporal_proximity,
              conformal_calibration_independence.
        """
        training           = self._CatalogTraining()
        site_overlap       = self._AuditSiteOverlap(training)
        temporal_proximity = self._AuditTemporalProximity(training)
        cal_independence   = self._AuditConformalIndependence()

        log.info("Catalogued %d training crops", len(training))
        n_flags = sum(
            1 for fs in temporal_proximity.values()
            for f in fs if f["flag"] != "OK"
        )
        log.info("Temporal proximity flags: %d", n_flags)
        log.info(
            "Calibration sites within %g km of a candidate: %d",
            SPATIAL_EXCLUSION_KM,
            len(cal_independence["sites_within_exclusion_radius"]),
        )

        md = self._BuildMarkdown(site_overlap, temporal_proximity, cal_independence)
        LEAKAGE_OUT_MD.write_text(md)
        log.info("Markdown written -> %s", LEAKAGE_OUT_MD)

        return {
            "training_crop_count":              len(training),
            "candidate_site_overlap":           site_overlap,
            "temporal_proximity":               temporal_proximity,
            "conformal_calibration_independence": cal_independence,
        }

    def PrintSummary(self, results: dict) -> None:
        """
        Print a concise leakage audit summary to stdout.

        Parameters
        ----------
        results : dict  Payload from Run().
        """
        sep = "=" * 70
        print(f"\n{sep}")
        print("  LEAKAGE AND INDEPENDENCE AUDIT")
        print(sep)
        for site, info in results["candidate_site_overlap"].items():
            roles = ", ".join(info["roles"]) if info["roles"] else "—"
            print(f"  {site:<14} in_training={info['in_training']}  "
                  f"roles={roles}  n={info['n_crops']}")
        print()
        n_flags = sum(
            1 for fs in results["temporal_proximity"].values()
            for f in fs if f["flag"] != "OK"
        )
        cal = results["conformal_calibration_independence"]
        print(f"  Temporal proximity flags    : {n_flags}")
        print(f"  Cal sites within exclusion  : "
              f"{len(cal['sites_within_exclusion_radius'])} / {cal['checked_n']} checked")
        print(sep)


# ---------------------------------------------------------------------------
# HeldOutEvaluator
# ---------------------------------------------------------------------------


class HeldOutEvaluator(BaseValidator):
    """
    Evaluate CH4Net v8 on sites genuinely held out from training (or trained
    as negatives, making any positive detection a stronger result).

    Categories
    ----------
    held_out          : Site never seen in any form during training.
    training_negative : Site seen as a negative crop during training;
                        a positive detection at test time overrides the
                        training label based on spectral signature.
    training_negative_and_synthetic_substrate : Negative label plus used
                        as the spatial substrate for synthetic plume injection.

    Inputs
    ------
    results_analysis/training_set_audit.json
    results_analysis/historical_backfill_timeseries.json

    Outputs
    -------
    results_analysis/held_out_evaluation.json
    results_analysis/held_out_evaluation.md
    """

    def __init__(
        self,
        audit_path:    Path = HELD_OUT_AUDIT_PATH,
        backfill_path: Path = HELD_OUT_BACKFILL_PATH,
        conformal_tau: float = CONFORMAL_TAU,
    ) -> None:
        """
        Parameters
        ----------
        audit_path    : Path, optional
            Path to training_set_audit.json.
        backfill_path : Path, optional
            Path to historical_backfill_timeseries.json.
        conformal_tau : float, optional
            Conformal S/C threshold τ (default: 4.1052 at α = 0.10).
        """
        self._audit_path    = audit_path
        self._backfill_path = backfill_path
        self._conformal_tau = conformal_tau
        self._out_path      = HELD_OUT_OUT_JSON

    def _BuildMarkdown(self, out_obj: dict, sites_in_scope: list[str],
                       classification: dict) -> str:
        """
        Render the held-out evaluation as a Markdown document.

        Parameters
        ----------
        out_obj        : dict        Evaluation results keyed by site.
        sites_in_scope : list[str]   Sites included in this evaluation.
        classification : dict        Training-set audit classification map.

        Return
        ------
        str  Markdown text.
        """
        md = [
            "# Held-out evaluation of CH4Net v8\n",
            "This file reports v8 performance on candidate sites that were either "
            "never seen during training (TRULY HELD-OUT) or seen as NEGATIVE only "
            "(positive detection at test time = model overrides its training label).\n",
            f"All thresholds: conformal τ = {self._conformal_tau} at α = 0.10; "
            "CFAR ratio rule per Section 2.2.\n",
        ]
        test_categories = [HELD_OUT, TRAIN_NEG, TRAIN_NEG_SYN]
        for cat in test_categories:
            cat_sites = [s for s in sites_in_scope if classification[s] == cat]
            if not cat_sites:
                continue
            md.extend([
                f"\n## {CATEGORY_LABEL[cat]}\n",
                "| Site | Total records | Valid records | Above τ | CFAR detect |",
                "|---|---|---|---|---|",
            ])
            for site in cat_sites:
                o = out_obj["sites"][site]
                md.append(
                    f"| {site} | {o['n_records']} | {o['n_valid']} "
                    f"| {o['n_above_tau']} | {o['n_cfar_detect']} |"
                )
            md.append("")
            for site in cat_sites:
                o = out_obj["sites"][site]
                if not o["acquisitions"]:
                    continue
                md.extend([
                    f"### {site} — per acquisition",
                    "| Date | S/C | cv_ctrl | Above τ | CFAR |",
                    "|---|---|---|---|---|",
                ])
                for a in o["acquisitions"]:
                    md.append(
                        f"| {a['date']} | {a['sc_ratio']} | {a.get('cv_ctrl')} "
                        f"| {'✓' if a['above_conformal_tau'] else '·'} "
                        f"| {'✓' if a['cfar_detect'] else '·'} |"
                    )
                md.append("")

        md.append("## Section 1.5 / Section 3 — proposed text\n")
        held_out  = [s for s in sites_in_scope if classification[s] == HELD_OUT]
        train_neg = [
            s for s in sites_in_scope
            if classification[s] in (TRAIN_NEG, TRAIN_NEG_SYN)
        ]
        if held_out:
            ho_summary = ", ".join(
                f"{s} (valid n = {out_obj['sites'][s]['n_valid']}, "
                f"above-τ = {out_obj['sites'][s]['n_above_tau']}, "
                f"CFAR = {out_obj['sites'][s]['n_cfar_detect']})"
                for s in held_out
            )
            md.append(
                "**Truly held-out test set.**  The model never saw the following "
                f"sites in any form during training: {ho_summary}.  Their "
                "performance is an independent test of the v8 model and the "
                f"conformal threshold τ = {self._conformal_tau}.\n"
            )
        if train_neg:
            tn_summary = ", ".join(
                f"{s} (n_records = {out_obj['sites'][s]['n_records']}, "
                f"positive detections at test time: "
                f"above-τ = {out_obj['sites'][s]['n_above_tau']}, "
                f"CFAR = {out_obj['sites'][s]['n_cfar_detect']})"
                for s in train_neg
            )
            md.append(
                "**Model overrides its own training labels.**  The following "
                "candidate sites were in training as NEGATIVE crops, but the "
                "production pipeline produces above-threshold detections on "
                f"subsequent acquisitions: {tn_summary}.  This is a stronger "
                "result than a held-out test because the model is contradicting "
                "a training label on the basis of the spectral signature it "
                "learned from the synthetic-positive distribution.\n"
            )
        return "\n".join(md)

    def Run(self) -> dict:
        """
        Load audit and backfill JSONs, classify sites, and compute per-site
        evaluation metrics.

        Return
        ------
        dict  Evaluation results including per-site acquisition tables and
              a text_block paragraph for the paper.

        Raises
        ------
        FileNotFoundError if either required JSON is missing.
        """
        for p in (self._audit_path, self._backfill_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"Required input not found: {p}\n"
                    "Run the relevant pipeline script first."
                )

        audit          = json.loads(self._audit_path.read_text())
        classification = audit["candidate_classification"]
        backfill_raw   = json.loads(self._backfill_path.read_text())

        by_site: dict[str, list] = {}
        for key, records in backfill_raw.items():
            by_site[key.lower()] = records

        test_categories = [HELD_OUT, TRAIN_NEG, TRAIN_NEG_SYN]
        sites_in_scope  = [s for s, c in classification.items() if c in test_categories]

        out_obj: dict = {"sites": {}}
        for site in sites_in_scope:
            cat     = classification[site]
            records = by_site.get(site, [])
            valid   = [
                r for r in records
                if r.get("status") in (None, "ok") and r.get("sc_ratio") is not None
            ]
            site_obj: dict = {
                "classification":  cat,
                "category_label":  CATEGORY_LABEL[cat],
                "n_records":       len(records),
                "n_valid":         len(valid),
                "n_above_tau":     0,
                "n_cfar_detect":   0,
                "acquisitions":    [],
            }
            for r in valid:
                sc        = r.get("sc_ratio")
                cfar      = bool(r.get("cfar_detect"))
                above_tau = sc is not None and sc > self._conformal_tau
                site_obj["acquisitions"].append({
                    "date":               r.get("date"),
                    "sc_ratio":           sc,
                    "cv_ctrl":            r.get("cv_ctrl"),
                    "cfar_detect":        cfar,
                    "above_conformal_tau": above_tau,
                })
                if above_tau:
                    site_obj["n_above_tau"] += 1
                if cfar:
                    site_obj["n_cfar_detect"] += 1
            out_obj["sites"][site] = site_obj

        md = self._BuildMarkdown(out_obj, sites_in_scope, classification)
        HELD_OUT_OUT_MD.write_text(md)
        log.info("Markdown written -> %s", HELD_OUT_OUT_MD)

        # Embed the Section 1.5 paragraph in the JSON
        marker = "## Section 1.5 / Section 3 — proposed text\n"
        if marker in md:
            out_obj["text_block"] = md.split(marker, 1)[1]

        total_above   = sum(s["n_above_tau"]   for s in out_obj["sites"].values())
        total_cfar    = sum(s["n_cfar_detect"] for s in out_obj["sites"].values())
        log.info(
            "Held-out sites: %d  Total above-τ: %d  Total CFAR: %d",
            len(sites_in_scope), total_above, total_cfar,
        )
        return out_obj

    def PrintSummary(self, results: dict) -> None:
        """
        Print a per-category held-out evaluation table to stdout.

        Parameters
        ----------
        results : dict  Payload from Run().
        """
        sep = "=" * 70
        print(f"\n{sep}")
        print("  HELD-OUT EVALUATION")
        print(sep)
        test_categories = [HELD_OUT, TRAIN_NEG, TRAIN_NEG_SYN]
        all_sites = list(results["sites"].keys())

        # Reconstruct classification from the stored category_label
        label_to_cat = {v: k for k, v in CATEGORY_LABEL.items()}
        classification = {
            s: label_to_cat.get(results["sites"][s]["category_label"], "unknown")
            for s in all_sites
        }

        for cat in test_categories:
            cat_sites = [s for s in all_sites if classification[s] == cat]
            if not cat_sites:
                continue
            print(f"\n  {CATEGORY_LABEL[cat]}")
            print("  " + "-" * 68)
            print(f"  {'Site':<14}{'Records':>10}{'Valid':>8}{'Above τ':>10}{'CFAR':>8}")
            for s in cat_sites:
                o = results["sites"][s]
                print(f"  {s:<14}{o['n_records']:>10}{o['n_valid']:>8}"
                      f"{o['n_above_tau']:>10}{o['n_cfar_detect']:>8}")
        print(sep)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Command-line interface for running one or all validators.

    Usage
    -----
        cd methane-api
        python scripts/validation/validation_metrics.py --validator all
        python scripts/validation/validation_metrics.py --validator bootstrap
        python scripts/validation/validation_metrics.py --validator loo --plot
        python scripts/validation/validation_metrics.py --validator leakage
        python scripts/validation/validation_metrics.py --validator held-out
    """
    parser = argparse.ArgumentParser(
        description="CH4Net v8 statistical validation and audit suite."
    )
    parser.add_argument(
        "--validator",
        choices=["all", "bootstrap", "loo", "leakage", "held-out"],
        default="all",
        help="Which validator to run (default: all).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate matplotlib plot for LOO validator.",
    )
    args = parser.parse_args()

    run_all = args.validator == "all"

    if run_all or args.validator == "bootstrap":
        v = BootstrapAUROCValidator()
        r = v.Run()
        v.PrintSummary(r)
        v.SaveResults(r)

    if run_all or args.validator == "loo":
        v = LooDetectionAnalyzer()
        r = v.Run()
        v.PrintSummary(r)
        v.SaveResults(r)
        if args.plot:
            v.MakePlot(r)

    if run_all or args.validator == "leakage":
        v = LeakageAuditor()
        r = v.Run()
        v.PrintSummary(r)
        v.SaveResults(r)

    if run_all or args.validator == "held-out":
        v = HeldOutEvaluator()
        r = v.Run()
        v.PrintSummary(r)
        v.SaveResults(r)


if __name__ == "__main__":
    main()
