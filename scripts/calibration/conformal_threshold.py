"""
conformal_threshold.py
======================
Conformal Prediction Threshold Calibration for CH4Net Detection.

Derives a statistically guaranteed false-positive-rate (FPR) bound on the
CH4Net S/C detection score using split conformal prediction, then extends it
to stratified (Mondrian) per-ecoregion thresholds.

Class hierarchy
---------------
NonEmitterScoreLoader         Data-processing class. Loads, validates, and filters
                              the non-emitter S/C score JSON produced by
                              run_nonemitter_inference.py. Exposes the clean score
                              list and per-record metadata to calibration classes.

BaseThresholdCalibrator       Abstract base class. Defines the Fit() and
                              PrintSummary() interface that all calibrators share.

ConformalCalibrator           Primary production calibrator. Implements split
(BaseThresholdCalibrator)     conformal prediction (Angelopoulos & Bates 2021)
                              globally across the full calibration set. Adds
                              Bootstrap() for sampling uncertainty on tau and
                              SampleSizeAnalysis() for adequacy reporting.

MondrianConformalCalibrator   Stratified extension. Applies ConformalCalibrator
(ConformalCalibrator)         independently per group (ecoregion or CLC land-cover
                              class), producing per-stratum tau values with
                              small-N warnings.

Usage
-----
    cd methane-api
    python scripts/calibration/conformal_threshold.py [--alpha 0.10] [--no-plot]

Input
-----
    results_analysis/nonemitter_sc_scores.json   (produced by run_nonemitter_inference.py)

Output
------
    results_analysis/calibrated_threshold.json
    results_analysis/conformal_calibration.png   (optional; requires matplotlib)

References
----------
Angelopoulos & Bates (2021) "A Gentle Introduction to Conformal Prediction and
    Distribution-Free Uncertainty Quantification". arXiv:2107.07511.
Shafer & Vovk (2008) "A Tutorial on Conformal Prediction". JMLR 9.
"""

import argparse
import json
import logging
import math
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("conformal_threshold")

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------

ROOT        = Path(__file__).resolve().parent.parent.parent   # methane-api/
SCORES_PATH = ROOT / "results_analysis" / "nonemitter_sc_scores.json"
OUTPUT_PATH = ROOT / "results_analysis" / "calibrated_threshold.json"
PLOT_PATH   = ROOT / "results_analysis" / "conformal_calibration.png"

PRODUCTION_THRESHOLD = 1.15    # Vaughan et al. (2024) published heuristic
ALPHA_LEVELS         = [0.05, 0.10, 0.20]   # FPR levels to report
MIN_RECORDS_REQUIRED = 3       # minimum valid scores for calibration
SMALL_N_WARNING_THRESHOLD = 5  # groups below this size get a warning flag
BOOTSTRAP_N_RESAMPLES     = 2000
BOOTSTRAP_SEED            = 42


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

class NonEmitterScoreLoader:
    """
    Data-processing class. Loads, validates, and filters the non-emitter S/C
    score records produced by run_nonemitter_inference.py.

    Filters to records with status='ok', a valid numeric sc_cfar (or sc_ratio)
    field, and no NaN values. Exposes the clean score list and the full record
    list for downstream calibration and Mondrian stratification.

    Typical usage
    -------------
        loader = NonEmitterScoreLoader()
        loader.Load()
        scores  = loader.scores        # list[float]
        records = loader.records       # list[dict], status='ok' only
    """

    def __init__(self, path: Path = SCORES_PATH):
        """
        Parameters
        ----------
        path : Path, optional
            Path to the nonemitter_sc_scores.json file.
            Defaults to SCORES_PATH (results_analysis/nonemitter_sc_scores.json).
        """
        self._path   = path
        self.scores  = []
        self.records = []

    def Load(self) -> "NonEmitterScoreLoader":
        """
        Read the JSON file, filter to valid records, and populate self.scores
        and self.records.

        Return
        ------
        self  — for method chaining.

        Raises
        ------
        FileNotFoundError if the scores file does not exist.
        ValueError if fewer than MIN_RECORDS_REQUIRED valid scores are found.
        """
        if not self._path.exists():
            raise FileNotFoundError(
                f"Scores file not found: {self._path}\n"
                "Run: python scripts/calibration/run_nonemitter_inference.py"
            )

        all_records = json.loads(self._path.read_text())
        ok_records  = [r for r in all_records if r.get("status") == "ok"]
        log.info(
            "Loaded %d scored locations (%d with status='ok')",
            len(all_records), len(ok_records),
        )

        valid_scores = []
        valid_records = []
        for r in ok_records:
            sc = r.get("sc_cfar") or r.get("sc_ratio")
            if sc is not None and not math.isnan(sc):
                valid_scores.append(float(sc))
                valid_records.append(r)

        if len(valid_scores) < MIN_RECORDS_REQUIRED:
            raise ValueError(
                f"Only {len(valid_scores)} valid S/C scores found "
                f"(need at least {MIN_RECORDS_REQUIRED}). "
                "Run run_nonemitter_inference.py on more locations."
            )

        self.scores  = valid_scores
        self.records = valid_records
        log.info("Using %d valid S/C scores for calibration", len(valid_scores))
        return self

    def Describe(self) -> dict:
        """
        Return summary statistics for the loaded score set.

        Return
        ------
        dict with n, min, max, mean, median, std.
        """
        if not self.scores:
            return {"n": 0}
        arr = np.array(self.scores)
        return {
            "n":      len(self.scores),
            "min":    round(float(arr.min()),    4),
            "max":    round(float(arr.max()),    4),
            "mean":   round(float(arr.mean()),   4),
            "median": round(float(np.median(arr)), 4),
            "std":    round(float(arr.std()),    4),
        }


# ---------------------------------------------------------------------------
# Abstract base calibrator
# ---------------------------------------------------------------------------

class BaseThresholdCalibrator(ABC):
    """
    Abstract base class for CH4Net detection threshold calibrators.

    All calibrators must implement Fit() (compute the threshold from scores)
    and PrintSummary() (report results to stdout). SaveResults() is provided
    as a concrete method on the base class since all calibrators share the
    same JSON serialisation pattern.
    """

    @abstractmethod
    def Fit(self, scores: list, alpha: float) -> float:
        """
        Compute the detection threshold from a list of non-emitter scores.

        Parameters
        ----------
        scores : list of float
            S/C scores from verified non-emitter locations.
        alpha : float
            Target FPR level (e.g. 0.10 for a 10% false-positive rate bound).

        Return
        ------
        float — calibrated threshold tau.
        """

    @abstractmethod
    def PrintSummary(self, results: dict) -> None:
        """
        Print a human-readable summary of calibration results to stdout.

        Parameters
        ----------
        results : dict
            Output dict produced by the calibrator's main run method.

        Return
        ------
        None
        """

    def SaveResults(self, results: dict, path: Path = OUTPUT_PATH) -> None:
        """
        Serialise calibration results to JSON.

        Parameters
        ----------
        results : dict
            Calibration output dict.
        path : Path, optional
            Output file path. Defaults to OUTPUT_PATH.

        Return
        ------
        None
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(results, fh, indent=2)
        log.info("Calibration results written -> %s", path)


# ---------------------------------------------------------------------------
# Primary conformal calibrator
# ---------------------------------------------------------------------------

class ConformalCalibrator(BaseThresholdCalibrator):
    """
    Split conformal prediction calibrator (Angelopoulos & Bates 2021).

    Computes the finite-sample conformal threshold tau as the
    ceil((n+1)(1-alpha))-th order statistic of the non-emitter S/C score
    distribution. Guarantees P(FPR <= alpha) for any exchangeable test set.

    Also provides Bootstrap() for sampling uncertainty on tau and
    SampleSizeAnalysis() for calibration-set adequacy reporting.

    References
    ----------
    Angelopoulos & Bates (2021) arXiv:2107.07511, Theorem 2.
    """

    def Fit(self, scores: list, alpha: float = 0.10) -> float:
        """
        Compute the split-conformal threshold at level (1 - alpha).

        The threshold is the ceil((n+1)(1-alpha))-th order statistic of scores,
        clamped to the n-th element to avoid extrapolation beyond the observed
        range. For very small n (n < 1/alpha) the threshold equals the maximum
        score, giving a conservative but valid bound.

        Parameters
        ----------
        scores : list of float
            Non-emitter S/C scores.
        alpha : float
            Target FPR level (default 0.10).

        Return
        ------
        float — calibrated tau.

        Raises
        ------
        ValueError if scores is empty.
        """
        n = len(scores)
        if n == 0:
            raise ValueError("Empty score set — cannot compute conformal quantile.")
        rank = min(math.ceil((n + 1) * (1 - alpha)), n)
        return float(sorted(scores)[rank - 1])

    def EmpiricalFpr(self, scores: list, threshold: float) -> float:
        """
        Compute the fraction of non-emitter scores that exceed threshold.

        Parameters
        ----------
        scores : list of float
            Non-emitter S/C scores.
        threshold : float
            Threshold value to evaluate.

        Return
        ------
        float in [0, 1], or NaN if scores is empty.
        """
        if not scores:
            return float("nan")
        return sum(1 for s in scores if s > threshold) / len(scores)

    def Bootstrap(
        self,
        scores: list,
        alpha: float,
        n_boot: int = BOOTSTRAP_N_RESAMPLES,
        seed: int   = BOOTSTRAP_SEED,
    ) -> dict:
        """
        Bootstrap confidence interval on tau to quantify finite-set sampling uncertainty.

        Resamples the calibration set with replacement n_boot times, computes
        tau on each resample, and returns the distribution statistics. This is
        NOT a formal conformal guarantee — it quantifies how much tau would vary
        if a different n-element sample were drawn from the same population.

        Parameters
        ----------
        scores : list of float
            Non-emitter S/C scores.
        alpha : float
            FPR level at which tau is computed.
        n_boot : int
            Number of bootstrap resamples (default 2,000).
        seed : int
            Random seed for reproducibility (default 42).

        Return
        ------
        dict with tau_boot_mean, tau_boot_std, tau_boot_p5, tau_boot_p95, n_bootstrap.
        """
        rng = np.random.default_rng(seed)
        scores_arr = np.array(scores)
        boot_taus  = []
        for _ in range(n_boot):
            resample = rng.choice(scores_arr, size=len(scores), replace=True)
            try:
                boot_taus.append(self.Fit(list(resample), alpha))
            except ValueError:
                pass
        boot_arr = np.array(boot_taus)
        return {
            "tau_boot_mean": round(float(np.mean(boot_arr)),              4),
            "tau_boot_std":  round(float(np.std(boot_arr, ddof=1)),       4),
            "tau_boot_p5":   round(float(np.percentile(boot_arr,  5)),    4),
            "tau_boot_p95":  round(float(np.percentile(boot_arr, 95)),    4),
            "n_bootstrap":   n_boot,
        }

    def SampleSizeAnalysis(self, n: int, alpha: float) -> dict:
        """
        Report on calibration-set adequacy for the desired FPR level.

        The conformal guarantee is exact for any n >= 1, but the threshold
        equals the sample maximum when n < ceil(1/alpha - 1), making it a
        loose (conservative) bound. Standard guidance: n >= 1/alpha - 1 for
        the guarantee to be practically informative.

        Parameters
        ----------
        n : int
            Size of the calibration set.
        alpha : float
            Target FPR level.

        Return
        ------
        dict with n_calibration, alpha, conformal_rank, min_n_nontrivial,
        n_adequate, and an explanatory note.
        """
        min_n = math.ceil(1 / alpha - 1)
        rank  = math.ceil((n + 1) * (1 - alpha))
        return {
            "n_calibration":    n,
            "alpha":            alpha,
            "conformal_rank":   rank,
            "min_n_nontrivial": min_n,
            "n_adequate":       n >= min_n,
            "note": (
                "Conformal guarantee is valid for any n >= 1 (finite-sample exact). "
                "For n < min_n_nontrivial the threshold equals the sample maximum, "
                "giving a valid but conservative bound."
            ),
        }

    def Run(
        self, loader: NonEmitterScoreLoader, alpha: float = 0.10
    ) -> dict:
        """
        Execute the full calibration workflow and return a JSON-serialisable results dict.

        Computes global conformal thresholds at all ALPHA_LEVELS, bootstraps the
        primary threshold, and assembles the full output payload.

        Parameters
        ----------
        loader : NonEmitterScoreLoader
            Loaded and validated score data (call loader.Load() first).
        alpha : float
            Primary FPR level for the recommendation (default 0.10).

        Return
        ------
        dict suitable for JSON serialisation and SaveResults().
        """
        scores  = loader.scores
        records = loader.records

        global_thresholds = {}
        for a in ALPHA_LEVELS:
            tau = self.Fit(scores, a)
            global_thresholds[a] = round(tau, 4)
            log.info(
                "  tau(alpha=%.2f) = %.4f   empirical FPR at production=%.3f  (n=%d, rank=%d)",
                a, tau, self.EmpiricalFpr(scores, PRODUCTION_THRESHOLD),
                len(scores), math.ceil((len(scores) + 1) * (1 - a)),
            )

        primary_tau = global_thresholds[alpha]
        prod_fpr    = self.EmpiricalFpr(scores, PRODUCTION_THRESHOLD)
        calib_fpr   = self.EmpiricalFpr(scores, primary_tau)

        log.info("Computing bootstrap CI on tau (alpha=%.2f)...", alpha)
        boot = self.Bootstrap(scores, alpha)

        results = {
            "schema_version":  "2.0.0",
            "method":          "split_conformal_prediction",
            "reference":       "Angelopoulos & Bates (2021) arXiv:2107.07511",
            "n_calibration":   len(scores),
            "all_scores":      [round(s, 4) for s in sorted(scores)],
            "score_stats":     loader.Describe(),

            "global_thresholds": {
                f"tau_alpha_{int(a*100):02d}": {
                    "alpha":                a,
                    "tau":                  t,
                    "guarantee":            f"FPR <= {a:.0%} finite-sample guarantee",
                    "empirical_fpr":        round(self.EmpiricalFpr(scores, t), 4),
                }
                for a, t in global_thresholds.items()
            },

            "primary_threshold": {
                "alpha":                       alpha,
                "tau":                         primary_tau,
                "production_threshold":        PRODUCTION_THRESHOLD,
                "empirical_fpr_at_production": round(prod_fpr,  4),
                "empirical_fpr_at_calibrated": round(calib_fpr, 4),
                "recommended_action": (
                    "Replace production threshold with tau_calibrated"
                    if primary_tau > PRODUCTION_THRESHOLD
                    else "Production threshold is already conservative"
                ),
            },

            "bootstrap_ci":          boot,
            "sample_size_analysis":  self.SampleSizeAnalysis(len(scores), alpha),

            "interpretation": (
                f"On n={len(scores)} non-emitter locations, S/C scores ranged from "
                f"{min(scores):.4f} to {max(scores):.4f}. "
                f"The production threshold {PRODUCTION_THRESHOLD} gives empirical "
                f"FPR {prod_fpr:.1%}. "
                f"The conformal threshold at alpha={alpha} is tau={primary_tau:.4f}, "
                f"guaranteeing FPR <= {alpha:.0%} on exchangeable test scenes."
            ),
        }
        return results

    def PrintSummary(self, results: dict) -> None:
        """
        Print a formatted calibration summary table to stdout.

        Parameters
        ----------
        results : dict
            Output dict returned by Run().

        Return
        ------
        None
        """
        pt    = results["primary_threshold"]
        boot  = results["bootstrap_ci"]
        sa    = results["sample_size_analysis"]
        scores_all = results["all_scores"]

        bar = "=" * 70
        print(f"\n{bar}")
        print("  CH4Net v8 — Conformal Threshold Calibration")
        print(bar)
        print(f"  Calibration set : n={results['n_calibration']} non-emitter locations")
        print(f"  Score range     : [{min(scores_all):.4f}, {max(scores_all):.4f}]")
        print(f"  Score mean/med  : {results['score_stats']['mean']:.4f} / "
              f"{results['score_stats']['median']:.4f}")
        print()
        print(f"  Production threshold : tau_prod = {PRODUCTION_THRESHOLD}")
        print(f"  Empirical FPR at tau_prod : {pt['empirical_fpr_at_production']:.1%}")
        print()
        print("  Conformal thresholds (finite-sample guaranteed FPR bound):")
        for key, info in results["global_thresholds"].items():
            star = " <- primary" if abs(info["alpha"] - pt["alpha"]) < 1e-6 else ""
            print(f"    alpha={info['alpha']:.2f}  tau={info['tau']:.4f}"
                  f"  (FPR <= {info['alpha']:.0%}){star}")
        print()
        print(f"  Bootstrap 90% CI on tau(alpha={pt['alpha']}) : "
              f"[{boot['tau_boot_p5']:.4f}, {boot['tau_boot_p95']:.4f}]")
        print(f"  Bootstrap mean : {boot['tau_boot_mean']:.4f}  "
              f"std : {boot['tau_boot_std']:.4f}")
        print()
        status = "adequate" if sa["n_adequate"] else "marginal"
        print(f"  Sample size : n={sa['n_calibration']}  "
              f"(min for non-trivial bound at alpha={pt['alpha']}: "
              f"n>={sa['min_n_nontrivial']}  [{status}])")
        print(bar)


# ---------------------------------------------------------------------------
# Mondrian (stratified) conformal calibrator
# ---------------------------------------------------------------------------

class MondrianConformalCalibrator(ConformalCalibrator):
    """
    Mondrian conformal prediction: stratified thresholds per group.

    Extends ConformalCalibrator by applying Fit() independently to each
    stratum (e.g. ecoregion or CLC land-cover class). Groups with fewer than
    SMALL_N_WARNING_THRESHOLD records receive a small_n_warning flag.

    References
    ----------
    Shafer & Vovk (2008) "A Tutorial on Conformal Prediction". JMLR 9.
    Venn prediction framework for per-category guarantees.
    """

    def FitByGroup(
        self,
        records: list,
        alpha: float,
        group_key: str = "ecoregion",
    ) -> dict:
        """
        Compute conformal thresholds independently for each group in records.

        Parameters
        ----------
        records : list of dict
            Per-record metadata dicts (must have group_key and sc_cfar/sc_ratio fields).
        alpha : float
            Target FPR level applied within each group.
        group_key : str
            Field name to group by (default 'ecoregion'). Also supports 'clc_class'.

        Return
        ------
        dict mapping group name -> calibration result dict with keys:
            tau, n, scores, min_sc, max_sc, mean_sc,
            empirical_fpr_at_production, small_n_warning.
        """
        groups = defaultdict(list)
        for r in records:
            key = r.get(group_key, "unknown")
            sc  = r.get("sc_cfar") or r.get("sc_ratio")
            if sc is not None:
                groups[key].append(float(sc))

        result = {}
        for group, scores in groups.items():
            n = len(scores)
            small_n = n < SMALL_N_WARNING_THRESHOLD
            try:
                tau = self.Fit(scores, alpha)
            except ValueError:
                tau = max(scores) if scores else float("nan")
            result[group] = {
                "tau":                         round(tau, 4),
                "n":                           n,
                "scores":                      [round(s, 4) for s in sorted(scores)],
                "min_sc":                      round(min(scores), 4),
                "max_sc":                      round(max(scores), 4),
                "mean_sc":                     round(float(np.mean(scores)), 4),
                "empirical_fpr_at_production": round(
                    self.EmpiricalFpr(scores, PRODUCTION_THRESHOLD), 4
                ),
                "small_n_warning": small_n,
            }
            if small_n:
                log.warning(
                    "  SMALL_N: group '%s' has only %d samples (recommend >= %d)",
                    group, n, SMALL_N_WARNING_THRESHOLD,
                )
        return result

    def Run(self, loader: NonEmitterScoreLoader, alpha: float = 0.10) -> dict:
        """
        Execute full calibration including global and Mondrian thresholds.

        Calls the parent ConformalCalibrator.Run() for the global result,
        then appends per-ecoregion and per-CLC-class Mondrian thresholds.

        Parameters
        ----------
        loader : NonEmitterScoreLoader
            Loaded and validated score data.
        alpha : float
            Primary FPR level (default 0.10).

        Return
        ------
        dict — global results dict augmented with mondrian_by_ecoregion
        and mondrian_by_clc_class keys.
        """
        results = super().Run(loader, alpha)

        log.info("Computing Mondrian thresholds by ecoregion and CLC class...")
        results["mondrian_by_ecoregion"] = self.FitByGroup(
            loader.records, alpha, group_key="ecoregion"
        )
        results["mondrian_by_clc_class"] = self.FitByGroup(
            loader.records, alpha, group_key="clc_class"
        )
        return results

    def PrintSummary(self, results: dict) -> None:
        """
        Print summary including global thresholds and Mondrian per-group results.

        Parameters
        ----------
        results : dict
            Output dict returned by this class's Run().

        Return
        ------
        None
        """
        super().PrintSummary(results)

        if "mondrian_by_ecoregion" in results:
            print("\n  Mondrian thresholds by ecoregion:")
            for eco, info in sorted(results["mondrian_by_ecoregion"].items()):
                warn = "  [SMALL_N]" if info["small_n_warning"] else ""
                print(f"    {eco:<22}  n={info['n']}  tau={info['tau']:.4f}{warn}")

        if "mondrian_by_clc_class" in results:
            print("\n  Mondrian thresholds by CLC land-cover class:")
            for clc, info in sorted(results["mondrian_by_clc_class"].items()):
                warn = "  [SMALL_N]" if info["small_n_warning"] else ""
                print(f"    {clc:<22}  n={info['n']}  tau={info['tau']:.4f}{warn}")
        print()


# ---------------------------------------------------------------------------
# Optional calibration plot
# ---------------------------------------------------------------------------

def make_calibration_plot(
    scores: list,
    global_thresholds: dict,
    records: list,
    out_path: Path,
) -> None:
    """
    Generate a two-panel calibration summary plot and save to out_path.

    Left panel : ECDF of non-emitter S/C scores with conformal and production thresholds.
    Right panel: Per-ecoregion dot plot of S/C scores.

    Parameters
    ----------
    scores : list of float
        Non-emitter S/C scores.
    global_thresholds : dict
        Mapping of alpha -> tau from ConformalCalibrator.Run().
    records : list of dict
        Per-record metadata (must have ecoregion and sc_cfar/sc_ratio fields).
    out_path : Path
        Output file path for the PNG.

    Return
    ------
    None
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed — skipping calibration plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("CH4Net v8 — Conformal Threshold Calibration", fontsize=13)

    # Left: ECDF
    ax = axes[0]
    sorted_sc = sorted(scores)
    n = len(sorted_sc)
    ecdf_y = [(i + 1) / n for i in range(n)]
    ax.step(sorted_sc, ecdf_y, where="post", lw=2, color="#2066a8",
            label="ECDF (non-emitters)")
    ax.axvline(PRODUCTION_THRESHOLD, color="red", ls="--", lw=1.5,
               label=f"Production tau={PRODUCTION_THRESHOLD}")
    colors = ["#1a9641", "#f97f0f", "#d7191c"]
    for (alpha, tau), color in zip(sorted(global_thresholds.items()), colors):
        ax.axvline(tau, color=color, ls=":", lw=1.5,
                   label=f"Conformal tau (alpha={alpha})={tau:.3f}")
    ax.set_xlabel("S/C ratio")
    ax.set_ylabel("ECDF")
    ax.set_title("Non-emitter S/C score distribution")
    ax.legend(fontsize=8)
    ax.set_xlim(left=0)
    ax.grid(alpha=0.3)

    # Right: per-ecoregion dot plot
    ax = axes[1]
    eco_scores = defaultdict(list)
    for r in records:
        eco = r.get("ecoregion", "unknown")
        sc  = r.get("sc_cfar") or r.get("sc_ratio")
        if sc is not None:
            eco_scores[eco].append(float(sc))

    ecoregions = sorted(eco_scores.keys())
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, eco in enumerate(ecoregions):
        y = i
        ax.scatter(eco_scores[eco], [y] * len(eco_scores[eco]),
                   color=palette[i % len(palette)], s=60, zorder=3, alpha=0.8)
    ax.axvline(PRODUCTION_THRESHOLD, color="red", ls="--", lw=1.5,
               label=f"tau_prod={PRODUCTION_THRESHOLD}")
    tau_10 = global_thresholds.get(0.10)
    if tau_10:
        ax.axvline(tau_10, color="#f97f0f", ls=":", lw=1.5,
                   label=f"tau_conf(alpha=0.10)={tau_10:.3f}")
    ax.set_yticks(list(range(len(ecoregions))))
    ax.set_yticklabels(ecoregions, fontsize=9)
    ax.set_xlabel("S/C ratio")
    ax.set_title("S/C scores by ecoregion")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Plot saved -> %s", out_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Command-line entry point. Instantiates NonEmitterScoreLoader and
    MondrianConformalCalibrator, runs the full calibration, prints the
    summary, saves JSON output, and optionally generates a plot.

    Return
    ------
    None
    """
    parser = argparse.ArgumentParser(
        description="Conformal threshold calibration from non-emitter S/C scores"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.10,
        help="Primary FPR level (default: 0.10)"
    )
    parser.add_argument(
        "--scores", type=Path, default=SCORES_PATH,
        help=f"Input scores JSON (default: {SCORES_PATH})"
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_PATH,
        help=f"Output calibration JSON (default: {OUTPUT_PATH})"
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip matplotlib calibration plot"
    )
    args = parser.parse_args()

    loader = NonEmitterScoreLoader(path=args.scores)
    loader.Load()

    calibrator = MondrianConformalCalibrator()
    results    = calibrator.Run(loader, alpha=args.alpha)

    calibrator.PrintSummary(results)
    calibrator.SaveResults(results, path=args.output)

    if not args.no_plot:
        thresholds = {
            info["alpha"]: info["tau"]
            for info in results["global_thresholds"].values()
        }
        make_calibration_plot(loader.scores, thresholds, loader.records, PLOT_PATH)
        if PLOT_PATH.exists():
            print(f"  Plot saved: {PLOT_PATH}")


if __name__ == "__main__":
    main()
