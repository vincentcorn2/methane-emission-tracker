"""
run_pipeline.py
===============
Single entry point for the complete CH4Net methane detection and quantification
pipeline. Executes all stages in sequence from raw Sentinel-2 to FastAPI-ready
risk scores and stress-test results.

This is the canonical way to reproduce the full analysis from scratch.

Usage
-----
    # Full pipeline (all sites, all stages)
    python run_pipeline.py

    # Specific sites only
    python run_pipeline.py --sites belchatow neurath maasvlakte

    # Skip stages (e.g. detection already done)
    python run_pipeline.py --skip-detection --skip-tropomi

    # Dry run — print plan but don't execute
    python run_pipeline.py --dry-run

Pipeline Stages
---------------
    Stage 1: Validate multi-date S/C ratios   (validate_multidate.py)
    Stage 2: TROPOMI cross-validation         (validate_tropomi.py)
    Stage 3: CEMF+IME+ERA5 quantification     (scripts/run_quantification.py)
    Stage 4: Risk model scoring               (src.api.risk_model.RiskModel)
    Stage 5: Portfolio stress test            (src.stress_testing.stress_test)
    Stage 6: Model validation report          (src.validation.model_validation)

Outputs
-------
    results_analysis/multidate_validation.json    (Stage 1)
    results_analysis/tropomi_validation.json      (Stage 2)
    results_analysis/quantification.json          (Stage 3)
    results_analysis/stress_test_results.json     (Stage 5)
    results_analysis/model_validation_report.json (Stage 6)

Environment
-----------
    CDS_API_KEY   — Copernicus CDS API key (required for Stage 2; ERA5 wind)
    COPERNICUS_CLIENT_ID / COPERNICUS_CLIENT_SECRET — for Stage 1 tile downloads
"""
import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_pipeline")

# ── Default portfolio for stress test ────────────────────────────────────────
DEFAULT_TICKERS = ["RWE.DE", "PGE.WA", "SHEL.L", "UN01.DE"]

# ── Stage control ─────────────────────────────────────────────────────────────

STAGES = [
    "multidate",    # S/C detection across all dates
    "tropomi",      # TROPOMI cross-validation
    "quantify",     # CEMF+IME+ERA5 quantification
    "risk",         # Risk model scoring
    "stress",       # Portfolio Monte Carlo stress test
    "validate",     # Model validation report (AUC, calibration, Kupiec, tornado)
]


def run_stage(name: str, cmd: list[str], dry_run: bool = False) -> bool:
    """Run one pipeline stage. Returns True on success."""
    logger.info("═══ Stage: %s ═══", name.upper())
    logger.info("Command: %s", " ".join(cmd))
    if dry_run:
        logger.info("[DRY RUN] Skipping execution")
        return True
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        logger.error("Stage '%s' failed with return code %d", name, result.returncode)
        return False
    logger.info("Stage '%s' completed successfully", name)
    return True


def stage_quantify(sites: list[str], dry_run: bool, no_era5: bool) -> bool:
    """Stage 3: CEMF+IME+ERA5 quantification via canonical runner."""
    cmd = [sys.executable, "scripts/run_quantification.py"]
    if sites:
        cmd += ["--sites"] + sites
    if dry_run:
        cmd.append("--dry-run")
    if no_era5:
        cmd.append("--no-era5")
    return run_stage("quantify", cmd, dry_run=False)  # run_quantification handles --dry-run itself


def stage_risk_and_stress(
    portfolio: list[str],
    dry_run: bool,
    out_dir: Path,
) -> bool:
    """Stages 4+5: Risk scoring and portfolio stress test."""
    logger.info("═══ Stage: RISK + STRESS ═══")
    if dry_run:
        logger.info("[DRY RUN] Skipping risk + stress")
        return True

    try:
        import logging as _log
        _log.disable(_log.CRITICAL)
        from src.api.risk_model import RiskModel, SITE_OPERATOR_MAP
        from src.stress_testing.stress_test import StressTestEngine

        model = RiskModel()
        all_sites = list(SITE_OPERATOR_MAP.keys())
        scores = {site: model.site_risk(site) for site in all_sites}

        engine = StressTestEngine()
        stress_result = engine.run_portfolio_stress(portfolio)

        _log.disable(_log.NOTSET)

        # Write stress test results
        out = {
            "meta": {"pipeline": "run_pipeline.py", "portfolio": portfolio},
            "portfolio_npv_mean_eur": stress_result.portfolio_npv_mean_eur,
            "portfolio_terminal_mean_eur": stress_result.portfolio_terminal_mean_eur,
            "portfolio_terminal_var95_eur": stress_result.portfolio_terminal_var95_eur,
            "portfolio_terminal_cvar95_eur": stress_result.portfolio_terminal_cvar95_eur,
            "scenarios_run": stress_result.scenarios_run,
            "n_paths": stress_result.n_paths,
        }
        stress_path = out_dir / "stress_test_results.json"
        with open(stress_path, "w") as f:
            json.dump(out, f, indent=2, default=str)
        logger.info("Stress test results → %s", stress_path)

        # Print per-site risk summary
        logger.info("─── Per-site risk summary ───")
        for site, score in scores.items():
            tier = score.get("risk_tier", "?")
            eur = score.get("carbon_liability_eur")
            eur_str = f"€{eur:,.0f}" if eur else "N/A (excluded or no data)"
            logger.info("  %-18s  %-10s  %s", site, tier, eur_str)

        return True

    except Exception as exc:
        logger.error("Risk+stress stage failed: %s", exc, exc_info=True)
        return False


def stage_model_validation(dry_run: bool, out_dir: Path) -> bool:
    """Stage 6: Full model validation report."""
    logger.info("═══ Stage: MODEL VALIDATION ═══")
    if dry_run:
        logger.info("[DRY RUN] Skipping validation")
        return True

    try:
        import logging as _log
        _log.disable(_log.CRITICAL)
        from src.validation.model_validation import ModelValidator
        _log.disable(_log.NOTSET)

        validator = ModelValidator()
        report = validator.full_validation_report()

        kupiec = validator.kupiec_test()
        iso = validator.isotonic_calibration()

        out = {
            "auc": report.auc,
            "auc_delong_ci": report.auc_ci,
            "brier_score": report.brier_score,
            "ece": report.ece,
            "hosmer_lemeshow_pvalue": report.hosmer_lemeshow_pvalue,
            "kupiec_test": kupiec,
            "isotonic_calibration": iso,
        }
        val_path = out_dir / "model_validation_report.json"
        with open(val_path, "w") as f:
            json.dump(out, f, indent=2, default=str)
        logger.info("Validation report → %s", val_path)
        logger.info("AUC=%.3f [%.3f, %.3f]  Brier=%.3f  ECE=%.3f",
                    report.auc, *(report.auc_ci or (0, 1)), report.brier_score, report.ece)
        logger.info("Kupiec H0 %s (p=%.3f)",
                    "REJECTED" if kupiec.get("reject_h0") else "NOT rejected",
                    kupiec.get("p_value", float("nan")))
        return True

    except Exception as exc:
        logger.error("Model validation stage failed: %s", exc, exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(description="CH4Net full pipeline runner")
    parser.add_argument("--sites", nargs="*", help="Site slugs to process (default: all)")
    parser.add_argument("--portfolio", nargs="*", default=DEFAULT_TICKERS,
                        help="Tickers for stress test (default: RWE.DE PGE.WA SHEL.L UN01.DE)")
    parser.add_argument("--skip-multidate", action="store_true")
    parser.add_argument("--skip-tropomi", action="store_true")
    parser.add_argument("--skip-quantify", action="store_true")
    parser.add_argument("--skip-stress", action="store_true")
    parser.add_argument("--skip-validate", action="store_true")
    parser.add_argument("--no-era5", action="store_true",
                        help="Use fallback wind instead of ERA5 (for offline runs)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands but don't execute")
    parser.add_argument("--out-dir", default="results_analysis",
                        help="Output directory (default: results_analysis)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    success = True
    failed_stages = []

    # Stage 1: multi-date detection
    if not args.skip_multidate:
        cmd = [sys.executable, "validate_multidate.py"]
        if args.sites:
            cmd += ["--sites"] + args.sites
        if not run_stage("multidate", cmd, args.dry_run):
            failed_stages.append("multidate")
            success = False

    # Stage 2: TROPOMI cross-validation
    if not args.skip_tropomi:
        cmd = [sys.executable, "validate_tropomi.py", "--no-download"]
        if args.sites:
            cmd += ["--sites"] + args.sites
        if not run_stage("tropomi", cmd, args.dry_run):
            failed_stages.append("tropomi")
            # Non-fatal: TROPOMI data may be unavailable offline

    # Stage 3: CEMF+IME+ERA5 quantification
    if not args.skip_quantify:
        if not stage_quantify(args.sites, args.dry_run, args.no_era5):
            failed_stages.append("quantify")
            success = False

    # Stages 4+5: risk + stress
    if not args.skip_stress:
        if not stage_risk_and_stress(args.portfolio, args.dry_run, out_dir):
            failed_stages.append("stress")
            success = False

    # Stage 6: model validation
    if not args.skip_validate:
        if not stage_model_validation(args.dry_run, out_dir):
            failed_stages.append("validate")
            # Non-fatal: validation data may be sparse

    # Summary
    logger.info("")
    logger.info("═══ Pipeline %s ═══", "COMPLETE" if success else "COMPLETED WITH ERRORS")
    if failed_stages:
        logger.error("Failed stages: %s", ", ".join(failed_stages))
        sys.exit(1)
    else:
        logger.info("All stages succeeded. Results in: %s/", args.out_dir)


if __name__ == "__main__":
    main()
