#!/usr/bin/env python3
"""
run_stress_test.py
==================
Execute the climate stress test on the detected emitter portfolio.

Usage:
  python scripts/run_stress_test.py
  python scripts/run_stress_test.py --tickers RWE.DE PGE.WA --horizon 10 --paths 50000
  python scripts/run_stress_test.py --all-detected   # auto-detect tickers from results
"""

import argparse
import json
import sys
from pathlib import Path
from dataclasses import asdict

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stress_testing.stress_test import StressTestEngine
from src.api.risk_model import TICKER_SITES


def main():
    parser = argparse.ArgumentParser(description="Run climate stress test on emitter portfolio")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Equity tickers to stress (e.g. RWE.DE PGE.WA)")
    parser.add_argument("--all-detected", action="store_true",
                        help="Auto-include all tickers with satellite detections")
    parser.add_argument("--horizon", type=int, default=10, help="Projection horizon (years)")
    parser.add_argument("--paths", type=int, default=50_000, help="Monte Carlo paths per scenario")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="results_analysis/stress_test_results.json",
                        help="Output JSON path")
    args = parser.parse_args()

    # Determine tickers
    if args.tickers:
        tickers = args.tickers
    elif args.all_detected:
        # Use all tickers that have at least one mapped site
        tickers = list(TICKER_SITES.keys())
    else:
        # Default: the main portfolio of interest
        tickers = ["RWE.DE", "PGE.WA", "UN01.DE", "SHEL.L"]

    print(f"\n{'=' * 70}")
    print(f"CLIMATE STRESS TEST — CH₄Net Satellite Emission Data")
    print(f"{'=' * 70}")
    print(f"Tickers:  {', '.join(tickers)}")
    print(f"Horizon:  {args.horizon} years")
    print(f"MC paths: {args.paths:,}")
    print(f"Seed:     {args.seed}")
    print(f"{'=' * 70}\n")

    # Run engine
    engine = StressTestEngine()
    result = engine.run_portfolio_stress(
        tickers=tickers,
        horizon_years=args.horizon,
        n_paths=args.paths,
        seed=args.seed,
    )

    # Print summary
    print(StressTestEngine.format_summary(result))

    # Save JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to serializable dict
    def _to_dict(obj):
        if hasattr(obj, "__dataclass_fields__"):
            return {k: _to_dict(v) for k, v in asdict(obj).items()}
        elif isinstance(obj, dict):
            return {k: _to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_to_dict(v) for v in obj]
        elif isinstance(obj, (float, int)):
            if isinstance(obj, float) and (obj != obj):  # NaN check
                return None
            return round(obj, 2) if isinstance(obj, float) else obj
        return obj

    with open(output_path, "w") as f:
        json.dump(_to_dict(result), f, indent=2)

    print(f"\nResults saved → {output_path}")
    print(f"  Portfolio VaR95 (disorderly): €{result.portfolio_terminal_var95_eur.get('disorderly', 0):,.0f}/yr")
    print(f"  Portfolio CVaR95 (disorderly): €{result.portfolio_terminal_cvar95_eur.get('disorderly', 0):,.0f}/yr")


if __name__ == "__main__":
    main()
