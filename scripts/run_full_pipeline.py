#!/usr/bin/env python3
"""
run_full_pipeline.py
====================
End-to-end demo: satellite detection → stress test → bank-level financed emissions.

This is the "money slide" for Ali Hirsa and the Bloomberg BBQ presentation:
  1. Satellite detects methane at Neurath, Belchatow, etc.
  2. Monte Carlo stress test projects carbon liability under NGFS scenarios
  3. Entity resolution maps emitters to RWE, PGE equities
  4. Credit exposure model attributes emissions to bank loan books
  5. Output: "Deutsche Bank faces €X in financed carbon liability under disorderly transition"

Usage:
  python scripts/run_full_pipeline.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stress_testing.stress_test import StressTestEngine
from src.stress_testing.scenarios import SCENARIOS
from src.entity_resolution.credit_exposure import CreditExposureModel


def main():
    print("=" * 85)
    print("CH₄Net v2 — FULL PIPELINE DEMO")
    print("Satellite Detection → Climate Stress Test → Bank Credit Exposure")
    print("=" * 85)

    # ── Step 1: Run stress test ──────────────────────────────────────────
    print("\n[1/3] Running Monte Carlo stress test (3 NGFS scenarios × 50K paths)...")
    engine = StressTestEngine()
    tickers = ["RWE.DE", "PGE.WA"]

    result = engine.run_portfolio_stress(
        tickers=tickers,
        scenarios=["orderly", "disorderly", "hot_house"],
        horizon_years=10,
        n_paths=50_000,
        seed=42,
    )

    # Print stress test summary
    print(StressTestEngine.format_summary(result))

    # ── Step 2: Extract issuer-level emissions per scenario ──────────────
    print("\n[2/3] Computing bank-level financed emissions (PCAF methodology)...")
    credit_model = CreditExposureModel()

    for scenario in ["orderly", "disorderly", "hot_house"]:
        # Get terminal-year emissions per issuer under this scenario
        issuer_emissions = {}
        for ir in result.issuer_results[scenario]:
            # Use VaR95 as the stress emission level (conservative)
            # Convert EUR liability back to tCO2e for PCAF calculation
            sc = SCENARIOS[scenario]
            terminal_year = 2024 + result.horizon_years
            terminal_price = sc.price_path.get(
                terminal_year, sc.price_path.get(max(sc.price_path.keys()), 65)
            )
            if terminal_price > 0:
                issuer_emissions[ir.ticker] = ir.terminal_cvar95_eur / terminal_price
            else:
                issuer_emissions[ir.ticker] = 0.0

        # Print bank-level attribution
        print(credit_model.format_bank_summary(
            issuer_emissions, ets_price_eur=terminal_price, scenario=scenario
        ))

    # ── Step 3: Save combined results ────────────────────────────────────
    print("\n[3/3] Saving combined results...")

    # Disorderly scenario detail for JSON output
    disorderly_emissions = {}
    for ir in result.issuer_results["disorderly"]:
        sc = SCENARIOS["disorderly"]
        terminal_price = sc.price_path.get(2034, 195.0)
        disorderly_emissions[ir.ticker] = ir.terminal_cvar95_eur / terminal_price

    bank_summary = credit_model.bank_level_summary(
        disorderly_emissions, ets_price_eur=195.0, scenario="disorderly"
    )

    # Strip non-serializable objects
    output = {
        "pipeline": "CH4Net v2 → NGFS Stress Test → PCAF Financed Emissions",
        "tickers": tickers,
        "scenarios": result.scenarios_run,
        "horizon_years": result.horizon_years,
        "n_paths": result.n_paths,
        "portfolio_stress": {
            scenario: {
                "terminal_mean_eur": result.portfolio_terminal_mean_eur[scenario],
                "terminal_var95_eur": result.portfolio_terminal_var95_eur[scenario],
                "terminal_cvar95_eur": result.portfolio_terminal_cvar95_eur[scenario],
                "npv_mean_eur": result.portfolio_npv_mean_eur[scenario],
            }
            for scenario in result.scenarios_run
        },
        "bank_level_disorderly": {
            bank: {
                "total_exposure_eur": agg["total_exposure_eur"],
                "total_financed_tCO2e": agg["total_financed_tCO2e"],
                "total_carbon_liability_eur": agg["total_carbon_liability_eur"],
                "n_issuers": agg["n_issuers"],
            }
            for bank, agg in bank_summary.items()
        },
    }

    output_path = Path("results_analysis/full_pipeline_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved → {output_path}")
    print(f"\nKey finding (disorderly scenario, terminal year):")
    print(f"  Portfolio CVaR95: €{result.portfolio_terminal_cvar95_eur['disorderly']:,.0f}/yr")
    for bank, agg in sorted(bank_summary.items(), key=lambda x: -x[1]["total_carbon_liability_eur"]):
        print(f"  {bank}: €{agg['total_carbon_liability_eur']:,.0f} financed carbon liability")


if __name__ == "__main__":
    main()
