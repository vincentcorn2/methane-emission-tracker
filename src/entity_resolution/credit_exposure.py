"""
credit_exposure.py
==================
Maps corporate issuers to bank loan exposure for ECB/EIB climate stress testing.

This is the critical "last mile" of entity resolution:
  Satellite pixel → emitter → company → **bank loan book** → supervisory risk

The chain:
  1. Ticker/LEI → outstanding bond ISINs (from GLEIF-ANNA or Bloomberg)
  2. Bond ISINs → holders/underwriters (SEC 13-F, Bundesbank SHS, ECB AnaCredit)
  3. Company → syndicated loan exposure (Refinitiv LPC / Bloomberg LEID)
  4. Aggregate: total financed emissions = Σ (attribution_factor × site_emissions)

For the ECB, the key regulatory datasets are:
  - AnaCredit: loan-level credit data for euro area banks
  - Securities Holdings Statistics (SHS): bond holdings by bank
  - EBA transparency exercise: bank-level exposure by sector/country

This module provides a simplified but structurally correct version using
publicly available data. A production deployment would integrate directly
with the ECB's SHS/AnaCredit pipelines.

Data model for bank exposure:
  An "exposure" is a (bank, issuer, instrument_type, amount_eur) tuple.
  The attribution factor = exposure_amount / issuer_total_debt.
  Financed emissions = attribution_factor × issuer_annual_tCO2e.

References:
  - ECB AnaCredit Regulation (EU) 2016/867
  - PCAF (2022) "The Global GHG Accounting and Reporting Standard"
  - Battiston et al. (2017) "A climate stress-test of the financial system"
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class BondIssue:
    """A single bond or debt instrument issued by a corporate entity."""
    isin: str
    issuer_lei: Optional[str]
    issuer_name: str
    currency: str
    outstanding_eur: float        # face value in EUR
    maturity_year: Optional[int]
    coupon_pct: Optional[float]
    seniority: str = "senior_unsecured"  # senior_secured / senior_unsecured / subordinated


@dataclass
class LoanExposure:
    """A bank's loan exposure to a corporate borrower."""
    bank_name: str
    bank_lei: Optional[str]
    borrower_name: str
    borrower_lei: Optional[str]
    borrower_ticker: Optional[str]
    instrument_type: str           # "term_loan" / "revolver" / "project_finance" / "bond_holding"
    exposure_eur: float            # current exposure amount in EUR
    maturity_year: Optional[int]
    sector: Optional[str]          # NACE Rev. 2 code
    country: str = "DE"


@dataclass
class FinancedEmission:
    """
    The final product for ECB/EIB supervisors:
    A bank's share of emissions from a satellite-detected emitter.

    PCAF attribution methodology:
      financed_emission = (exposure / total_debt) × issuer_emissions
    """
    bank_name: str
    issuer_ticker: Optional[str]
    issuer_name: str
    site_name: str

    # Exposure
    exposure_eur: float
    total_issuer_debt_eur: float
    attribution_factor: float      # exposure / total_debt

    # Emissions (from stress test engine)
    issuer_annual_tCO2e: float     # satellite-derived
    financed_tCO2e: float          # attribution_factor × issuer_annual_tCO2e

    # Financial risk
    carbon_liability_eur: float    # financed_tCO2e × ETS price
    scenario: Optional[str] = None


# ── Synthetic exposure data (for demonstration / ECB presentation) ───────────
# In production, this would be populated from AnaCredit / SHS / Bloomberg

# Major European banks with known exposure to coal/lignite utilities
# Sources: EBA transparency exercise, annual reports, syndicated loan databases
SYNTHETIC_EXPOSURES: list[LoanExposure] = [
    # RWE AG — German lignite/coal + renewables utility
    LoanExposure("Deutsche Bank", "7LTWFZYICNSX8D621K86", "RWE AG", "529900V6AXYMSXWB1T07",
                 "RWE.DE", "term_loan", 1_200_000_000, 2029, "D35.11", "DE"),
    LoanExposure("Commerzbank", "851WYGNLUQLFZBSBER06", "RWE AG", "529900V6AXYMSXWB1T07",
                 "RWE.DE", "revolver", 800_000_000, 2027, "D35.11", "DE"),
    LoanExposure("BNP Paribas", "R0MUWSFPU8MPRO8K5P83", "RWE AG", "529900V6AXYMSXWB1T07",
                 "RWE.DE", "bond_holding", 450_000_000, 2031, "D35.11", "DE"),
    LoanExposure("ING Group", "3TK20IVIUJ8J3ZU0QE75", "RWE AG", "529900V6AXYMSXWB1T07",
                 "RWE.DE", "term_loan", 600_000_000, 2028, "D35.11", "DE"),
    LoanExposure("EIB", None, "RWE AG", "529900V6AXYMSXWB1T07",
                 "RWE.DE", "project_finance", 350_000_000, 2035, "D35.11", "DE"),

    # PGE S.A. — Polish state-controlled lignite utility
    LoanExposure("PKO Bank Polski", "P4GTT6GF1W40CVIMFR43", "PGE S.A.", "259400MP67JG7BAR5E41",
                 "PGE.WA", "term_loan", 900_000_000, 2028, "D35.11", "PL"),
    LoanExposure("Bank Pekao", "5493000LKS7B3UTF7H35", "PGE S.A.", "259400MP67JG7BAR5E41",
                 "PGE.WA", "revolver", 500_000_000, 2026, "D35.11", "PL"),
    LoanExposure("EIB", None, "PGE S.A.", "259400MP67JG7BAR5E41",
                 "PGE.WA", "project_finance", 750_000_000, 2032, "D35.11", "PL"),
    LoanExposure("Deutsche Bank", "7LTWFZYICNSX8D621K86", "PGE S.A.", "259400MP67JG7BAR5E41",
                 "PGE.WA", "bond_holding", 200_000_000, 2029, "D35.11", "PL"),

    # Uniper SE — Maasvlakte (post-nationalisation, German state-owned)
    LoanExposure("KfW", None, "Uniper SE", "549300UXRTWGKVDKYY84",
                 "UN01.DE", "term_loan", 2_000_000_000, 2028, "D35.11", "DE"),
    LoanExposure("Deutsche Bank", "7LTWFZYICNSX8D621K86", "Uniper SE", "549300UXRTWGKVDKYY84",
                 "UN01.DE", "revolver", 400_000_000, 2027, "D35.11", "DE"),
]

# Approximate total debt for attribution factor calculation
ISSUER_TOTAL_DEBT_EUR: dict[str, float] = {
    "RWE.DE":  12_500_000_000,   # ~€12.5bn total financial debt (FY2023)
    "PGE.WA":  8_000_000_000,    # ~€8bn (PLN 36bn @ 4.5)
    "UN01.DE": 8_500_000_000,    # ~€8.5bn (post-bailout restructured debt)
    "SHEL.L":  65_000_000_000,   # ~€65bn
}


class CreditExposureModel:
    """
    Maps satellite-detected emissions to bank-level financed emissions.

    This is the module that makes the project relevant to ECB supervisors:
    it answers "How much of this detected methane emission is financed by
    Bank X's loan book?"

    Phase 1 (current): Uses synthetic but structurally realistic exposure data.
    Phase 2: Integrate with AnaCredit extracts and SHS data via ECB DG-MF API.
    """

    def __init__(self, exposures: Optional[list[LoanExposure]] = None):
        self._exposures = exposures or SYNTHETIC_EXPOSURES
        self._by_ticker: dict[str, list[LoanExposure]] = {}
        for exp in self._exposures:
            if exp.borrower_ticker:
                self._by_ticker.setdefault(exp.borrower_ticker, []).append(exp)

    def get_bank_exposures(self, ticker: str) -> list[LoanExposure]:
        """Get all bank exposures for a given issuer ticker."""
        return self._by_ticker.get(ticker, [])

    def get_exposures_by_bank(self, bank_name: str) -> list[LoanExposure]:
        """Get all exposures for a given bank."""
        return [e for e in self._exposures if e.bank_name == bank_name]

    def compute_financed_emissions(
        self,
        ticker: str,
        issuer_annual_tCO2e: float,
        ets_price_eur: float = 65.0,
        scenario: Optional[str] = None,
    ) -> list[FinancedEmission]:
        """
        Compute PCAF-compliant financed emissions for each bank with exposure.

        PCAF attribution:
          financed_emission_i = (exposure_i / total_debt) × issuer_emission

        This is what goes into a bank's Scope 3 Category 15 disclosure
        and what ECB supervisors use for the climate stress test.
        """
        exposures = self.get_bank_exposures(ticker)
        if not exposures:
            return []

        total_debt = ISSUER_TOTAL_DEBT_EUR.get(ticker, 1.0)
        issuer_name = exposures[0].borrower_name if exposures else "Unknown"

        results = []
        for exp in exposures:
            attribution = exp.exposure_eur / total_debt
            financed_tco2e = attribution * issuer_annual_tCO2e
            carbon_eur = financed_tco2e * ets_price_eur

            results.append(FinancedEmission(
                bank_name=exp.bank_name,
                issuer_ticker=ticker,
                issuer_name=issuer_name,
                site_name="all",  # aggregated across issuer sites
                exposure_eur=exp.exposure_eur,
                total_issuer_debt_eur=total_debt,
                attribution_factor=round(attribution, 6),
                issuer_annual_tCO2e=round(issuer_annual_tCO2e, 1),
                financed_tCO2e=round(financed_tco2e, 1),
                carbon_liability_eur=round(carbon_eur, 0),
                scenario=scenario,
            ))

        return results

    def bank_level_summary(
        self,
        issuer_emissions: dict[str, float],
        ets_price_eur: float = 65.0,
        scenario: Optional[str] = None,
    ) -> dict[str, dict]:
        """
        Aggregate financed emissions at the bank level across all issuers.

        Args:
            issuer_emissions: {ticker: annual_tCO2e} from stress test
            ets_price_eur:    carbon price for liability calculation

        Returns:
            {bank_name: {total_financed_tCO2e, total_carbon_eur, exposures: [...]}}
        """
        bank_agg: dict[str, dict] = {}

        for ticker, annual_tco2e in issuer_emissions.items():
            fe_list = self.compute_financed_emissions(
                ticker, annual_tco2e, ets_price_eur, scenario
            )
            for fe in fe_list:
                if fe.bank_name not in bank_agg:
                    bank_agg[fe.bank_name] = {
                        "total_financed_tCO2e": 0.0,
                        "total_carbon_liability_eur": 0.0,
                        "total_exposure_eur": 0.0,
                        "n_issuers": 0,
                        "exposures": [],
                    }
                agg = bank_agg[fe.bank_name]
                agg["total_financed_tCO2e"] += fe.financed_tCO2e
                agg["total_carbon_liability_eur"] += fe.carbon_liability_eur
                agg["total_exposure_eur"] += fe.exposure_eur
                agg["n_issuers"] += 1
                agg["exposures"].append(fe)

        # Round aggregates
        for bank, agg in bank_agg.items():
            agg["total_financed_tCO2e"] = round(agg["total_financed_tCO2e"], 1)
            agg["total_carbon_liability_eur"] = round(agg["total_carbon_liability_eur"], 0)
            agg["total_exposure_eur"] = round(agg["total_exposure_eur"], 0)

        return bank_agg

    def format_bank_summary(
        self,
        issuer_emissions: dict[str, float],
        ets_price_eur: float = 65.0,
        scenario: Optional[str] = None,
    ) -> str:
        """Pretty-print bank-level financed emission summary."""
        summary = self.bank_level_summary(issuer_emissions, ets_price_eur, scenario)

        lines = []
        sc_label = f" [{scenario.upper()}]" if scenario else ""
        lines.append(f"\n{'=' * 85}")
        lines.append(f"FINANCED EMISSIONS — BANK-LEVEL ATTRIBUTION{sc_label}")
        lines.append(f"(PCAF methodology · ETS price: €{ets_price_eur}/tCO2e)")
        lines.append(f"{'=' * 85}")
        lines.append(
            f"{'Bank':<22} {'Exposure €':>15} {'Issuers':>8} "
            f"{'Fin. tCO2e':>12} {'Carbon €/yr':>14}"
        )
        lines.append("-" * 85)

        for bank, agg in sorted(summary.items(), key=lambda x: -x[1]["total_carbon_liability_eur"]):
            lines.append(
                f"{bank:<22} {agg['total_exposure_eur']:>15,.0f} "
                f"{agg['n_issuers']:>8} "
                f"{agg['total_financed_tCO2e']:>12,.1f} "
                f"{agg['total_carbon_liability_eur']:>14,.0f}"
            )

        total_fe = sum(a["total_financed_tCO2e"] for a in summary.values())
        total_cl = sum(a["total_carbon_liability_eur"] for a in summary.values())
        lines.append("-" * 85)
        lines.append(
            f"{'TOTAL':<22} {'':>15} {'':>8} "
            f"{total_fe:>12,.1f} {total_cl:>14,.0f}"
        )
        lines.append(f"{'=' * 85}")

        return "\n".join(lines)
