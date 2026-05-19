"""
scripts/finance_transition_risk.py
==================================
Translate the Bełchatów methane detection record into transition-risk financial
scenarios for PGE Polska Grupa Energetyczna S.A. — the owner-operator of the
mine through its GiEK subsidiary.

This module is a *stylized scenario analysis*, not a predictive market forecast.
Equity and credit-spread shocks are illustrative stress assumptions chosen from
the academic transition-risk literature; they are not calibrated to PGE's
historical return distribution or implied volatility surface. The carbon-cost
calculation is grounded in the EU ETS pricing regime and the IPCC AR5 GWP100
factor for methane, both of which are public, dated, and verifiable.

Transmission channels modelled:
  (1) Implied annualised carbon-cost exposure under EU ETS-equivalent pricing.
  (2) Credit-spread stress on a hypothetical investment-grade PGE bond holding
      (Moody's Baa1, Fitch BBB at time of writing).
  (3) Equity-repricing scenario on a hypothetical PGE share holding.

Outputs:
  - results_analysis/finance_transition_risk.json (machine-readable)
  - stdout report and Mild/Moderate/Severe sensitivity table

Run:
  python3 scripts/finance_transition_risk.py
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("finance_transition_risk")


# ---------------------------------------------------------------------------
# Inputs — physical methane evidence
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MethaneEvidence:
    """Annualised CH4 estimate produced by the CH4Net + CEMF pipeline."""
    site_label: str
    annual_ch4_t_mean: float
    annual_ch4_t_lower: float    # 95% CI lower bound
    annual_ch4_t_upper: float    # 95% CI upper bound
    detection_overpasses: int
    source: str                  # path/identifier for traceability


BELCHATOW_EVIDENCE = MethaneEvidence(
    site_label="KWB Bełchatów (Climate TRACE asset 16168)",
    annual_ch4_t_mean=11_481.0,
    annual_ch4_t_lower=6_563.0,
    annual_ch4_t_upper=16_400.0,
    detection_overpasses=37,
    source="results_analysis/belchatow_annual_timeseries.json",
)


# ---------------------------------------------------------------------------
# Inputs — carbon pricing and GWP regime
# ---------------------------------------------------------------------------

GWP100_METHANE = 28      # IPCC AR5 (used by EU MRV and ETS reporting)
GWP20_METHANE  = 83      # IPCC AR6 short-horizon coefficient (sensitivity only)

@dataclass(frozen=True)
class CarbonPriceCase:
    name: str
    eur_per_tco2e: float
    rationale: str

CARBON_PRICE_CASES = [
    CarbonPriceCase("Low",     50.0, "Lower-band EUA realised price, 2023"),
    CarbonPriceCase("Central", 70.0, "EUA central reference, 2024-2025"),
    CarbonPriceCase("Upper",   95.0, "Forward-curve estimate, post-2027 MSR tightening"),
]


# ---------------------------------------------------------------------------
# Inputs — PGE issuer profile (public information; sourced from rating circulars)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IssuerProfile:
    name: str
    ticker: str
    market_cap_usd_b_2026: float
    fx_usd_per_eur: float
    moodys_rating: str
    fitch_rating: str
    rating_outlook: str
    bond_duration_yrs: float  # used as default for the credit-spread scenario


PGE_PROFILE = IssuerProfile(
    name="PGE Polska Grupa Energetyczna S.A.",
    ticker="WSE:PGE",
    market_cap_usd_b_2026=6.34,
    fx_usd_per_eur=1.08,
    moodys_rating="Baa1",
    fitch_rating="BBB",
    rating_outlook="Stable (both agencies)",
    bond_duration_yrs=5.0,
)


# ---------------------------------------------------------------------------
# Inputs — hypothetical position sizes and illustrative stress tiers
# ---------------------------------------------------------------------------

HYPOTHETICAL_BOND_NOTIONAL_EUR   = 10_000_000.0
HYPOTHETICAL_EQUITY_NOTIONAL_EUR = 10_000_000.0

@dataclass(frozen=True)
class StressTier:
    name: str
    equity_shock_pct: float        # negative = drawdown
    spread_widening_bp: float      # positive = widening
    narrative: str

STRESS_TIERS = [
    StressTier(
        name="Mild",
        equity_shock_pct=-3.0,
        spread_widening_bp=15.0,
        narrative="ESG-rating downgrade; transition risk priced gradually",
    ),
    StressTier(
        name="Moderate",
        equity_shock_pct=-7.0,
        spread_widening_bp=35.0,
        narrative="EU Methane Regulation enforcement action; provisional fines",
    ),
    StressTier(
        name="Severe",
        equity_shock_pct=-10.0,
        spread_widening_bp=50.0,
        narrative="Sustained inventory-vs-satellite gap publicised; credit-watch negative",
    ),
]


# ---------------------------------------------------------------------------
# Channel 1 — implied carbon-cost exposure
# ---------------------------------------------------------------------------

def methane_to_co2e_tonnes(t_ch4: float, gwp: float = GWP100_METHANE) -> float:
    """Convert physical methane tonnes to CO2-equivalent tonnes."""
    return t_ch4 * gwp


def implied_carbon_cost_eur(
    t_ch4: float, eur_per_tco2e: float, gwp: float = GWP100_METHANE
) -> float:
    """
    Implied annual carbon-cost exposure under an EU ETS-equivalent pricing regime.

    This is *not* an actual booked ETS liability — coal-mine methane is not
    currently a covered emission under the EU ETS. It represents the
    order-of-magnitude potential exposure if a comparable price signal applied,
    consistent with the EU Methane Regulation (2024/1787) trajectory.
    """
    return methane_to_co2e_tonnes(t_ch4, gwp) * eur_per_tco2e


# ---------------------------------------------------------------------------
# Channel 2 — credit-spread stress (repurposed CR01 math, single-name)
# ---------------------------------------------------------------------------

def bond_cr01_eur(position_notional_eur: float, duration_yrs: float) -> float:
    """
    CR01 — change in market value per 1 basis point of credit-spread widening.
    Linear duration approximation: dP/ds ≈ -duration × notional, scaled by 1bp.
    """
    return position_notional_eur * duration_yrs * 1e-4


def credit_spread_pnl_eur(
    position_notional_eur: float,
    duration_yrs: float,
    spread_widening_bp: float,
) -> float:
    """Bondholder mark-to-market P&L for a parallel spread move (loss = negative)."""
    cr01 = bond_cr01_eur(position_notional_eur, duration_yrs)
    return -cr01 * spread_widening_bp


# ---------------------------------------------------------------------------
# Channel 3 — equity-repricing scenario (single-name PGE)
# ---------------------------------------------------------------------------

def equity_repricing_pnl_eur(
    position_notional_eur: float, equity_shock_pct: float
) -> float:
    """Long-equity P&L under a stylized shock (loss = negative)."""
    return position_notional_eur * (equity_shock_pct / 100.0)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def carbon_cost_table(
    evidence: MethaneEvidence, prices: Iterable[CarbonPriceCase]
) -> list[dict]:
    """Build the carbon-cost exposure table across price cases and CI bounds."""
    rows = []
    for p in prices:
        rows.append({
            "price_case": p.name,
            "eur_per_tco2e": p.eur_per_tco2e,
            "exposure_eur_mean": round(
                implied_carbon_cost_eur(evidence.annual_ch4_t_mean, p.eur_per_tco2e), 0
            ),
            "exposure_eur_lower": round(
                implied_carbon_cost_eur(evidence.annual_ch4_t_lower, p.eur_per_tco2e), 0
            ),
            "exposure_eur_upper": round(
                implied_carbon_cost_eur(evidence.annual_ch4_t_upper, p.eur_per_tco2e), 0
            ),
            "rationale": p.rationale,
        })
    return rows


def stress_table(
    issuer: IssuerProfile,
    bond_notional_eur: float,
    equity_notional_eur: float,
    tiers: Iterable[StressTier],
) -> list[dict]:
    """Build the Mild/Moderate/Severe sensitivity table."""
    rows = []
    cr01 = bond_cr01_eur(bond_notional_eur, issuer.bond_duration_yrs)
    for tier in tiers:
        equity_pnl = equity_repricing_pnl_eur(equity_notional_eur, tier.equity_shock_pct)
        spread_pnl = credit_spread_pnl_eur(
            bond_notional_eur, issuer.bond_duration_yrs, tier.spread_widening_bp
        )
        rows.append({
            "tier": tier.name,
            "equity_shock_pct": tier.equity_shock_pct,
            "spread_widening_bp": tier.spread_widening_bp,
            "equity_pnl_eur": round(equity_pnl, 0),
            "spread_pnl_eur": round(spread_pnl, 0),
            "combined_pnl_eur": round(equity_pnl + spread_pnl, 0),
            "narrative": tier.narrative,
        })
    return {"cr01_per_bp_eur": round(cr01, 2), "rows": rows}


def _fmt_eur(x: float) -> str:
    if abs(x) >= 1e6:
        return f"€{x/1e6:,.2f}M"
    if abs(x) >= 1e3:
        return f"€{x/1e3:,.1f}K"
    return f"€{x:,.0f}"


def print_report(payload: dict) -> None:
    e = payload["methane_evidence"]
    print("=" * 78)
    print(f"PGE / {e['site_label']} — transition-risk scenarios")
    print("=" * 78)
    print(f"Annualised CH4 (mean):  {e['annual_ch4_t_mean']:>10,.0f} t  "
          f"(95% CI {e['annual_ch4_t_lower']:,.0f}–{e['annual_ch4_t_upper']:,.0f})")
    print(f"CO2e (GWP100 = {GWP100_METHANE}):   {payload['co2e_t_mean']:>10,.0f} t CO2e  "
          f"(GWP20 = {GWP20_METHANE} → {payload['co2e_t_mean_gwp20']:,.0f} t CO2e)")

    print()
    print("Channel 1 — Implied carbon-cost exposure (EU ETS-equivalent pricing)")
    print("-" * 78)
    print(f"{'Price case':<10} {'€/tCO2e':>8}  {'Mean':>14}  "
          f"{'Lower (CI)':>14}  {'Upper (CI)':>14}")
    for r in payload["carbon_cost_table"]:
        print(f"{r['price_case']:<10} {r['eur_per_tco2e']:>8.0f}  "
              f"{_fmt_eur(r['exposure_eur_mean']):>14}  "
              f"{_fmt_eur(r['exposure_eur_lower']):>14}  "
              f"{_fmt_eur(r['exposure_eur_upper']):>14}")

    st = payload["stress_table"]
    print()
    print("Channels 2 & 3 — Hypothetical position stress test")
    print("-" * 78)
    print(f"Issuer:          {payload['issuer_profile']['name']}")
    print(f"Ratings:         Moody's {payload['issuer_profile']['moodys_rating']}, "
          f"Fitch {payload['issuer_profile']['fitch_rating']} "
          f"({payload['issuer_profile']['rating_outlook']})")
    print(f"Bond position:   {_fmt_eur(HYPOTHETICAL_BOND_NOTIONAL_EUR)} notional, "
          f"{payload['issuer_profile']['bond_duration_yrs']:.1f}y modified duration")
    print(f"  CR01:          {_fmt_eur(st['cr01_per_bp_eur'])} per +1 bp spread move")
    print(f"Equity position: {_fmt_eur(HYPOTHETICAL_EQUITY_NOTIONAL_EUR)} notional")
    print()
    print(f"{'Tier':<10} {'ΔEq (%)':>8} {'ΔSpr (bp)':>10}  "
          f"{'Equity P&L':>14}  {'Bond P&L':>14}  {'Combined':>14}")
    for r in st["rows"]:
        print(f"{r['tier']:<10} {r['equity_shock_pct']:>+8.1f} "
              f"{r['spread_widening_bp']:>+10.1f}  "
              f"{_fmt_eur(r['equity_pnl_eur']):>14}  "
              f"{_fmt_eur(r['spread_pnl_eur']):>14}  "
              f"{_fmt_eur(r['combined_pnl_eur']):>14}")
    print()
    print("Notes:")
    print("  • Position sizes are hypothetical, not held positions.")
    print("  • Equity and spread shocks are illustrative stress tiers, not")
    print("    calibrated point estimates of market response.")
    print("  • Coal-mine methane is not currently a covered EU ETS emission;")
    print("    the carbon-cost figures are order-of-magnitude exposure proxies,")
    print("    not booked liabilities.")
    print("=" * 78)


def main() -> dict:
    log.info("Building transition-risk payload for %s", BELCHATOW_EVIDENCE.site_label)

    co2e_mean = methane_to_co2e_tonnes(BELCHATOW_EVIDENCE.annual_ch4_t_mean, GWP100_METHANE)
    co2e_mean_gwp20 = methane_to_co2e_tonnes(BELCHATOW_EVIDENCE.annual_ch4_t_mean, GWP20_METHANE)

    payload = {
        "schema_version": "1.0.0",
        "framing": ("Stylized scenario analysis. Equity and spread shocks are "
                    "illustrative stress tiers, not predictive forecasts."),
        "methane_evidence": asdict(BELCHATOW_EVIDENCE),
        "gwp_factors": {"GWP100": GWP100_METHANE, "GWP20": GWP20_METHANE},
        "co2e_t_mean": round(co2e_mean, 0),
        "co2e_t_lower": round(methane_to_co2e_tonnes(BELCHATOW_EVIDENCE.annual_ch4_t_lower), 0),
        "co2e_t_upper": round(methane_to_co2e_tonnes(BELCHATOW_EVIDENCE.annual_ch4_t_upper), 0),
        "co2e_t_mean_gwp20": round(co2e_mean_gwp20, 0),
        "carbon_cost_table": carbon_cost_table(BELCHATOW_EVIDENCE, CARBON_PRICE_CASES),
        "issuer_profile": asdict(PGE_PROFILE),
        "hypothetical_bond_notional_eur": HYPOTHETICAL_BOND_NOTIONAL_EUR,
        "hypothetical_equity_notional_eur": HYPOTHETICAL_EQUITY_NOTIONAL_EUR,
        "stress_table": stress_table(
            PGE_PROFILE,
            HYPOTHETICAL_BOND_NOTIONAL_EUR,
            HYPOTHETICAL_EQUITY_NOTIONAL_EUR,
            STRESS_TIERS,
        ),
    }

    print_report(payload)

    out_path = Path("results_analysis/finance_transition_risk.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    log.info("Wrote %s", out_path)
    return payload


if __name__ == "__main__":
    main()
