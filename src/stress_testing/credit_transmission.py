"""
credit_transmission.py
======================
Merton-KMV structural credit model for mapping carbon-liability shock to
rating migration.

Replaces the hand-calibrated EBITDA-threshold notch table in stress_test.py
with a structural approach defensible as "Merton-KMV" in a model-validation
interview.

Theory
------
Merton (1974): equity = call option on firm assets with strike = face value of debt.
Distance-to-Default (DD):
    DD = (ln(V/D) + (μ - σ²/2) * T) / (σ * √T)

where:
    V = market value of assets  (≈ equity_MV + book_debt)
    D = default threshold       (≈ short-term debt + 0.5 * long-term debt)
    μ = risk-free rate          (≈ 3%)
    σ = asset volatility        (≈ equity_vol / leverage_ratio)
    T = time horizon            (1 year)

Carbon liability shock reduces V:
    V_stressed = V - PV(carbon_liability)

DD_stressed = (ln(V_stressed / D) + ...) / (σ * √T)

Rating mapping: KMV/Moody's EDF-to-rating calibration.
Source: Crosbie & Bohn (2003) "Modelling Default Risk", Moody's KMV.

DD to 1-year EDF (Expected Default Frequency):
    EDF(DD) ≈ N(-DD)  [simplified; full KMV uses empirical mapping]

EDF to Moody's rating (approximate):
    Aaa:  EDF < 0.02%
    Aa:   EDF 0.02–0.06%
    A:    EDF 0.06–0.19%
    Baa:  EDF 0.19–0.59%
    Ba:   EDF 0.59–1.90%
    B:    EDF 1.90–6.0%
    Caa+: EDF > 6%

Implementation
--------------
We use a simplified but structurally correct implementation:
  1. Infer V, D, σ_assets from equity market cap + financial ratios
  2. Compute baseline DD
  3. Subtract PV(carbon_liability) from V to get DD_stressed
  4. Map both DDs to ratings via N(-DD) → EDF → rating table
  5. Return notch shift = rating_stressed - rating_baseline

This makes the credit-transmission step auditable and structurally consistent
with Pillar 2 methodology guidance (ECB 2021, p.48: "Structural credit models
are preferred for transmission of climate risk to credit risk").
"""
import math
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── EDF–rating mapping (Moody's-equivalent) ──────────────────────────────────
# Each entry: (max_edf_pct, rating_label, rating_notch_idx)
# Notch index: 0=Aaa, 1=Aa, 2=A, 3=Baa, 4=Ba, 5=B, 6=Caa, 7=Ca/C (default)
_EDF_RATING_TABLE: list[tuple[float, str, int]] = [
    (0.02,  "Aaa", 0),
    (0.06,  "Aa",  1),
    (0.19,  "A",   2),
    (0.59,  "Baa", 3),
    (1.90,  "Ba",  4),
    (6.00,  "B",   5),
    (20.0,  "Caa", 6),
    (100.0, "Ca/C", 7),
]

# Market-cap proxies (EUR billions) for issuers in portfolio
# Source: Bloomberg, approximate as of Q4 2024
_ISSUER_MARKET_CAP_EUR_B: dict[str, float] = {
    "RWE.DE":  20.0,   # RWE AG
    "PGE.WA":   3.0,   # PGE S.A.
    "UN01.DE":  7.0,   # Uniper SE
    "SHEL.L":  180.0,  # Shell plc (LSE, adj. for JV stake exposure)
}

# Net debt (EUR billions) — approximate, sourced from annual reports
_ISSUER_NET_DEBT_EUR_B: dict[str, float] = {
    "RWE.DE":  10.0,
    "PGE.WA":   5.0,
    "UN01.DE":  6.0,
    "SHEL.L":   35.0,
}

# Annual equity volatility (σ_equity) — 252-day realised vol, approximate
_ISSUER_EQUITY_VOL: dict[str, float] = {
    "RWE.DE":  0.30,
    "PGE.WA":  0.35,
    "UN01.DE": 0.45,
    "SHEL.L":  0.20,
}

# Risk-free rate (ECB deposit facility, approx 2024 level)
_RISK_FREE_RATE: float = 0.03

# Default horizon (years)
_HORIZON_YEARS: float = 1.0


@dataclass
class MertonResult:
    ticker: str
    equity_eur: float            # market cap (EUR)
    asset_value_eur: float       # V = equity + debt proxy
    default_threshold_eur: float # D = 0.65 * total_liabilities
    asset_vol: float             # σ_assets
    dd_baseline: float           # Distance-to-Default before shock
    dd_stressed: float           # Distance-to-Default after carbon liability shock
    edf_baseline_pct: float      # EDF before shock (%)
    edf_stressed_pct: float      # EDF after shock (%)
    rating_baseline: str         # Moody's equivalent rating, baseline
    rating_stressed: str         # Moody's equivalent rating, stressed
    notch_idx_baseline: int      # Rating notch index (0=Aaa ... 7=Ca/C)
    notch_idx_stressed: int      # Rating notch index after shock
    implied_notch_downgrade: int # notch_idx_stressed - notch_idx_baseline (≥0)
    carbon_pv_eur: float         # PV of carbon liability that caused the shock


def _edf_to_rating(edf_pct: float) -> tuple[str, int]:
    """Map EDF (%) to Moody's rating label and notch index."""
    for max_edf, label, idx in _EDF_RATING_TABLE:
        if edf_pct <= max_edf:
            return label, idx
    return "Ca/C", 7


def _dd_to_edf_pct(dd: float) -> float:
    """Simplified Merton EDF: N(-DD), returned as percentage."""
    # Use erfc approximation (no scipy dependency)
    # N(-DD) = Φ(-DD) = 0.5 * erfc(DD / √2)
    return float(0.5 * math.erfc(dd / math.sqrt(2)) * 100)


def merton_dd_shift(
    ticker: str,
    carbon_pv_eur: float,
    discount_rate: float = 0.05,
    horizon_years: float = _HORIZON_YEARS,
) -> MertonResult:
    """
    Compute Merton Distance-to-Default before and after a carbon liability shock.

    The carbon_pv_eur is the present value of expected future carbon liabilities
    (NPV of Monte Carlo path mean), which is subtracted from the firm's asset value
    to compute the stressed DD.

    Args:
        ticker:          issuer equity ticker (must be in _ISSUER_MARKET_CAP_EUR_B)
        carbon_pv_eur:   PV of carbon liabilities from the stress test (EUR)
        discount_rate:   discount rate for PV (default 5%)
        horizon_years:   DD time horizon (default 1 year)

    Returns:
        MertonResult with baseline and stressed DD, ratings, notch shift.
    """
    equity_eur = _ISSUER_MARKET_CAP_EUR_B.get(ticker, 5.0) * 1e9
    net_debt_eur = _ISSUER_NET_DEBT_EUR_B.get(ticker, 5.0) * 1e9
    equity_vol = _ISSUER_EQUITY_VOL.get(ticker, 0.30)

    # Asset value: equity + net debt (simplified; full KMV iterates)
    V = equity_eur + net_debt_eur

    # Default threshold: 0.65 * total_liabilities (KMV calibration)
    # Here we use 0.65 * net_debt as a conservative proxy
    D = 0.65 * net_debt_eur if net_debt_eur > 0 else equity_eur * 0.5

    # Asset volatility: Modigliani-Miller equity vol de-levered
    leverage_ratio = V / equity_eur
    sigma_assets = equity_vol / leverage_ratio

    def _compute_dd(asset_value: float) -> float:
        if asset_value <= 0 or D <= 0:
            return -10.0  # effectively in default
        dd = (math.log(asset_value / D) + (_RISK_FREE_RATE - 0.5 * sigma_assets**2) * horizon_years) / \
             (sigma_assets * math.sqrt(horizon_years))
        return dd

    dd_baseline = _compute_dd(V)

    # Stressed asset value: subtract PV of carbon liabilities
    V_stressed = max(V - carbon_pv_eur, D * 0.01)  # floor above zero
    dd_stressed = _compute_dd(V_stressed)

    edf_baseline_pct = _dd_to_edf_pct(dd_baseline)
    edf_stressed_pct = _dd_to_edf_pct(dd_stressed)

    rating_base, notch_base = _edf_to_rating(edf_baseline_pct)
    rating_stress, notch_stress = _edf_to_rating(edf_stressed_pct)

    notch_down = max(0, notch_stress - notch_base)

    logger.info(
        "%s: V=%.1fB€ D=%.1fB€ σ_a=%.2f | DD %.2f→%.2f | EDF %.3f%%→%.3f%% | %s→%s (%+d notches)",
        ticker, V / 1e9, D / 1e9, sigma_assets,
        dd_baseline, dd_stressed,
        edf_baseline_pct, edf_stressed_pct,
        rating_base, rating_stress, notch_down,
    )

    return MertonResult(
        ticker=ticker,
        equity_eur=equity_eur,
        asset_value_eur=V,
        default_threshold_eur=D,
        asset_vol=sigma_assets,
        dd_baseline=round(dd_baseline, 4),
        dd_stressed=round(dd_stressed, 4),
        edf_baseline_pct=round(edf_baseline_pct, 4),
        edf_stressed_pct=round(edf_stressed_pct, 4),
        rating_baseline=rating_base,
        rating_stressed=rating_stress,
        notch_idx_baseline=notch_base,
        notch_idx_stressed=notch_stress,
        implied_notch_downgrade=notch_down,
        carbon_pv_eur=carbon_pv_eur,
    )


def dd_to_rating(dd: float) -> tuple[str, int]:
    """
    Public helper: Distance-to-Default → (Moody's rating label, notch index).
    """
    edf_pct = _dd_to_edf_pct(dd)
    return _edf_to_rating(edf_pct)
