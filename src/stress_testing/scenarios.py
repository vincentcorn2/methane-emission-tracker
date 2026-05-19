"""
scenarios.py
============
ECB / NGFS climate transition scenarios with carbon-price and policy-shock
paths, calibrated to the 2024 NGFS Phase IV reference scenarios used by
the ECB in its Pillar 2 climate stress tests.

Each scenario defines:
  - A deterministic EU ETS carbon price path  EUR / tCO2e  (annual, 2024–2050)
  - A stochastic shock model (GBM with scenario-specific drift + vol)
  - A methane-specific regulatory multiplier (CBAM / IED recast trajectory)

Three canonical scenarios (NGFS Phase IV):
  1. Orderly Transition     — gradual carbon price rise to ~€150 by 2050
  2. Disorderly Transition  — delayed action, sharp 2030–35 carbon price spike
  3. Hot House World        — no new policy; price stays near current levels

Usage:
  from src.stress_testing.scenarios import SCENARIOS, simulate_ets_paths
  paths = simulate_ets_paths("disorderly", n_paths=10000, horizon_years=10)
  # paths.shape == (10000, 10)  — annual ETS prices per simulation path
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ClimateScenario:
    """One NGFS-style climate transition scenario."""
    name: str
    description: str

    # Deterministic carbon price path (EUR/tCO2e), indexed from base_year
    base_year: int
    price_path: dict[int, float]  # year → EUR/tCO2e

    # Stochastic GBM parameters (annualised, for Monte Carlo)
    drift_pct: float      # annual drift (e.g. 0.06 = +6%/yr)
    vol_pct: float        # annual volatility (e.g. 0.20 = 20%)

    # Methane-specific multiplier trajectory (IED recast / CBAM / ESRS)
    # 1.0 = no additional methane penalty; >1.0 = methane-specific surcharge
    ch4_multiplier_path: dict[int, float] = field(default_factory=dict)

    # Shock: one-off carbon price jump in a specific year (disorderly)
    shock_year: int | None = None
    shock_magnitude_pct: float = 0.0   # e.g. 0.40 = +40% jump


# ── NGFS Phase IV scenarios (calibrated to ECB 2024 CST) ────────────────────

def _interp_path(anchors: dict[int, float], start: int = 2024, end: int = 2050) -> dict[int, float]:
    """Linear interpolation between anchor years."""
    years = sorted(anchors.keys())
    path = {}
    for y in range(start, end + 1):
        if y in anchors:
            path[y] = anchors[y]
        elif y < years[0]:
            path[y] = anchors[years[0]]
        elif y > years[-1]:
            path[y] = anchors[years[-1]]
        else:
            # Find bracketing anchors
            lo = max(yr for yr in years if yr <= y)
            hi = min(yr for yr in years if yr >= y)
            if lo == hi:
                path[y] = anchors[lo]
            else:
                frac = (y - lo) / (hi - lo)
                path[y] = anchors[lo] + frac * (anchors[hi] - anchors[lo])
    return {y: round(v, 2) for y, v in path.items()}


ORDERLY = ClimateScenario(
    name="orderly",
    description=(
        "Net Zero 2050 — immediate, gradual policy tightening. Carbon price "
        "rises steadily from €65 to €150/tCO2e by 2050. Low volatility."
    ),
    base_year=2024,
    price_path=_interp_path({
        2024: 65, 2025: 70, 2027: 80, 2030: 100,
        2035: 120, 2040: 135, 2045: 145, 2050: 150,
    }),
    drift_pct=0.04,   # +4%/yr long-term
    vol_pct=0.15,     # moderate vol
    ch4_multiplier_path=_interp_path({
        2024: 1.0, 2026: 1.05, 2028: 1.10, 2030: 1.15,
        2035: 1.25, 2040: 1.30, 2050: 1.35,
    }),
)

DISORDERLY = ClimateScenario(
    name="disorderly",
    description=(
        "Delayed Transition — minimal policy until 2030, then abrupt "
        "tightening. Carbon price spikes 40% in 2030, rises to €200+ "
        "by 2040. High volatility, methane penalties front-loaded."
    ),
    base_year=2024,
    price_path=_interp_path({
        2024: 65, 2025: 68, 2027: 72, 2029: 75,
        2030: 130, 2032: 160, 2035: 195, 2040: 220,
        2045: 230, 2050: 240,
    }),
    drift_pct=0.03,
    vol_pct=0.30,     # high vol — policy uncertainty
    shock_year=2030,
    shock_magnitude_pct=0.40,
    ch4_multiplier_path=_interp_path({
        2024: 1.0, 2026: 1.0, 2029: 1.0,
        2030: 1.30, 2032: 1.40, 2035: 1.50, 2040: 1.50, 2050: 1.50,
    }),
)

HOT_HOUSE = ClimateScenario(
    name="hot_house",
    description=(
        "Current Policies — no new climate legislation beyond existing. "
        "Carbon price stagnates at €55–70, declining in real terms. "
        "No methane surcharge. Physical risk increases instead."
    ),
    base_year=2024,
    price_path=_interp_path({
        2024: 65, 2025: 63, 2027: 60, 2030: 58,
        2035: 55, 2040: 52, 2045: 50, 2050: 48,
    }),
    drift_pct=-0.01,  # slight decline in real terms
    vol_pct=0.12,     # low vol — policy certainty (no action)
    ch4_multiplier_path=_interp_path({
        2024: 1.0, 2030: 1.0, 2040: 1.0, 2050: 1.0,
    }),
)

SCENARIOS: dict[str, ClimateScenario] = {
    "orderly":    ORDERLY,
    "disorderly": DISORDERLY,
    "hot_house":  HOT_HOUSE,
}


# ── Monte Carlo path simulation ─────────────────────────────────────────────

def simulate_ets_paths(
    scenario_name: str,
    n_paths: int = 10_000,
    horizon_years: int = 10,
    base_year: int = 2024,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Simulate EU ETS carbon price paths under a given scenario using GBM.

    The deterministic path from the scenario sets the conditional mean;
    GBM diffusion (drift + vol) generates stochastic variation around it.

    Args:
        scenario_name: key in SCENARIOS dict
        n_paths:       number of Monte Carlo paths
        horizon_years: years to simulate forward
        base_year:     starting year
        rng:           numpy random Generator (for reproducibility)

    Returns:
        ndarray of shape (n_paths, horizon_years) — annual ETS prices (EUR/tCO2e)
    """
    sc = SCENARIOS[scenario_name]
    if rng is None:
        rng = np.random.default_rng(42)

    paths = np.zeros((n_paths, horizon_years))
    dt = 1.0  # annual steps

    # Starting price
    p0 = sc.price_path.get(base_year, 65.0)
    paths[:, 0] = p0

    for t in range(1, horizon_years):
        year = base_year + t
        # Deterministic drift targets the scenario's price path
        target = sc.price_path.get(year, sc.price_path.get(
            max(y for y in sc.price_path if y <= year), p0
        ))
        # Mean-reverting GBM: pull toward target with scenario vol
        prev = paths[:, t - 1]
        # Drift = scenario drift + mean-reversion pull toward deterministic target
        reversion_speed = 0.3  # how strongly paths revert to deterministic path
        drift = sc.drift_pct + reversion_speed * np.log(target / prev)

        # Shock year: one-off jump
        if sc.shock_year and year == sc.shock_year:
            drift += sc.shock_magnitude_pct

        # GBM step
        z = rng.standard_normal(n_paths)
        log_return = (drift - 0.5 * sc.vol_pct**2) * dt + sc.vol_pct * np.sqrt(dt) * z
        paths[:, t] = prev * np.exp(log_return)

        # Floor at €10 (carbon price can't go to zero in any real scenario)
        paths[:, t] = np.maximum(paths[:, t], 10.0)

    return paths


def get_ch4_multiplier(scenario_name: str, year: int) -> float:
    """Get the methane-specific surcharge multiplier for a given scenario and year."""
    sc = SCENARIOS[scenario_name]
    path = sc.ch4_multiplier_path
    if not path:
        return 1.0
    if year in path:
        return path[year]
    # Interpolate
    years = sorted(path.keys())
    if year <= years[0]:
        return path[years[0]]
    if year >= years[-1]:
        return path[years[-1]]
    lo = max(y for y in years if y <= year)
    hi = min(y for y in years if y >= year)
    frac = (year - lo) / (hi - lo)
    return path[lo] + frac * (path[hi] - path[lo])
