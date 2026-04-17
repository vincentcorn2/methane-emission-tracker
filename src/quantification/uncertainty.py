"""
uncertainty.py
==============
Single source of truth for the CH4 emission uncertainty budget.

All modules that produce or consume flow-rate uncertainty bounds must import
from here.  Prior to Phase 3, three conflicting values existed:
  - ime.py          estimate():          ±50% (lines 144-145, 0.5 / 1.5)
  - ime.py          estimate_from_cemf(): ±40% (lines 238-239, 0.6 / 1.4)
  - risk_model.py   FLOW_UNCERTAINTY_FACTOR: 0.50 (±50%)
  - emission_logger.py: hardcoded 40 (uncertainty_pct = 40)

Post-ERA5 quadrature budget (from WS2 Technical Report Section 4):
  σ_wind   ≈ 15%  (ERA5 10m vs. plume-layer; Varon 2021 AMT Table 2)
  σ_cemf   ≈ 15%  (CEMF sensitivity coeff uncertainty; Varon 2021)
  σ_length ≈ 15%  (plume-length projection along ERA5 wind vector)
  σ_pixel  ≈  8%  (Sentinel-2 20m pixel vs. true plume edge)
  ─────────────────────────────────────────────────────────────────
  σ_total  = √(15²+15²+15²+8²) ≈ 28%  → rounded to 30% for conservatism

Reference: Varon et al. 2021 AMT 14, 2771–2785, Table 2.
           WS2 Technical Report Section 4 (April 2026).

The fallback wind budget uses:
  σ_wind_fallback ≈ 50%  (climatological 3.5 m/s vs. actual; large spread)

UNCERTAINTY_PCT_ERA5    = 30   ← use when wind_source == "ERA5_reanalysis"
UNCERTAINTY_PCT_FALLBACK = 50  ← use when wind_source == "climatological_fallback_*"
"""

# ── Half-width uncertainty (% of point estimate) ──────────────────────────────

UNCERTAINTY_PCT_ERA5: int = 30
"""
Emission flow-rate uncertainty (half-width %) when ERA5 wind is used.
Derived from post-ERA5 quadrature budget: ±30% covers σ_wind+σ_cemf+σ_length+σ_pixel.
"""

UNCERTAINTY_PCT_FALLBACK: int = 50
"""
Emission flow-rate uncertainty (half-width %) when climatological fallback
wind (3.5 m/s) is used.  The broader bound reflects the larger wind uncertainty.
"""


def get_uncertainty_pct(wind_source: str) -> int:
    """
    Return the appropriate half-width uncertainty percentage for a given wind source.

    Args:
        wind_source: e.g. "ERA5_reanalysis" or "climatological_fallback_3.5ms"

    Returns:
        30 (ERA5) or 50 (fallback)
    """
    if "ERA5" in wind_source or "era5" in wind_source.lower():
        return UNCERTAINTY_PCT_ERA5
    return UNCERTAINTY_PCT_FALLBACK


def apply_uncertainty(
    flow_rate_kgh: float,
    wind_source: str,
    override_pct: int = None,
) -> tuple[float, float, int]:
    """
    Compute lower and upper bounds from a point flow estimate.

    Args:
        flow_rate_kgh: point estimate in kg/h
        wind_source:   label from ERA5Client (determines which ± to use)
        override_pct:  if provided, use this instead of the wind-source lookup

    Returns:
        (lower_kgh, upper_kgh, uncertainty_pct_used)
    """
    pct = override_pct if override_pct is not None else get_uncertainty_pct(wind_source)
    f = pct / 100.0
    return (
        round(flow_rate_kgh * (1 - f), 2),
        round(flow_rate_kgh * (1 + f), 2),
        pct,
    )
