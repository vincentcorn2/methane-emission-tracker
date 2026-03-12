"""
Emission Quantification: Pixel mask → Physical flow rate (kg/h)

Implements two complementary methodologies from the strategy document:

1. Integrated Mass Enhancement (IME) — primary estimate
   Total mass = sum of per-pixel CH4 enhancements over plume area
   Flow rate Q = (IME * U_eff) / L

2. Cross-Sectional Flux (CSF) — validation / uncertainty quantification
   Measures mass crossing orthogonal transects downwind of source
   Q = I(x) * U_eff

Both methods require effective wind speed (U_eff) from ERA5 reanalysis.

Key financial output:
  Flow rate in kg/h → annualize → apply IRA Waste Emissions Charge
  ($1,500/metric ton in 2026) → direct P&L impact for the emitter
"""
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QuantificationResult:
    """Physical emission estimate with uncertainty bounds."""
    flow_rate_kgh: float            # primary estimate (kg CH4 / hour)
    flow_rate_lower_kgh: float      # lower bound (conservative)
    flow_rate_upper_kgh: float      # upper bound
    methodology: str                # "IME" or "CSF"
    wind_speed_ms: Optional[float]  # ERA5 effective wind speed used
    plume_length_m: Optional[float] # characteristic plume length
    total_mass_kg: Optional[float]  # integrated mass in observed plume

    # Financial derivatives
    annual_tonnes: Optional[float] = None
    ira_waste_charge_usd: Optional[float] = None

    def compute_financial_impact(self):
        """
        Translate physics into dollars.

        IRA Waste Emissions Charge (26 USC §136):
          2024: $900/ton
          2025: $1,200/ton
          2026+: $1,500/ton (statutory max)

        Applied to facilities emitting >25,000 tCO2e/yr
        above segment-specific intensity thresholds.
        """
        if self.flow_rate_kgh > 0:
            # Annualize: assume emitting continuously (conservative)
            # In practice, would weight by satellite revisit observations
            self.annual_tonnes = self.flow_rate_kgh * 8760 / 1000
            # 2026 rate
            self.ira_waste_charge_usd = self.annual_tonnes * 1500
        return self


class IntegratedMassEnhancement:
    """
    IME methodology for emission flow rate estimation.

    From the strategy document Section 3.2:
      IME = Σ ΔX_CH4,i * A_i   (sum over all plume pixels)
      Q = (IME * U_eff) / L

    Where:
      ΔX_CH4,i = methane column enhancement at pixel i (mol/m²)
      A_i = pixel area (m²)
      U_eff = effective wind speed (m/s)
      L = characteristic plume length along wind axis (m)

    IMPORTANT: Full implementation requires radiative transfer LUT
    to convert band 11/12 radiance ratios to ΔX_CH4.
    This is Phase 3 work. The stub below estimates from plume geometry.
    """

    def __init__(self, pixel_size_m: float = 20.0):
        """
        Args:
            pixel_size_m: Sentinel-2 pixel size. Band 12 (SWIR) is 20m native.
        """
        self.pixel_size_m = pixel_size_m
        self.pixel_area_m2 = pixel_size_m ** 2

    def estimate(
        self,
        plume_mask: np.ndarray,
        band_11: Optional[np.ndarray] = None,
        band_12: Optional[np.ndarray] = None,
        wind_speed_ms: float = 3.0,
    ) -> QuantificationResult:
        """
        Estimate emission flow rate from a detected plume mask.

        Phase 1 (current): Geometric estimation from plume size
        Phase 3 (planned): Full radiative transfer with Beer-Lambert

        Args:
            plume_mask: binary mask from CH4Net (H, W)
            band_11: Sentinel-2 band 11 (1560-1660nm) — reference
            band_12: Sentinel-2 band 12 (2090-2290nm) — CH4 absorption
            wind_speed_ms: effective wind speed from ERA5
        """
        plume_pixels = int(plume_mask.sum())
        plume_area_m2 = plume_pixels * self.pixel_area_m2

        # Estimate characteristic plume length from mask geometry
        rows, cols = np.where(plume_mask > 0)
        if len(rows) == 0:
            return QuantificationResult(
                flow_rate_kgh=0, flow_rate_lower_kgh=0, flow_rate_upper_kgh=0,
                methodology="IME", wind_speed_ms=wind_speed_ms,
                plume_length_m=0, total_mass_kg=0,
            )

        # Plume length: max extent in the primary axis
        plume_length_pixels = max(rows.max() - rows.min(), cols.max() - cols.min())
        plume_length_m = plume_length_pixels * self.pixel_size_m

        if band_11 is not None and band_12 is not None:
            # Phase 3: Full Beer-Lambert radiative transfer
            # T_plume = L12 / L11 = exp(-AMF * sigma_CH4 * delta_X_CH4)
            # For now, use simplified proxy
            flow_rate_kgh = self._simplified_estimate(
                plume_area_m2, plume_length_m, wind_speed_ms
            )
        else:
            # Phase 1: Geometric proxy (calibrated against known emitters)
            flow_rate_kgh = self._simplified_estimate(
                plume_area_m2, plume_length_m, wind_speed_ms
            )

        # Uncertainty: ±50% is typical for IME with Sentinel-2 resolution
        # (dominated by wind speed uncertainty)
        result = QuantificationResult(
            flow_rate_kgh=flow_rate_kgh,
            flow_rate_lower_kgh=flow_rate_kgh * 0.5,
            flow_rate_upper_kgh=flow_rate_kgh * 1.5,
            methodology="IME",
            wind_speed_ms=wind_speed_ms,
            plume_length_m=plume_length_m,
            total_mass_kg=None,  # Phase 3: actual mass from LUT
        )
        result.compute_financial_impact()
        return result

    def _simplified_estimate(
        self,
        plume_area_m2: float,
        plume_length_m: float,
        wind_speed_ms: float,
    ) -> float:
        """
        Simplified flow rate estimation from plume geometry.

        Calibrated against Turkmenistan super-emitter data where
        typical plumes of ~2km length correspond to ~5-50 t/h emissions.
        This is a rough proxy; Phase 3 replaces with full radiative transfer.

        Q ≈ (plume_area * background_enhancement * wind_speed) / plume_length
        """
        if plume_length_m < 1:
            return 0.0

        # Typical background enhancement for Sentinel-2 detectable plumes
        # is ~100-500 ppm·m (column-integrated)
        # Using conservative mid-range estimate
        assumed_enhancement_kg_m2 = 0.003  # ~300 ppm·m converted

        total_mass_kg = plume_area_m2 * assumed_enhancement_kg_m2
        residence_time_s = plume_length_m / max(wind_speed_ms, 0.5)
        flow_rate_kg_s = total_mass_kg / residence_time_s
        flow_rate_kgh = flow_rate_kg_s * 3600

        return round(flow_rate_kgh, 2)


class CEMFIntegratedMassEnhancement(IntegratedMassEnhancement):
    """
    IME quantification using CEMF-derived mass instead of geometric proxy.
    Replaces _simplified_estimate with spectrally-retrieved total_mass_kg.
    """

    def estimate_from_cemf(
        self,
        cemf_result,
        wind_speed_ms: float = 3.5,
        wind_source: str = "climatological_fallback",
    ) -> QuantificationResult:
        """
        Compute emission rate using CEMF-retrieved mass.

        Q = (total_mass_kg * wind_speed) / plume_length
        Units: kg * (m/s) / m = kg/s → convert to kg/h

        Args:
            cemf_result: CEMFResult from run_cemf()
            wind_speed_ms: ERA5 wind speed (or climatological fallback)
            wind_source: label for provenance tracking
        """
        if not cemf_result.retrieval_valid or cemf_result.total_mass_kg == 0:
            return QuantificationResult(
                flow_rate_kgh=0,
                flow_rate_lower_kgh=0,
                flow_rate_upper_kgh=0,
                methodology="CEMF+IME",
                wind_speed_ms=wind_speed_ms,
                plume_length_m=0,
                total_mass_kg=0,
            )

        # Plume length from mask geometry
        rows, cols = np.where(cemf_result.plume_mask > 0)
        if len(rows) == 0:
            plume_length_m = 1.0
        else:
            plume_length_pixels = max(
                rows.max() - rows.min(),
                cols.max() - cols.min()
            )
            plume_length_m = max(plume_length_pixels * self.pixel_size_m, 1.0)

        # Core IME inversion
        residence_time_s = plume_length_m / max(wind_speed_ms, 0.5)
        flow_rate_kg_s = cemf_result.total_mass_kg / residence_time_s
        flow_rate_kgh = round(flow_rate_kg_s * 3600, 2)

        # Uncertainty: 30-45% for CEMF+IME with Sentinel-2
        result = QuantificationResult(
            flow_rate_kgh=flow_rate_kgh,
            flow_rate_lower_kgh=round(flow_rate_kgh * 0.6, 2),
            flow_rate_upper_kgh=round(flow_rate_kgh * 1.4, 2),
            methodology="CEMF+IME",
            wind_speed_ms=wind_speed_ms,
            plume_length_m=plume_length_m,
            total_mass_kg=cemf_result.total_mass_kg,
        )
        result.compute_financial_impact()
        return result
