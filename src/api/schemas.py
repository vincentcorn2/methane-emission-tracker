"""
API Schemas: The machine-readable contract for financial clients.

These Pydantic models define the EXACT structure of every API response.
Quantitative trading algorithms will parse these fields programmatically.
Any breaking change (missing key, type mutation, null where non-null expected)
will crash client pipelines. Treat these as immutable once published.

Version these schemas. When you must change them, bump the API version.
"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
from enum import Enum


class CloudQuality(str, Enum):
    OPTIMAL = "optimal"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNKNOWN = "unknown"


class EmissionEventResponse(BaseModel):
    """
    Primary API response: a fully resolved methane emission event.

    This is what a hedge fund's trading algorithm receives.
    Every field has a specific downstream use documented below.
    """
    # Event identification
    event_uuid: str = Field(
        ...,
        description="Unique event ID for deduplication and time-series integrity"
    )
    timestamp_utc: datetime = Field(
        ...,
        description="Sentinel-2 acquisition time — align with market tick data"
    )

    # Geospatial
    latitude: float = Field(
        ..., description="Plume centroid latitude"
    )
    longitude: float = Field(
        ..., description="Plume centroid longitude"
    )

    # Detection
    model_confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="CH4Net probability score — filter by risk tolerance"
    )
    plume_area_pixels: int = Field(
        ..., description="Plume spatial extent in Sentinel-2 pixels"
    )

    # Quantification (null if quantification module not yet run)
    flow_rate_kgh: Optional[float] = Field(
        None, description="Estimated CH4 emission rate (kg/hour) via IME"
    )
    flow_rate_lower_kgh: Optional[float] = Field(
        None, description="Lower bound estimate (50th percentile)"
    )
    flow_rate_upper_kgh: Optional[float] = Field(
        None, description="Upper bound estimate"
    )
    annual_tonnes_estimate: Optional[float] = Field(
        None, description="Annualized emission in metric tonnes (for IRA calc)"
    )
    ira_waste_charge_usd: Optional[float] = Field(
        None, description="Estimated IRA §136 Waste Emissions Charge liability"
    )

    # Entity resolution (null if asset not matched)
    asset_id: Optional[str] = Field(
        None, description="Facility ID from WRI/JRC database"
    )
    asset_name: Optional[str] = Field(
        None, description="Facility name"
    )
    asset_distance_km: Optional[float] = Field(
        None, description="Distance from plume centroid to matched asset"
    )
    asset_fuel_type: Optional[str] = Field(
        None, description="Primary fuel: gas, oil, coal, etc."
    )
    owner_name: Optional[str] = Field(
        None, description="Immediate facility owner (often a subsidiary)"
    )
    corporate_lei: Optional[str] = Field(
        None, description="20-char ISO 17442 Legal Entity Identifier of parent"
    )
    financial_ticker: Optional[str] = Field(
        None, description="Tradable equity ticker, e.g. 'XOM'"
    )
    exchange: Optional[str] = Field(
        None, description="Exchange code, e.g. 'NYSE'"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "event_uuid": "evt-2026-03-05-T7-001",
                "timestamp_utc": "2026-03-05T10:23:45Z",
                "latitude": 39.45965,
                "longitude": 53.77921,
                "model_confidence": 0.92,
                "plume_area_pixels": 340,
                "flow_rate_kgh": 1250.0,
                "flow_rate_lower_kgh": 625.0,
                "flow_rate_upper_kgh": 1875.0,
                "annual_tonnes_estimate": 10950.0,
                "ira_waste_charge_usd": 16425000.0,
                "asset_id": "WRI-TKM-00142",
                "asset_name": "Turkmenistan Gas Processing",
                "asset_distance_km": 1.2,
                "asset_fuel_type": "Gas",
                "owner_name": "Turkmengaz",
                "corporate_lei": None,
                "financial_ticker": None,
                "exchange": None,
            }
        }


class SearchRequest(BaseModel):
    """Input for searching/triggering detection over a region."""
    wkt_polygon: str = Field(
        ...,
        description="WKT POLYGON defining the region of interest"
    )
    start_date: str = Field(
        ...,
        description="ISO 8601 start date, e.g. '2026-01-01T00:00:00.000Z'"
    )
    end_date: str = Field(
        ...,
        description="ISO 8601 end date"
    )
    max_cloud_cover: float = Field(
        default=10.0,
        description="Maximum cloud cover percentage (server-side filter)"
    )
    min_confidence: float = Field(
        default=0.18,
        description="Minimum CH4Net confidence threshold"
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    asset_db_loaded: bool
    asset_count: int
    version: str


class PortfolioRiskRequest(BaseModel):
    """
    Input for portfolio-level risk calculations.

    A bank submits its portfolio tickers; the API returns
    aggregate emission exposure and estimated EU ETS carbon liabilities.
    """
    tickers: list[str] = Field(
        ..., description="List of equity tickers to assess, e.g. ['RWE.DE', 'PGE.WA']"
    )
    lookback_days: int = Field(
        default=90, description="Days of emission history to aggregate"
    )


class SiteRiskScore(BaseModel):
    """
    Full risk decomposition for a single industrial site.

    All monetary figures are in EUR and use EU ETS carbon pricing.
    Confidence intervals use Wilson 90% CI on empirical detection probability.
    """
    site:              str
    operator:          Optional[str]
    ticker:            Optional[str]
    exchange:          Optional[str]

    # Detection stats
    n_valid_dates:     int    = Field(description="S2 acquisitions passing bad-scene filter")
    n_detections:      int    = Field(description="Dates with S/C > 1.15 (CFAR threshold)")
    p_detect:          Optional[float] = Field(description="Empirical detection probability")
    p_detect_lo_90:    Optional[float] = Field(description="Wilson 90% CI lower bound on p_detect")
    p_detect_hi_90:    Optional[float] = Field(description="Wilson 90% CI upper bound on p_detect")
    mean_sc_detected:  Optional[float] = Field(description="Mean S/C ratio on detected dates")

    # Flow rate
    flow_rate_kgh:     Optional[float] = Field(description="CH4 emission rate (kg/hr)")
    flow_rate_source:  Optional[str]   = Field(description="'cemf_ime' or 'sc_proxy'")

    # Annual GHG liability
    annual_tCO2e:         Optional[float] = Field(description="Expected annual CH4 as CO2e (t/yr)")
    annual_tCO2e_lo_90:   Optional[float] = Field(description="Lower bound at 90% CI")
    annual_tCO2e_hi_90:   Optional[float] = Field(description="Upper bound at 90% CI")
    carbon_liability_eur: Optional[float] = Field(description="Expected annual carbon liability (EUR)")
    carbon_eur_lo_90:     Optional[float] = Field(description="Lower bound (EUR)")
    carbon_eur_hi_90:     Optional[float] = Field(description="Upper bound (EUR)")
    ets_price_eur_tonne:  float           = Field(description="EU ETS spot price used (EUR/tCO2e)")
    gwp100_ch4:           float           = Field(description="CH4 GWP-100 factor used (IPCC AR6)")

    # Cross-sensor validation
    tropomi_score:           Optional[float] = Field(
        description="Fraction of dates with dual-sensor confirm (TROPOMI ΔXCHₓ ≥ 5 ppb + S/C > 1.15)"
    )
    n_dual_sensor_confirms:  int = Field(description="Count of dates with dual-sensor confirmation")

    # Risk classification
    risk_tier: str = Field(
        description="HIGH_DUAL_SENSOR | HIGH | MEDIUM | LOW | UNDETECTED | NO_DATA"
    )


class TickerRiskSummary(BaseModel):
    """Aggregated emission risk for all sites linked to one equity ticker."""
    sites:                list[str]
    annual_tCO2e:         float = Field(description="Sum of expected annual CH4 as CO2e across sites")
    annual_tCO2e_lo_90:   float = Field(description="Sum of lower bounds at 90% CI")
    annual_tCO2e_hi_90:   float = Field(description="Sum of upper bounds at 90% CI")
    carbon_liability_eur: float = Field(description="Total annual EU ETS carbon liability (EUR)")
    carbon_eur_lo_90:     float
    carbon_eur_hi_90:     float
    risk_tier:            str   = Field(description="Highest risk tier across all linked sites")
    ets_price_eur_tonne:  float
    site_scores:          list[SiteRiskScore] = Field(description="Per-site breakdown")


class PortfolioRiskResponse(BaseModel):
    """
    Aggregate methane emission risk across a portfolio of equities.

    Carbon liabilities are computed using EU ETS pricing at the time of model
    loading.  All figures carry 90% confidence intervals propagated from the
    empirical detection probability (Wilson CI) and a ±50% flow-rate uncertainty
    heuristic (ERA5 wind + CEMF pixel-area caveat).
    """
    total_annual_tCO2e:   float = Field(description="Portfolio total CH4 as CO2e (t/yr)")
    total_carbon_eur:     float = Field(description="Portfolio total carbon liability (EUR/yr)")
    data_coverage_pct:    float = Field(description="Fraction of submitted tickers with satellite data (%)")
    per_ticker:           dict[str, TickerRiskSummary]
    unmatched_tickers:    list[str] = Field(description="Tickers not found in site operator map")
    ets_price_eur_tonne:  float     = Field(description="EU ETS price used (EUR/tCO2e)")


# ── Stress Testing Schemas ───────────────────────────────────────────────────

class StressTestRequest(BaseModel):
    """Input for climate transition stress testing."""
    tickers: list[str] = Field(
        ..., description="Equity tickers to stress (e.g. ['RWE.DE', 'PGE.WA'])"
    )
    scenarios: Optional[list[str]] = Field(
        None, description="Scenario names: 'orderly', 'disorderly', 'hot_house'. Default: all three."
    )
    horizon_years: int = Field(
        default=10, ge=1, le=30, description="Projection horizon in years"
    )
    n_paths: int = Field(
        default=50_000, ge=1000, le=500_000, description="Monte Carlo simulation paths"
    )


class SiteStressResponse(BaseModel):
    """Stress test result for one emitter site under one scenario."""
    site: str
    scenario: str
    horizon_years: int
    p_detect: float
    flow_rate_kgh: float
    flow_source: str
    mean_annual_cost_eur: list[float] = Field(description="Mean carbon cost per year")
    p95_annual_cost_eur: list[float] = Field(description="95th percentile cost per year")
    terminal_mean_eur: float = Field(description="Mean annual cost at horizon end")
    terminal_var95_eur: float = Field(description="95% VaR at horizon end")
    terminal_cvar95_eur: float = Field(description="95% CVaR (expected shortfall) at horizon end")
    npv_cumulative_mean_eur: float = Field(description="NPV of cumulative costs over horizon")
    npv_cumulative_p95_eur: float = Field(description="NPV 95th percentile")


class IssuerStressResponse(BaseModel):
    """Stress test result for one issuer under one scenario."""
    ticker: str
    operator: str
    scenario: str
    sites: list[str]
    terminal_mean_eur: float
    terminal_var95_eur: float
    terminal_cvar95_eur: float
    ebitda_eur_m: Optional[float]
    carbon_cost_to_ebitda_pct: Optional[float]
    implied_notch_downgrade: int
    implied_pd_bps: Optional[float]
    lgd: float
    npv_cumulative_mean_eur: float
    npv_cumulative_p95_eur: float
    site_results: list[SiteStressResponse]


class StressTestResponse(BaseModel):
    """
    Full portfolio stress test across NGFS climate transition scenarios.

    Returns Monte Carlo carbon liability distributions with credit
    transmission metrics (notch downgrade, PD shift, LGD adjustment).
    """
    scenarios_run: list[str]
    horizon_years: int
    n_paths: int
    discount_rate: float
    issuer_results: dict[str, list[IssuerStressResponse]] = Field(
        description="scenario_name → list of issuer results"
    )
    portfolio_terminal_mean_eur: dict[str, float] = Field(description="scenario → total mean")
    portfolio_terminal_var95_eur: dict[str, float] = Field(description="scenario → total VaR95")
    portfolio_terminal_cvar95_eur: dict[str, float] = Field(description="scenario → total CVaR95")
    portfolio_npv_mean_eur: dict[str, float] = Field(description="scenario → total NPV mean")
