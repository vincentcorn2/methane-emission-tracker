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
    aggregate emission exposure and estimated IRA liabilities.
    """
    tickers: list[str] = Field(
        ..., description="List of equity tickers to assess"
    )
    lookback_days: int = Field(
        default=90, description="Days of emission history to aggregate"
    )


class PortfolioRiskResponse(BaseModel):
    """Aggregate emission risk across a portfolio."""
    total_annual_tonnes: float
    total_ira_liability_usd: float
    per_ticker: dict  # ticker → {annual_tonnes, liability, event_count}
    data_coverage_pct: float  # what % of tickers had satellite coverage
