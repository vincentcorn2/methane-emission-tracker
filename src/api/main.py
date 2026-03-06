"""
Methane Emission Tracker API

FastAPI application exposing satellite-derived methane intelligence
for financial institutions.

Endpoints:
  GET  /health          — Service health and model status
  POST /detect          — Trigger detection over a region/time window
  GET  /events          — Query historical emission events
  GET  /events/{id}     — Get a single event with full resolution
  POST /portfolio-risk  — Aggregate risk across a portfolio of tickers

Run:
  uvicorn src.api.main:app --reload --port 8000
  Then open http://localhost:8000/docs for interactive Swagger UI
"""
import uuid
import logging
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    EmissionEventResponse,
    SearchRequest,
    HealthResponse,
    PortfolioRiskRequest,
    PortfolioRiskResponse,
)
from src.entity_resolution.resolver import AssetDatabase, CorporateResolver

logger = logging.getLogger(__name__)

# --- Application state (initialized on startup) ---
asset_db = AssetDatabase()
corporate_resolver = CorporateResolver()
model_loaded = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load databases and model on startup."""
    global model_loaded

    # Load asset databases if files exist
    import os
    wri_path = os.environ.get("WRI_DB_PATH", "data/global_power_plant_database.csv")
    jrc_path = os.environ.get("JRC_DB_PATH", "data/jrc_power_plants.csv")

    if os.path.exists(wri_path):
        asset_db.load_wri_database(wri_path)
        logger.info("WRI database loaded")
    else:
        logger.warning("WRI database not found at %s — entity resolution disabled", wri_path)

    if os.path.exists(jrc_path):
        asset_db.load_jrc_database(jrc_path)
        logger.info("JRC database loaded")

    # TODO Phase 1: Load CH4Net model weights
    # detector = CH4NetDetector("weights/best_model.pth")
    # model_loaded = True

    yield  # App runs here

    # Cleanup on shutdown (if needed)
    logger.info("Shutting down")


app = FastAPI(
    title="Methane Emission Tracker API",
    description=(
        "Satellite-derived methane intelligence for financial institutions. "
        "Provides real-time emission detection, physical flow rate quantification, "
        "and entity resolution mapping emissions to tradable financial instruments."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Health Check
# ──────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Service health and readiness status."""
    return HealthResponse(
        status="ok",
        model_loaded=model_loaded,
        asset_db_loaded=asset_db._loaded,
        asset_count=len(asset_db._assets),
        version="0.1.0",
    )


# ──────────────────────────────────────────────
# Detection Endpoint
# ──────────────────────────────────────────────

@app.post("/detect", response_model=list[EmissionEventResponse])
async def detect_emissions(request: SearchRequest):
    """
    Trigger methane detection pipeline over a region and time window.

    Full pipeline:
      1. Query Copernicus for Sentinel-2 tiles (server-side cloud filter)
      2. Run CH4Net inference on each tile
      3. Quantify detected plumes (IME flow rate)
      4. Resolve to nearest industrial asset → corporate parent → ticker

    Returns a list of detected emission events with full financial context.
    """
    # TODO: Wire up the full pipeline. For now, return example structure.
    # This endpoint definition locks in the API contract so frontend/clients
    # can start integrating immediately.

    example_event = EmissionEventResponse(
        event_uuid=f"evt-{uuid.uuid4().hex[:12]}",
        timestamp_utc=datetime.utcnow(),
        latitude=39.46,
        longitude=53.78,
        model_confidence=0.87,
        plume_area_pixels=280,
        flow_rate_kgh=None,       # Phase 3
        asset_id=None,            # Phase 2: after WRI DB loaded
        asset_name=None,
        owner_name=None,
        corporate_lei=None,
        financial_ticker=None,
    )

    return [example_event]


# ──────────────────────────────────────────────
# Event Query Endpoints
# ──────────────────────────────────────────────

@app.get("/events", response_model=list[EmissionEventResponse])
async def list_events(
    ticker: Optional[str] = Query(None, description="Filter by equity ticker"),
    min_confidence: float = Query(0.18, description="Minimum detection confidence"),
    start_date: Optional[str] = Query(None, description="ISO 8601 start date"),
    end_date: Optional[str] = Query(None, description="ISO 8601 end date"),
    limit: int = Query(100, le=1000),
):
    """
    Query historical emission events.

    Financial clients use this for backtesting:
      - Get all XOM-linked events in last 90 days
      - Filter by confidence > 0.5 for high-conviction signals
      - Bulk export for econometric model training
    """
    # TODO: Implement database query (PostgreSQL + PostGIS, or DuckDB)
    return []


@app.get("/events/{event_uuid}", response_model=EmissionEventResponse)
async def get_event(event_uuid: str):
    """Get full details for a specific emission event."""
    # TODO: Database lookup
    raise HTTPException(status_code=404, detail=f"Event {event_uuid} not found")


# ──────────────────────────────────────────────
# Portfolio Risk Endpoint
# ──────────────────────────────────────────────

@app.post("/portfolio-risk", response_model=PortfolioRiskResponse)
async def portfolio_risk(request: PortfolioRiskRequest):
    """
    Aggregate methane emission risk across a portfolio of equities.

    Use case for banks (TCFD compliance):
      Submit your loan book tickers → get total financed emissions
      and estimated regulatory liabilities.

    Use case for hedge funds:
      Compare reported vs. satellite-observed emissions across holdings
      to identify greenwashing exposure.
    """
    # TODO: Aggregate from event database
    return PortfolioRiskResponse(
        total_annual_tonnes=0.0,
        total_ira_liability_usd=0.0,
        per_ticker={},
        data_coverage_pct=0.0,
    )
