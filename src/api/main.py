"""
Methane Emission Tracker API

FastAPI application exposing satellite-derived methane intelligence
for financial institutions.

Endpoints:
  GET  /health                — Service health and model status
  POST /detect                — Trigger detection over a region/time window
  GET  /events                — Query historical emission events
  GET  /events/{id}           — Get a single event with full resolution
  GET  /site-risk/{site_name} — Full risk score for a single industrial site
  POST /portfolio-risk        — Aggregate risk across a portfolio of tickers

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
    SiteRiskScore,
    TickerRiskSummary,
    StressTestRequest,
    StressTestResponse,
    IssuerStressResponse,
    SiteStressResponse,
)
from src.api.risk_model import RiskModel
from src.entity_resolution.resolver import AssetDatabase, CorporateResolver
from src.stress_testing.stress_test import StressTestEngine

logger = logging.getLogger(__name__)

# --- Application state (initialized on startup) ---
asset_db = AssetDatabase()
corporate_resolver = CorporateResolver()
risk_model: Optional[RiskModel] = None
stress_engine: Optional[StressTestEngine] = None
model_loaded = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load databases and model on startup."""
    global model_loaded, risk_model, stress_engine

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

    # Load risk model (reads pre-computed validation JSON files)
    try:
        risk_model = RiskModel()
        logger.info("RiskModel loaded")
    except Exception as e:
        logger.warning("RiskModel failed to load: %s — /portfolio-risk will return empty results", e)
        risk_model = None

    # Load stress test engine
    try:
        stress_engine = StressTestEngine()
        logger.info("StressTestEngine loaded")
    except Exception as e:
        logger.warning("StressTestEngine failed to load: %s", e)
        stress_engine = None

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

    **Use case — banks (TCFD / SFDR compliance):**
    Submit your loan-book tickers → receive total financed emissions and
    estimated EU ETS carbon liabilities with 90% confidence intervals.

    **Use case — hedge funds:**
    Compare reported vs. satellite-observed emissions across holdings to
    identify potential greenwashing exposure or unpriced regulatory risk.

    **Methodology:**
    - Detection probability estimated from multi-date Sentinel-2 CH4Net inference
    - Flow rates from CEMF/IME quantification (or S/C proxy where unavailable)
    - Annual GHG liability = p_detect × flow_kgh × 8760 × 0.85 × GWP-100
    - Carbon liability priced at EU ETS spot (updated in risk_model.py constants)
    - TROPOMI S5P L2 XCH4 cross-validation provides dual-sensor confirmation tier

    Returns 404 if the risk model is not loaded (validation data files absent).
    """
    if risk_model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Risk model not available. Ensure results_analysis/multidate_validation.json "
                "exists (run validate_multidate.py first)."
            ),
        )

    raw = risk_model.portfolio_risk(request.tickers, request.lookback_days)

    # Coerce per-ticker dicts → typed Pydantic models
    per_ticker_typed: dict[str, TickerRiskSummary] = {}
    for ticker, td in raw["per_ticker"].items():
        site_scores_typed = [SiteRiskScore(**s) for s in td["site_scores"]]
        per_ticker_typed[ticker] = TickerRiskSummary(
            sites=td["sites"],
            annual_tCO2e=td["annual_tCO2e"],
            annual_tCO2e_lo_90=td["annual_tCO2e_lo_90"],
            annual_tCO2e_hi_90=td["annual_tCO2e_hi_90"],
            carbon_liability_eur=td["carbon_liability_eur"],
            carbon_eur_lo_90=td["carbon_eur_lo_90"],
            carbon_eur_hi_90=td["carbon_eur_hi_90"],
            risk_tier=td["risk_tier"],
            ets_price_eur_tonne=td["ets_price_eur_tonne"],
            site_scores=site_scores_typed,
        )

    return PortfolioRiskResponse(
        total_annual_tCO2e=raw["total_annual_tCO2e"],
        total_carbon_eur=raw["total_carbon_eur"],
        data_coverage_pct=raw["data_coverage_pct"],
        per_ticker=per_ticker_typed,
        unmatched_tickers=raw["unmatched_tickers"],
        ets_price_eur_tonne=raw["ets_price_eur_tonne"],
    )


@app.get("/site-risk/{site_name}", response_model=SiteRiskScore)
async def site_risk(site_name: str):
    """
    Full risk decomposition for a single industrial site.

    **site_name** must match one of the sites tracked by the CH4Net validation
    pipeline, e.g.: `weisweiler`, `belchatow`, `neurath`, `boxberg`, `groningen`.

    Returns detection statistics, flow rate estimates, annualised GHG liability
    with 90% CI, TROPOMI dual-sensor confirmation score, and risk tier.
    """
    if risk_model is None:
        raise HTTPException(
            status_code=503,
            detail="Risk model not available. Run validate_multidate.py first.",
        )

    from src.api.risk_model import SITE_OPERATOR_MAP
    known_sites = set(SITE_OPERATOR_MAP.keys())

    if site_name not in known_sites:
        raise HTTPException(
            status_code=404,
            detail=f"Site '{site_name}' not found. Known sites: {sorted(known_sites)}",
        )

    score = risk_model.site_risk(site_name)
    return SiteRiskScore(**score)


# ──────────────────────────────────────────────
# Climate Stress Test Endpoint
# ──────────────────────────────────────────────

@app.post("/stress-test", response_model=StressTestResponse)
async def run_stress_test(request: StressTestRequest):
    """
    Run NGFS Phase IV climate transition stress test on a portfolio.

    **ECB Pillar 2 climate stress testing methodology:**

    Takes satellite-detected methane emissions and projects forward-looking
    carbon liability distributions under three NGFS scenarios:

    - **Orderly** (Net Zero 2050): gradual carbon price rise to €150/tCO2e
    - **Disorderly** (Delayed Transition): policy shock in 2030, spike to €240
    - **Hot House World**: carbon price stagnates at €50–65

    Monte Carlo propagates uncertainty from both detection probability
    (Wilson CI → Beta distribution) and carbon price volatility (GBM).

    Returns per-issuer VaR/CVaR, credit transmission metrics (notch
    downgrade, implied PD, LGD adjustment), and portfolio-level aggregates.

    **Use case — ECB/EIB supervisors:**
    Identify which loan-book exposures face material carbon cost increases
    under stress, and estimate the implied credit quality migration.
    """
    if stress_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Stress test engine not available. Ensure validation data files exist.",
        )

    result = stress_engine.run_portfolio_stress(
        tickers=request.tickers,
        scenarios=request.scenarios,
        horizon_years=request.horizon_years,
        n_paths=request.n_paths,
    )

    # Convert dataclass results to Pydantic response
    issuer_resp: dict[str, list[IssuerStressResponse]] = {}
    for scenario, issuers in result.issuer_results.items():
        issuer_resp[scenario] = []
        for ir in issuers:
            site_resps = [
                SiteStressResponse(
                    site=sr.site,
                    scenario=sr.scenario,
                    horizon_years=sr.horizon_years,
                    p_detect=sr.p_detect,
                    flow_rate_kgh=sr.flow_rate_kgh,
                    flow_source=sr.flow_source,
                    mean_annual_cost_eur=[round(v, 2) for v in sr.mean_annual_cost_eur],
                    p95_annual_cost_eur=[round(v, 2) for v in sr.p95_annual_cost_eur],
                    terminal_mean_eur=round(sr.terminal_mean_eur, 2),
                    terminal_var95_eur=round(sr.terminal_var95_eur, 2),
                    terminal_cvar95_eur=round(sr.terminal_cvar95_eur, 2),
                    npv_cumulative_mean_eur=round(sr.npv_cumulative_mean_eur, 2),
                    npv_cumulative_p95_eur=round(sr.npv_cumulative_p95_eur, 2),
                )
                for sr in ir.site_results
            ]
            issuer_resp[scenario].append(IssuerStressResponse(
                ticker=ir.ticker,
                operator=ir.operator,
                scenario=ir.scenario,
                sites=ir.sites,
                terminal_mean_eur=round(ir.terminal_mean_eur, 2),
                terminal_var95_eur=round(ir.terminal_var95_eur, 2),
                terminal_cvar95_eur=round(ir.terminal_cvar95_eur, 2),
                ebitda_eur_m=ir.ebitda_eur_m,
                carbon_cost_to_ebitda_pct=round(ir.carbon_cost_to_ebitda_pct, 4) if ir.carbon_cost_to_ebitda_pct else None,
                implied_notch_downgrade=ir.implied_notch_downgrade,
                implied_pd_bps=ir.implied_pd_bps,
                lgd=ir.lgd,
                npv_cumulative_mean_eur=round(ir.npv_cumulative_mean_eur, 2),
                npv_cumulative_p95_eur=round(ir.npv_cumulative_p95_eur, 2),
                site_results=site_resps,
            ))

    return StressTestResponse(
        scenarios_run=result.scenarios_run,
        horizon_years=result.horizon_years,
        n_paths=result.n_paths,
        discount_rate=result.discount_rate,
        issuer_results=issuer_resp,
        portfolio_terminal_mean_eur={k: round(v, 2) for k, v in result.portfolio_terminal_mean_eur.items()},
        portfolio_terminal_var95_eur={k: round(v, 2) for k, v in result.portfolio_terminal_var95_eur.items()},
        portfolio_terminal_cvar95_eur={k: round(v, 2) for k, v in result.portfolio_terminal_cvar95_eur.items()},
        portfolio_npv_mean_eur={k: round(v, 2) for k, v in result.portfolio_npv_mean_eur.items()},
    )
