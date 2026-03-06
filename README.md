# Methane Emission Tracker API
## Satellite-Derived Methane Intelligence for Financial Institutions

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Gateway                          │
│   /detect  /quantify  /resolve  /events  /portfolio-risk        │
└──────┬──────────┬───────────┬──────────┬───────────┬────────────┘
       │          │           │          │           │
  ┌────▼───┐ ┌───▼────┐ ┌───▼───┐ ┌───▼────┐ ┌───▼──────────┐
  │Ingestion│ │CH4Net  │ │Physics│ │Entity  │ │Risk          │
  │Module   │ │Detect  │ │Quant  │ │Resolve │ │Calculations  │
  │         │ │        │ │       │ │        │ │              │
  │Copernicus│ │UNet    │ │IME    │ │WRI DB  │ │IRA Waste Fee │
  │OData API│ │Infer   │ │CSF    │ │JRC DB  │ │CVaR Inputs   │
  │Token    │ │Thresh  │ │ERA5   │ │OpenCorp│ │TCFD Metrics  │
  │Cache    │ │Calibr  │ │Wind   │ │LEI→ISIN│ │              │
  └─────────┘ └────────┘ └───────┘ └────────┘ └──────────────┘
```

### Execution Order (Priority-Ranked)

**Phase 1: Foundation (Week 1-2)**
- [x] Modularize Copernicus ingestion out of notebook
- [x] Fix hardcoded credentials → environment variables
- [x] Modularize CH4Net model into importable module
- [x] Define JSON schema contract for emission events
- [x] Stand up basic FastAPI skeleton

**Phase 2: Entity Resolution — THE DIFFERENTIATOR (Week 2-4)**
- [ ] Ingest WRI Global Power Plant Database
- [ ] Ingest JRC European Power Plants Database
- [ ] Build geospatial nearest-asset matcher (GeoPandas)
- [ ] Prototype corporate hierarchy resolver (OpenCorporates → LEI)
- [ ] LEI → ISIN → Ticker mapping pipeline

**Phase 3: Quantification (Week 3-5)**
- [ ] Implement IME (Integrated Mass Enhancement) flow rate
- [ ] ERA5 wind speed integration for residence time
- [ ] Cross-Sectional Flux validation methodology
- [ ] IRA Waste Emissions Charge calculator ($1500/ton in 2026)

**Phase 4: API Polish & Risk Applications (Week 5-6)**
- [ ] GraphQL endpoint for low-latency queries
- [ ] Climate VaR input calculator
- [ ] TCFD financed emissions aggregator
- [ ] Backtest-ready historical data export (REST bulk endpoint)

### Quick Start

```bash
# 1. Set environment variables
cp config/.env.example config/.env
# Edit .env with your Copernicus credentials

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the API
uvicorn src.api.main:app --reload --port 8000

# 4. Open docs
# http://localhost:8000/docs
```

### Why This Architecture Matters for a Climate Risk Quant Role

This project demonstrates the exact skillset described in Section 9.2
of our strategy document — bridging atmospheric physics with financial
mathematics. The entity resolution pipeline (Section 6) is the component
that separates a generic ML project from an institutional-grade data product.

A portfolio manager doesn't care about pixel masks. They care about:
"How does this leak change XOM's EPS by Q3?"

That translation — from physics to P&L — is what this API delivers.
