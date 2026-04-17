# CH₄Net v2 — Satellite Methane Detection for Climate Financial Risk

**ECB / EIB Sponsored Research · Columbia University MSOR · 2024–2025**

> *First dual-sensor (Sentinel-2 U-Net + TROPOMI S5P) confirmation of methane emissions from European power plants, translated into EU ETS carbon liability estimates for climate stress testing.*

---

## What This Is

A production-grade pipeline that turns raw Sentinel-2 SWIR satellite imagery into **probability-weighted EU ETS carbon liability scores** for power-sector portfolios — no corporate disclosure required.

Built for the European Central Bank (ECB) and European Investment Bank (EIB) as a Columbia University MSOR capstone, in response to the need for independent, satellite-derived emission signals in Pillar 2 climate stress testing.

---

## Key Results (Summer 2024, JRC Top-10 EU Power Plants, >25 GW)

### JRC Top-10 Coverage Status

| # | Site / Operator / MW | Detection | Wind | Mask | Quantification | Expected Q (kg/h) |
|---|---------------------|-----------|------|------|---------------|------------------|
| 1 | **Bełchatów** / PGE / 4,830 MW | 2024-08-24 T34UCB, S/C=27.3 | ERA5 2.19 m/s | ORIGINAL (BT kills signal) | CEMF pending Phase 2 | ~240–350 |
| 2 | **Neurath** / RWE / 4,400 MW | 2024-06-25 T31UGS, S/C=45.1, TROPOMI +12.2 ppb | ERA5 needed | ORIGINAL | **HIGHEST PRIORITY** — no record yet | ~600–1,000 |
| 3 | Niederaussem / RWE / 3,827 MW | 0/3 dates, TROPOMI background | N/A | N/A | Excluded from quant (non-detection) | 0 |
| 4 | Boxberg / LEAG / 2,575 MW | TROPOMI-only +25.7 ppb, S/C=0 | ERA5 for TROPOMI | N/A (open-pit blind spot) | TROPOMI-only, high-uncertainty | ~800–1,500 |
| 5 | Turceni / Oltenia / ~2,200 MW | Phase 5 | ERA5 when run | ORIGINAL | Phase 5 — new detection run | TBD |
| 6 | Maritsa East 2 / AES-NEK / ~1,600 MW | Phase 5 | ERA5 when run | ORIGINAL | Phase 5 — new detection run | TBD |
| 7 | **Weisweiler** / RWE / 1,060 MW | S/C=23.5 | ERA5 incoming Apr 20 | ORIGINAL | WS2 pending Apr 20 | ~300–500 |
| 8 | Rovinari / Oltenia / ~1,320 MW | Phase 5 | ERA5 when run | ORIGINAL | Phase 5 — new detection run | TBD |
| 9 | **Lippendorf** / LEAG / 1,866 MW | S/C=155.4 original → **BT=0.19** | N/A — EXCLUDED | BITEMPORAL (proves terrain) | **EXCLUDED** (terrain_artifact) | 0 |
| 10 | **Rybnik** / PGE / ~1,775 MW | S/C=2.0, TROPOMI-consistent | ERA5 incoming Apr 20 | ORIGINAL | WS2 pending Apr 20 | ~80–200 |

*Plus two out-of-top-10 documentation rows: Maasvlakte (TROPOMI-confirmed, ERA5 6.16 m/s, Q=426 kg/h) and Groningen (EXCLUDED: cfar_suppressed_false_positive).*

### S/C Detection Summary

| Site | Capacity | Detection Rate | Mean S/C | TROPOMI ΔXCHâ‚„ | Verdict |
|------|----------|---------------|----------|----------------|---------|
| **Neurath** (RWE, DE) | 4,400 MW | **2/3** · p̂=0.67 [0.21, 0.94] | 45.1 | **+12.2 ppb ✓✓** | **DUAL-SENSOR CONFIRM** |
| **Bełchatów** (PGE, PL) | 4,830 MW | **3/3** · p̂=1.00 [0.44, 1.00] | 59.9 | — (cloud-blocked) | Strongest emitter in dataset |
| **Weisweiler** (RWE, DE) | 1,060 MW | 1/1 · p̂=1.00 [0.21, 1.00] | 23.5 | −7.7 ppb | S2 detection (TROPOMI negative) |
| Groningen (NAM, NL) | gas field | — | — | **−0.99 ppb** | **EXCLUDED**: CFAR-suppressed false positive |
| Maasvlakte (Uniper, NL) | 1,070 MW | 1/1 | — | TROPOMI confirmed | Canonical worked example |
| Lippendorf (LEAG, DE) | 1,866 MW | — | 155.4→0.19 BT | — | **EXCLUDED**: terrain artifact |

**The Neurath dual-sensor confirmation** (S/C=45 from S2 U-Net AND ΔXCH₄=+12.2 ppb from TROPOMI on the same date) is the headline result: two independent sensors detecting the same emission event.

**Bełchatów** — Europe's single largest CO₂ emitter — shows the strongest and most consistent S/C signal (peak S/C=143 on 2024-07-10), with TROPOMI confirmation blocked by afternoon convective cloud cover over central Poland.

**Data integrity**: Lippendorf (S/C=155.4) and Groningen are explicitly excluded from all financial outputs via `EXCLUDED_SITES` in `src/api/risk_model.py`. Their records are retained in `results_analysis/quantification.json` for provenance with `excluded=true`. The pre-integration snapshot is preserved at `results_analysis/quantification.pre_integration.json`.

---

## How It Works

```
Sentinel-2 L2A (B11/B12 SWIR, 20 m)
         │
         ▼
   CH₄Net U-Net  (13.5 M params, 64/128/256/512/512 ch, div_factor=1)
         │
         ▼
   Signal-to-Control ratio  =  mean(P_site) / mean(P_control)
         │
         ├── CFAR threshold: S/C > 1.15 + 3·σ_ctrl/μ_ctrl
         │
         ├── Bitemporal differencing (winter reference subtraction)
         │           removes permanent terrain SWIR features
         │
         ▼
   Detection verdict  +  Wilson 95% CI  +  p̂_detect
         │
         │    ┌── Sentinel-5P TROPOMI S5P L2 CH₄
         │    │   ΔXCHâ‚„ = site XCHâ‚„ − P20(annulus 0.25–1.0°)
         │    │   Multi-orbit search (3 daytime orbits/date, 11–14 UTC)
         │    └──────────────────────────────────────────
         ▼                                              │
   Dual-sensor assessment ◄──────────────────────────────┘
         │
         ▼
   EU ETS liability = p̂ × Q̂_CH₄ × GWP₁₀₀(29.8) × π_ETS(€60–65/tCO₂e)
         │
         ▼
   FastAPI  →  /portfolio-risk  ·  /site-risk/{name}
```

---

## Cross-Validation Design

The dual-sensor framework resolves a fundamental ambiguity in remote sensing: a high S/C ratio could be a genuine emission *or* a terrain artefact. TROPOMI provides independent confirmation:

| S/C | ΔXCHâ‚„ | Interpretation |
|-----|---------|---------------|
| > 1.15 | ≥ +5 ppb | ✓✓ **DUAL-SENSOR CONFIRM** — highest confidence |
| > 1.15 | ≈ 0 | ✓ S2-only — real but below TROPOMI sensitivity, or cloud-blocked |
| ≈ 1.0 | ≥ +5 ppb | ~ TROPOMI-only — S2 missed or bad scene |
| ≈ 1.0 | ≈ 0 | ✗ Non-detection |

TROPOMI S5P crosses central Europe at ~13:30 LST (ascending node). Sentinel-2 passes at ~10:30 LST. The 3-hour gap means afternoon cloud build-up (common in summer over Poland) can block TROPOMI even on clean S2 acquisition days — explaining Bełchatów's TROPOMI gaps without invalidating the S2 detections.

---

## Repository Structure

```
methane-api/
├── run_pipeline.py                  # Single entry point — runs all 6 stages
├── src/
│   ├── ingestion/
│   │   ├── copernicus_client.py     # Copernicus Data Space API wrapper
│   │   └── era5_client.py           # ERA5 wind retrieval (CDS API)
│   ├── quantification/
│   │   ├── cemf.py                  # CEMF spectral retrieval (Varon 2021)
│   │   ├── ime.py                   # IME inversion: Q = (M × U) / L
│   │   ├── uncertainty.py           # SSOT: ±30% ERA5, ±50% fallback
│   │   ├── canonical_writer.py      # QuantificationRecord + write_quantification_record()
│   │   ├── runner.py                # run_quantification() — steps 6–10 of provenance chain
│   │   └── emission_logger.py       # JSONL emission time-series logger
│   ├── api/
│   │   ├── main.py                  # FastAPI app (/portfolio-risk, /site-risk)
│   │   ├── schemas.py               # Pydantic v2 models
│   │   └── risk_model.py            # Wilson CI + EU ETS liability; EXCLUDED_SITES registry
│   ├── stress_testing/
│   │   ├── scenarios.py             # NGFS Phase IV (Orderly/Disorderly/Hot House)
│   │   ├── stress_test.py           # Beta×LogNormal×GBM Monte Carlo; Merton-KMV credit
│   │   └── credit_transmission.py   # Merton-KMV DD shift → rating migration
│   └── validation/
│       ├── model_validation.py      # ROC/AUC, calibration, Kupiec, isotonic, MC convergence
│       └── sensitivity.py           # Tornado OAT sensitivity (10 params)
├── scripts/
│   ├── run_quantification.py        # Canonical batch quantification (all sites)
│   └── legacy/
│       └── run_quant_fixed_v0.py    # [DEPRECATED] pre-ERA5 fallback winds
├── tests/
│   └── test_runner.py               # 20 unit tests (bitemporal rule, upsert, exclusions)
├── docs/
│   └── model_validation_summary.md  # SR 11-7 / BCBS 239 model validation one-pager
├── results_analysis/
│   ├── quantification.json               # Canonical CEMF+IME records (schema v1.0.0)
│   ├── quantification.pre_integration.json  # Pre-integration snapshot (provenance)
│   ├── multidate_validation.json         # S/C per site × date
│   ├── tropomi_validation.json           # TROPOMI ΔXCH₄ per site × date
│   ├── stress_test_results.json          # Monte Carlo stress output
│   └── methane_portfolio_risk_dashboard.ipynb  # Interactive results notebook
└── weights/
    └── european_model_v8.pth        # Fine-tuned CH₄Net v8 weights
```

---

## API

```bash
# Start the API
uvicorn src.api.main:app --reload

# Query portfolio risk
curl -X POST http://localhost:8000/portfolio-risk \
  -H "Content-Type: application/json" \
  -d '{"holdings": [{"ticker": "RWE", "site": "neurath"}, {"ticker": "PGE", "site": "belchatow"}]}'

# Query single site
curl http://localhost:8000/site-risk/neurath
```

Response schema:
```json
{
  "site": "neurath",
  "p_detect": 0.667,
  "p_detect_lo_95": 0.21,
  "p_detect_hi_95": 0.94,
  "mean_sc_when_detected": 45.1,
  "tropomi_confirmed": true,
  "eu_ets_liability_eur_annual": 1840000,
  "label": "confirmed"
}
```

---

## Running the Pipeline

```bash
# Activate conda environment
conda activate methane

# Full pipeline — all 6 stages (detection → quantification → stress → validation)
python run_pipeline.py

# Specific sites only (stages 1–3)
python run_pipeline.py --sites belchatow neurath maasvlakte

# Skip detection (tiles already processed), run quantification + stress only
python run_pipeline.py --skip-multidate --skip-tropomi

# Offline mode (no ERA5 / Copernicus downloads)
python run_pipeline.py --no-era5

# Dry run — print plan without executing
python run_pipeline.py --dry-run

# Canonical CEMF+IME+ERA5 quantification only
python scripts/run_quantification.py --sites neurath belchatow

# TROPOMI cross-validation (requires CDS_API_KEY)
export CDS_API_KEY=your-cds-key
python validate_tropomi.py --sites neurath belchatow weisweiler

# Run tests
python -m pytest tests/ -v
```

**Uncertainty:**  
Flow rate bounds use ±30% (ERA5 wind) or ±50% (climatological fallback).  
Source: `src/quantification/uncertainty.py` (SSOT).  
Derivation: WS2 Technical Report Section 4, quadrature of wind/plume/CEMF components (≈28%, rounded to 30%).

---

## Technical Details

**Model:** CH₄Net v2 U-Net, 13.5 M parameters
- Encoder: 64 → 128 → 256 → 512 → 512 channels
- `div_factor=1` (no channel halving at bottleneck)
- Trained on Sentinel-2 B11/B12 SWIR patches with synthetic plume augmentation (α=10–25%, Gaussian mask)
- Fine-tuned on European coal/lignite sites from JRC-PPDB-OPEN v1.0 (~7,117 EU power units)

**Detection signal:**
- S/C = mean(P_emission_buffer) / mean(P_control_region)
- Classical threshold: S/C > 1.15
- CFAR adaptive threshold: 1.15 + K·(σ_ctrl/μ_ctrl), K=3.0

**TROPOMI validation:**
- Product: S5P OFFL L2 CH₄ (COPERNICUS/S5P/OFFL/L2\_\_CH4\_\_\_)
- XCHâ‚„ variable: `methane_mixing_ratio_bias_corrected`
- QA filter: qa_value ≥ 0.5 (ESA scientific standard)
- Background: P₂₀ of 0.25–1.0° annulus (robust to neighbouring emitters)
- Orbit selection: ascending node 11:00–14:00 UTC, closest to 12:30 UTC, up to 3 orbits/date

**Risk model:**
- Detection probability: Wilson score interval (95% CI)
- EU ETS liability: p̂ × Q̂_CH₄ × GWP₁₀₀ × π_ETS
- Tail risk: 95th-percentile CI upper bound × liability

---

## Requirements

```bash
conda create -n methane python=3.12
conda activate methane
pip install torch torchvision sentinelhub netCDF4 fastapi uvicorn pydantic numpy requests
```

Copernicus Data Space account required for data download (free): https://dataspace.copernicus.eu

---

*Columbia University · Department of Industrial Engineering and Operations Research*  
*Sponsored by European Central Bank (ECB) and European Investment Bank (EIB)*
