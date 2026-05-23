# Model Validation Summary — CH4Net v8 + CEMF+IME + NGFS Stress Test

**Document type:** SR 11-7 / BCBS 239 Model Validation Summary  
**Model name:** CH4Net v8 Methane Detection + CEMF+IME Quantification + NGFS Climate Stress Test  
**Validation date:** April 2026  
**Primary model owner:** Vincent Cornelius  
**Version:** v1.0 (post-integration-plan Phase 0–4)

---

## 1. Model Purpose and Use

**What this model does:**  
Estimates per-facility methane emission flow rates for EU industrial sites using Sentinel-2 L1C satellite imagery, then translates those estimates into carbon liability exposure for equity holders and debt issuers under NGFS Phase IV climate transition scenarios.

**Primary use cases:**
- Portfolio-level climate stress testing under ECB Pillar 2 / NGFS Phase IV scenarios
- Per-site emission detection probability and annualised carbon liability (EU ETS + EU Methane Regulation)
- Credit risk transmission: Merton-KMV structural DD shift from carbon liability shock
- Regulatory reporting support: ESRS E1, PCAF financed emissions attribution

**What this model is NOT:**
- Not a real-time monitoring system (Sentinel-2 revisit is 5–10 days; EU-wide coverage is multi-month)
- Not a compliance enforcement tool (outputs are probabilistic estimates, not certified measurements)
- Not applicable to open-pit mine sites (CH4Net detects gas-phase plumes; open-pit surface emissions are a blind spot confirmed empirically at Boxberg, LEAG)
- Not a per-unit allocator within multi-unit facilities (spatial resolution 20m; report at facility level only)

---

## 2. Methodology and Assumptions

### Detection: CH4Net v8

- Architecture: U-Net with 13.5M parameters; encoder depth factor `div_factor=1`
- Input: Sentinel-2 L1C 12-band 10m image, 100×100 patch tiling with 50% overlap
- Training: fine-tuned from checkpoint via Approaches C–E (v5–v11); final v8 trained with focal+Dice loss, CFAR adaptive thresholding
- Detection criterion: S/C ratio > CFAR threshold (base 1.15 + 3.0 × CV_control_region)
- Bitemporal differencing: applied only to sites with known seasonal spectral contrast (Lippendorf, Groningen); disabled for industrial sites (confirmed empirically: Bełchatów S/C 27.3 → 1.94 under BT)

### Quantification: CEMF + IME

- CEMF: Continuous Enhancement Matched Filter, sensitivity 4×10⁻⁷ reflectance per ppb·m (Varon et al. 2021, AMT 14, 2771–2785, Section 2.2)
- Background: scene-derived matched filter using CH4Net non-plume pixels as reference population
- IME inversion: Q = (M_total × U_wind) / L_plume (single-overpass formulation)
- Wind: ERA5 10m U+V at 0.25° resolution, hourly, matched to scene acquisition timestamp via `src/ingestion/era5_client.ERA5Client.get_wind()`
- Fallback: 3.5 m/s climatological fallback when ERA5 unavailable (flagged in `wind_source`)

### Financial: NGFS Phase IV Stress Test

- Three canonical scenarios: Orderly Transition, Disorderly Transition, Hot House World
- Carbon price: GBM with mean-reversion to NGFS Phase IV price path; EU ETS as primary
- CH4 liability: annual_tCO2e = p_detect × Q × 8760 / 1000 × GWP-100 (29.8, IPCC AR6)
- EU Methane Regulation: Article 27 jump risk from 2028 (P=0.10/yr above threshold)
- EU ETS: €65/tCO2e base; CBAM excluded for domestic EU facilities (documented)
- IRA Waste Emissions Charge: retained as secondary comparator for US-jurisdiction readers

---

## 3. Data Lineage (BCBS 239)

| Stage | Source | Artefact |
|-------|--------|----------|
| Raw scene | Copernicus Open Access Hub (SAFE format) | `data/npy_cache/*.npy` + `*_geo.json` |
| Detection | CH4Net v8 inference | `results_validation/<site>/detection_*.tif` |
| S/C ratio | `validate_multidate.py` | `results_analysis/multidate_validation.json` |
| TROPOMI cross-val | S5P OFFL L2__CH4___ (qa≥0.5) | `results_analysis/tropomi_validation.json` |
| CEMF retrieval | `src/quantification/cemf.run_cemf()` | in-memory CEMFResult |
| ERA5 wind | `src/ingestion/era5_client.ERA5Client.get_wind()` | ERA5 NetCDF cache |
| IME inversion | `src/quantification/ime.CEMFIntegratedMassEnhancement.estimate_from_cemf()` | QuantificationResult |
| Canonical record | `src/quantification/canonical_writer.write_quantification_record()` | `results_analysis/quantification.json` |
| Risk scoring | `src/api/risk_model.RiskModel.site_risk()` | FastAPI `/site-risk/<site>` |
| Monte Carlo stress | `src/stress_testing/stress_test.StressTestEngine.run_portfolio_stress()` | `results_analysis/stress_test_results.json` |
| Credit transmission | `src/stress_testing/credit_transmission.merton_dd_shift()` | IssuerStressResult.{dd_baseline, dd_stressed} |

Every number in any shared output must be traceable to a row in this table. BCBS 239 Principle 2: data must be aggregatable, accurate, and timely.

---

## 4. Performance — Discrimination (AUC ± DeLong CI)

Run `ModelValidator().compute_roc()` for current figures.

Baseline (multidate_validation.json + tropomi_validation.json):
- **AUC**: see `full_validation_report()["auc"]`  
- **DeLong 95% CI**: computed via `ModelValidator._delong_ci()`
- **Optimal threshold** (max F1): typically S/C ≈ 1.15–1.20

Key reference sites:
- Neurath: S/C=45.1, TROPOMI +12.2 ppb → dual-sensor confirm (strongest positive)
- Weisweiler: S/C=23.5, TROPOMI mixed wind-direction signal
- Bełchatów: S/C=27.3 (original mask), 1.94 (BT mask — correctly discarded)
- Lippendorf: S/C=155.4 → EXCLUDED (terrain artifact; BT confirms S/C=0.19)

---

## 5. Performance — Calibration (Brier, ECE, Hosmer-Lemeshow)

Run `ModelValidator().compute_calibration()` and `._hosmer_lemeshow()` for current figures.

Expected calibration error (ECE) and Brier score are computed over site-date pairs where TROPOMI ground truth is available. Hosmer-Lemeshow χ²(8) tests bin-wise calibration.

**Isotonic calibration (PAV):** `ModelValidator().isotonic_calibration()` maps raw S/C → empirical P(TROPOMI confirms) using pool-adjacent-violators monotone regression. This is the SR 11-7 backtesting artefact for detection probability.

**Kupiec unconditional-coverage test:** `ModelValidator().kupiec_test()` tests whether the observed exceedance rate at 90% confidence equals the expected 10% rate. For the current dataset (n≈16 paired observations), the test has low power but the framework is structurally correct per Basel III VaR backtesting methodology (Kupiec 1995, JoD 2(4)).

---

## 6. Uncertainty Decomposition

Post-ERA5 quadrature budget (WS2 Technical Report, Section 4, April 2026):

| Source | Component σ (%) | Reference |
|--------|----------------|-----------|
| ERA5 wind speed (0.25° representativeness) | 15–17.5% | Varon et al. 2021, AMT Table 2 |
| Plume length (bbox vs. wind-axis projection) | 15–17.5% | IME formulation; see `ime.py` |
| CEMF spectral coefficient (4×10⁻⁷ uncertainty) | 10–12.5% | Varon et al. 2021, AMT Sec 2.2 |
| Pixel-edge quantisation (20m native) | ~8% | Geometric |
| **Total in quadrature** | **≈28%** → **30% (conservative)** | `src/quantification/uncertainty.py` |

SSOT: `src/quantification/uncertainty.UNCERTAINTY_PCT_ERA5 = 30`. All modules read from this constant. The fallback wind budget is ±50% (documented in `uncertainty.UNCERTAINTY_PCT_FALLBACK`).

---

## 7. Sensitivity (Tornado) and Stress (NGFS Phase IV)

**Tornado (OAT):** `src/validation/sensitivity.run_tornado()` perturbs 10 parameters one-at-a-time:

1. CEMF sensitivity coefficient (4×10⁻⁷ ±20%)
2. ERA5 wind speed (±30%)
3. Plume length projection (±20%)
4. CFAR K-factor (3.0 ±0.5)
5. S/C threshold (1.15 ±0.05)
6. Uncertainty % (30 ±5)
7. EU ETS price (€65 ±€15)
8. CH4 GWP-100 (29.8 ±2.0, IPCC AR6 vs. AR5/AR7)
9. Discount rate (3% ±1%)
10. EU Methane Regulation multiplier (1.0 ±0.2)

Each perturbation re-runs the Monte Carlo at n=10,000 paths (subsampled for speed). Results are ranked by |ΔVaR95| and displayed as a horizontal tornado chart.

**NGFS scenarios:** Orderly, Disorderly, Hot House World — calibrated to ECB 2024 Pillar 2 Climate Stress Test reference paths (`src/stress_testing/scenarios.py`).

---

## 8. Limitations and Out-of-Scope

| Limitation | Affected sites | Mitigation |
|-----------|---------------|-----------|
| Open-pit terrain blind spot | Boxberg (LEAG, 2,575 MW) | Documented as `ch4net_blind_open_pit`; TROPOMI-only estimate flagged as high-uncertainty |
| Lippendorf terrain artifact | Lippendorf (LEAG, 1,866 MW) | **EXCLUDED** from all financial outputs; `exclusion_reason = terrain_artifact` |
| Groningen CFAR false positive | Groningen (NAM, gas field) | **EXCLUDED**; `exclusion_reason = cfar_suppressed_false_positive` |
| Bełchatów: BT kills real signal | Bełchatów (PGE, 4,830 MW) | CEMF run on original mask; BT mask retained for documentation only |
| Multi-unit non-allocation | Neurath (2 units × 1,100 MW) | Report at facility level; `facility_unit_mw_breakdown` in JSON for reference |
| Climatological wind fallback | Sites without ERA5 retrieval | ±50% uncertainty; flagged in `wind_source` |
| CBAM scope | Bełchatów, Rybnik, Rovinari, Turceni | CBAM does NOT apply to domestic EU facilities; documented explicitly |

---

## 9. Ongoing Monitoring

- **TROPOMI expansion:** `validate_tropomi.py` extended to 2025 full-year (Phase 5+)
- **Multi-date weighted emissions:** replace `p_detect × Q_annual` with date-weighted emission integral when ≥5 detection dates are available
- **JRC top-10 coverage:** Phases 2 and 5 fill Neurath (ERA5 pending), Weisweiler, Rybnik, Turceni, Rovinari, Maritsa East 2
- **Model drift monitoring:** track S/C ratio distribution and AUC over rolling 90-day windows as new Sentinel-2 tiles are processed

---

## 10. Governance

| Role | Responsible | Cadence |
|------|------------|---------|
| Primary model owner | Vincent Cornelius | Per commit |
| WS2 quantification | Mihika (CEMF+IME+ERA5) | Per April 20 deliverable |
| Model validation review | Opus (external review agent) | Per integration plan milestone |
| Independent review | ECB/EIB submission review | Before v1.0.0 tag |

Versioning: `results_analysis/quantification.json` carries `schema_version = "1.0.0"`. Breaking changes to the schema require a version bump and changelog entry. The pre-integration snapshot is preserved at `results_analysis/quantification.pre_integration.json`.

---

*This document is generated as part of the WS1+WS2 Integration Plan (April 2026). For methodology details, see: `src/quantification/`, `src/stress_testing/`, `src/validation/`. For data lineage, see the canonical provenance chain in Part 3 of the Integration Plan.*
