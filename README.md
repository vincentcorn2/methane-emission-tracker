# Methane from Space: Satellite Detection and Quantification of Fugitive Coal Mine Emissions

**ECB / EIB Sponsored Research · Columbia University IEOR 4737 · 2025–2026**

Advisors: Prof. Ali Hirsa, Miao Wang (Columbia) · Dr. Oleg Reichmann (ECB) · Dr. Giuseppe Bonavolontà (EIB)

---

## What This Does

A production pipeline that turns freely available Sentinel-2 satellite imagery into statistically calibrated methane emission estimates and Carbon-Value-at-Risk figures for European coal mine operators — no corporate disclosure required.

The pipeline detects methane plumes using a domain-adapted deep learning model (CH4Net v8), applies a conformal-calibrated detection threshold with a finite-sample ≤10% false-positive guarantee, quantifies emission rates via CEMF+IME inversion with ERA5 wind retrieval, and translates the results into a Monte Carlo Climate VaR distribution for financial risk analysis.

---

## Key Results

**Primary case study: KWB Bełchatów, Poland** (Europe's largest coal mine, operated by PGE)

| Metric | Value |
|---|---|
| Above-threshold detections | 30 (spanning 2020–2024) |
| Detection-weighted annual estimate | 11,481 t CH₄/yr |
| 95% sampling CI | [6,563 – 16,400] t/yr |
| Recovery vs. Climate TRACE inventory | ~39% (consistent with published 30–60% range) |
| TROPOMI cross-validation | Sept 9 2021 — +12.7 ppb same-day confirmation |
| Conformal threshold τ (α=0.10) | 3.5796 (n=35 calibration sites, FPR ≤ 10%) |

**Monte Carlo Climate VaR (10,000 simulations):**

| Risk Metric | GWP100 (M€) | GWP20 (M€) |
|---|---|---|
| Mean expected liability | 20.3 | 60.3 |
| 95% Climate VaR | 36.4 | 107.8 |
| 99% Climate VaR | 48.1 | 142.6 |
| 99% Expected Shortfall | 54.5 | 161.5 |

**Rybnik finding:** The most externally validated site in the candidate set (5 TROPOMI enhancements, 4 Carbon Mapper quantified overpasses at 1,150–2,019 kg/hr) never clears the calibrated detection rule. The cause is a training distribution gap in Silesian industrial-fringe terrain — a documented limitation with direct implications for underground hard-coal methane monitoring.

---

## Repository Structure

```
methane-api/
│
├── report.md                     # Main paper
├── technical_appendix.md         # Full methodology and validation details
├── requirements.txt
│
├── src/                          # Core library modules
│   ├── detection/
│   │   └── ch4net_model.py       # CH4Net U-Net inference
│   ├── ingestion/
│   │   ├── copernicus_client.py  # Sentinel-2 download via CDSE API
│   │   ├── era5_client.py        # ERA5 wind retrieval (CDS API)
│   │   └── preprocessing.py      # Tile → .npy pipeline
│   ├── quantification/
│   │   ├── cemf.py               # CEMF spectral retrieval (Varon 2021)
│   │   ├── ime.py                # IME inversion: Q = (M × U) / L
│   │   ├── governance.py         # Uncertainty flags and penalty budget
│   │   ├── canonical_writer.py   # Structured quantification record writer
│   │   └── uncertainty.py        # Per-source uncertainty budget
│   └── stress_testing/           # Financial scenario modules
│
├── scripts/
│   ├── detection/                # CH4Net inference and production rule audit
│   ├── calibration/              # Conformal threshold, non-emitter expansion
│   ├── timeseries/               # Bełchatów / site annual timeseries runs
│   ├── quantification/           # CEMF+IME, ERA5 fetch, recompute annualisation
│   ├── finance/                  # Monte Carlo CVaR, deterministic stress scenarios
│   ├── validation/               # Bootstrap AUROC/AP, leakage audit, LOO stability
│   ├── analysis/                 # Site-specific: Rybnik wind test, TROPOMI co-location
│   └── archive/                  # Superseded one-off scripts
│
├── results_analysis/             # All pipeline outputs (JSON, MD, PNG)
│   ├── belchatow_annual_timeseries.json
│   ├── production_rule_audit.json
│   ├── calibrated_threshold.json
│   ├── quantification.json
│   ├── finance_climate_var.json
│   └── ...
│
├── data/
│   ├── npy_cache/                # Sentinel-2 tile cache (.npy)
│   ├── crops/                    # Training crops (positive / negative / synthetic)
│   └── nonemitter_tiles/         # Conformal calibration site tiles
│
├── weights/
│   └── european_model_v8.pth     # Production CH4Net weights (European fine-tune)
│
├── results/                      # Detection TIFs (bitemporal, nonemitter)
├── archive/                      # Early experiments, meeting docs, legacy pipeline
└── config/                       # Environment and API settings
```

---

## Running the Pipeline

```bash
conda activate methane
cd ~/Downloads/methane-api
```

**Run detection and production rule audit (uses τ=3.5796 auto-read from calibrated_threshold.json):**
```bash
python scripts/detection/apply_bitemporal_diff.py --sites belchatow
python scripts/detection/audit_production_rule.py
```

**Bełchatów annual timeseries + annualisation:**
```bash
caffeinate -i python scripts/timeseries/belchatow_annual_timeseries.py
python scripts/quantification/recompute_annualisation.py
```

**Run new site timeseries (e.g. Turów):**
```bash
python scripts/timeseries/run_new_site_timeseries.py --site turow --dry-run
caffeinate -i python scripts/timeseries/run_new_site_timeseries.py --site turow
```

**Conformal calibration (n=35):**
```bash
python scripts/calibration/run_mac_inference.py --phase 2
python scripts/calibration/run_mac_inference.py --phase 3
```

**Monte Carlo Climate VaR:**
```bash
python scripts/finance/finance_climate_var.py
# Output → results_analysis/finance_climate_var.json
```

**Bootstrap AUROC/AP (real crops only):**
```bash
python scripts/validation/bootstrap_auroc_ap.py
```

---

## Model

**CH4Net v8** — U-Net, 13.5M parameters, fine-tuned on European coal terrain

- Base weights: Vaughan et al. (2024) global pretrain
- Fine-tuning: 11 retraining experiments on European-specific dataset (14 TROPOMI-confirmed positives, 51 synthetic, 22 verified negatives)
- Detection: Signal-to-Control ratio with ratio-space CFAR adaptive threshold
- Conformal threshold: τ = 3.5796 at α=0.10 (n=35 non-emitter calibration sites, FPR ≤ 10% guaranteed)

---

## Key Dependencies

```bash
pip install torch torchvision numpy scipy rasterio pydantic requests cdsapi
```

- Copernicus Data Space account (free): https://dataspace.copernicus.eu
- CDS API key for ERA5 winds: https://cds.climate.copernicus.eu

---

## References

- Vaughan et al. (2024). CH4Net. *Atmospheric Measurement Techniques*, 17, 2583–2593.
- Varon et al. (2021). CEMF+IME. *Atmospheric Measurement Techniques*, 14, 2771–2785.
- Angelopoulos & Bates (2021). Conformal prediction. *arXiv:2107.07511*.
- Worden et al. (2025). NIST IR 8575 — methane plume quantification practices.
- Desnos et al. (2024). Climate VaR stochastic approach. *Amundi Institute Working Paper*.
- Reinders, Schoenmaker & van Dijk (2025). Climate risk stress testing. *Journal of Climate Finance*, 10.

---

*Columbia University · Department of Industrial Engineering and Operations Research*
*Sponsored by the European Central Bank (ECB) and European Investment Bank (EIB)*
