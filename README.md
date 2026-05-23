# Satellite-Based Coal Mine Methane Monitoring in Europe

**Columbia University · AI Applications in Finance (IEOR 4737) · European Central Bank & European Investment Bank**

| Student | Email |
|---|---|
| Konstantina Mavrogianni | konstantina.mavrogianni@columbia.edu |
| Vincent C. Cornelius | vcc2127@columbia.edu |
| Mihika Saraf | ms7321@columbia.edu |
| Yuhao Zhang | yz5075@columbia.edu |

**Academic Advisors:** Prof. Ali Hirsa · Miao Wang · **ECB & EIB:** Dr. Oleg Reichmann · Dr. Giuseppe Bonavolontà

---

## What This Does

A production pipeline that turns freely available Sentinel-2 satellite imagery into statistically calibrated methane emission estimates and Carbon-Value-at-Risk figures for European coal mine operators — no corporate disclosure required.

The pipeline detects methane plumes using a domain-adapted deep learning model (CH4Net v8), applies a conformal-calibrated detection threshold with a finite-sample ≤10% false-positive guarantee, quantifies emission rates via CEMF+IME inversion with ERA5 wind retrieval, and translates the results into a Monte Carlo Climate VaR distribution for financial risk analysis.

---

## Key Results

**Primary case study: KWB Bełchatów, Poland** (Europe's largest coal mine, operated by PGE)

| Metric | Value |
|---|---|
| Above-threshold detections | 31 (2019–2024) |
| Quantification-supporting | 30 (sc_cfar > τ, full CEMF+IME) |
| Detection-weighted mean flow | 476 kg/hr (95% CI: 341–612 kg/hr) |
| Annual emission estimate | 4,174 t CH₄/yr |
| 95% bootstrap CI | [2,987 – 5,360] t/yr |
| Recovery vs. Climate TRACE 2024 | 14.1% (95% CI: 10.1%–18.1%) |
| TROPOMI cross-validation | Sept 9 2021 — +12.7 ppb same-day confirmation |
| Conformal threshold τ (α=0.10) | 3.5796 (n=35 calibration sites, empirical FPR 5.7%) |

**Monte Carlo Climate VaR (10,000 simulations):**

| Risk Metric | GWP100 (M€) | GWP20 (M€) |
|---|---|---|
| Mean expected liability | 7.51 | 22.25 |
| 99% Value-at-Risk | 17.77 | 53.14 |
| 99% Expected Shortfall | 20.22 | 60.32 |
| CO₂e (deterministic) | 116,872 t/yr | 346,442 t/yr |

**Rybnik–Chwałowice finding:** The most externally validated site in the candidate set (5 TROPOMI enhancements, 4 Carbon Mapper quantified overpasses at 1,150–2,019 kg/hr) never clears the calibrated detection rule. Sub-threshold signals are present but the CFAR gate — inflated by Silesian industrial-fringe terrain heterogeneity — suppresses them below τ. The cause is a training distribution gap, not an absence of model response.

---

## Repository Structure

```
methane-api/
│
├── START_HERE.ipynb              # Interactive walkthrough — start here
├── technical_appendix.md         # Full methodology and validation details
├── requirements.txt
├── download_weights.py           # Fetch CH4Net weights from cloud storage
├── verify_outputs.py             # Sanity-check all pipeline outputs vs. paper
│
├── src/                          # Core OOP library modules
│   ├── detection/
│   │   └── ch4net_model.py       # CH4Net U-Net inference
│   ├── ingestion/
│   │   ├── copernicus_client.py  # Sentinel-2 download via CDSE API
│   │   ├── era5_client.py        # ERA5 wind retrieval (CDS API)
│   │   └── preprocessing.py     # Tile → .npy pipeline
│   ├── quantification/
│   │   ├── cemf.py               # CEMF spectral retrieval (Varon 2021)
│   │   ├── ime.py                # IME inversion: Q = (M × U) / L
│   │   ├── governance.py         # Uncertainty flags and penalty budget
│   │   ├── canonical_writer.py   # Structured quantification record writer
│   │   └── uncertainty.py        # Per-source uncertainty budget
│   ├── validation/               # Model validation and sensitivity analysis
│   ├── entity_resolution/        # Credit exposure and entity resolver
│   ├── stress_testing/           # Financial scenario modules
│   └── api/                      # FastAPI endpoints and risk schemas
│
├── scripts/                      # Module scripts + demo notebooks
│   ├── calibration/              # demo_calibration.ipynb — conformal threshold
│   ├── detection/                # Bitemporal diff and production rule audit
│   ├── finance/                  # demo_finance.ipynb — Monte Carlo CVaR
│   ├── quantification/           # demo_quantification.ipynb — CEMF+IME
│   ├── timeseries/               # demo_timeseries.ipynb — annual timeseries
│   └── validation/               # demo_validation.ipynb — AUROC, leakage, LOO
│
├── figures/                      # Paper figures (figure1–3, .py + .png; figure2_belchatow, figure3_timeseries)
│
├── data/
│   ├── crops/                    # Training crops (positive / negative / synthetic, 41 MB)
│   ├── nonemitter_tiles/         # Conformal calibration site tiles (116 KB)
│   ├── 16168_climate_trace_ch4.csv
│   └── rybnik_chwalowice_carbon_mapper.csv
│   # Excluded from git (re-downloadable):
│   # data/npy_cache/   — Sentinel-2 tile cache (211 GB, re-download via CDSE)
│   # data/downloads/   — TROPOMI NetCDF files (7.2 GB, re-download from Copernicus)
│
├── weights/                      # CH4Net model weights (excluded from git, ~52 MB each)
│   # Run: python download_weights.py
│
├── results_analysis/             # Canonical pipeline outputs (JSON, PNG, MD, HTML)
│   ├── calibrated_threshold.json
│   ├── belchatow_annual_timeseries.json
│   ├── finance_climate_var.json
│   ├── finance_transition_risk.json
│   ├── bootstrap_auroc_ap.json
│   ├── leakage_audit.json
│   ├── loo_detection_stability.json
│   ├── held_out_evaluation.json
│   ├── nonemitter_sc_scores.json
│   └── ...
│
├── config/                       # Environment and API settings
├── tests/                        # Integration test runner
└── scripts/archive/              # Legacy scripts, experiments, old results
    ├── docs/                     # Meeting notes and internal reports
    ├── scripts_analysis/         # Old one-off analysis scripts
    ├── scripts_archived/         # Superseded pipeline versions
    ├── experiments/              # Ablation and approach experiments
    ├── early_pipeline/           # Pre-OOP pipeline versions
    ├── legacy_results/           # v5–v7 model evaluations, intermediate outputs
    └── results_analysis_clutter/ # Old backup files and run logs
```

---

## Reproducing Results

**Interactive notebook (recommended):** open [START_HERE.ipynb](START_HERE.ipynb) — covers every result with inline explanations and a live API demo.

**Programmatic check:**
```bash
conda activate methane
cd ~/Downloads/methane-api
python verify_outputs.py --verbose   # confirm all outputs match paper
```

**Download model weights (required for inference):**
```bash
python download_weights.py   # downloads european_model_v8.pth
```

**Demo notebooks (no data download needed — reads pre-computed results):**
```bash
jupyter nbconvert --to notebook --execute scripts/calibration/demo_calibration.ipynb
jupyter nbconvert --to notebook --execute scripts/finance/demo_finance.ipynb
jupyter nbconvert --to notebook --execute scripts/validation/demo_validation.ipynb
jupyter nbconvert --to notebook --execute scripts/quantification/demo_quantification.ipynb
jupyter nbconvert --to notebook --execute scripts/timeseries/demo_timeseries.ipynb
```

**Full pipeline re-run (requires CDSE credentials and 211 GB tile cache):**
```bash
# Detection
python scripts/detection/apply_bitemporal_diff.py --sites belchatow

# Conformal calibration (n=35)
caffeinate -i python scripts/calibration/run_mac_inference.py

# Annual timeseries
caffeinate -i python scripts/timeseries/belchatow_annual_timeseries.py

# Monte Carlo Climate VaR
python scripts/finance/finance_climate_var.py

# Bootstrap AUROC/AP
python scripts/validation/bootstrap_auroc_ap.py
```

---

## Model

**CH4Net v8** — U-Net, div_factor=1, 13.5M parameters, fine-tuned on European coal terrain

- Base weights: Vaughan et al. (2024) global pretrain
- Fine-tuning: 11 retraining experiments on European-specific dataset (14 TROPOMI-confirmed positives, 51 synthetic, 22 verified negatives)
- Detection: Signal-to-Control ratio with ratio-space CFAR adaptive threshold
- Conformal threshold: τ = 3.5796 at α=0.10 (n=35 non-emitter calibration sites, empirical FPR 5.7%)

---

## Installation

```bash
conda create -n methane python=3.11
conda activate methane
pip install -r requirements.txt
```

Required external credentials:
- Copernicus Data Space (CDSE) account — https://dataspace.copernicus.eu
- CDS API key for ERA5 winds — https://cds.climate.copernicus.eu

See `requirements.txt` for full package list with versions.

---

## References

- Angelopoulos, A. N. and Bates, S. (2021). A gentle introduction to conformal prediction and distribution-free uncertainty quantification. *arXiv:2107.07511*.
- Carbon Mapper (2026). Carbon mapper data portal: Methane and CO2 super-emitter observations. https://carbonmapper.org/data. Accessed May 2026.
- Climate TRACE (2026). Climate TRACE emissions inventory and data platform. https://climatetrace.org. Accessed May 2026.
- Copernicus Climate Change Service (2026). ERA5 hourly data on single levels from 1940 to present. https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels. Accessed May 2026.
- Copernicus Data Space Ecosystem (2026). Copernicus data space ecosystem API documentation. https://documentation.dataspace.copernicus.eu. Accessed May 2026.
- Desnos, B., Le Guenedal, T., Morais, P., and Roncalli, T. (2023). From climate stress testing to climate value-at-risk: A stochastic approach. *SSRN Electronic Journal*.
- Ehret, T., De Truchis, A., Mazzolini, M., Morel, J.-M., d'Aspremont, A., Lauvaux, T., Duren, R., Cusworth, D., and Facciolo, G. (2022). Global tracking and quantification of oil and gas methane emissions from recurrent Sentinel-2 imagery. *Environmental Science & Technology*, 56(14):10517–10529.
- European Central Bank Banking Supervision (2026). Good practices for advancing climate and nature-related risk management. https://www.bankingsupervision.europa.eu. Accessed May 2026.
- European Parliament and Council (2024). Regulation (EU) 2024/1787 on the reduction of methane emissions in the energy sector. https://eur-lex.europa.eu.
- European Space Agency (2021). Copernicus Sentinel-5P TROPOMI Level 2 methane total column products, version 02. Accessed May 2026.
- GHGSat (2026). Satellite-based greenhouse gas emissions monitoring. https://www.ghgsat.com. Accessed May 2026.
- Intergovernmental Panel on Climate Change (2024). Global warming potential values from IPCC assessment reports. https://www.ipcc.ch. Accessed May 2026.
- PGE Polska Grupa Energetyczna S.A. (2025a). Consolidated financial statements for the year ended December 31, 2024. https://www.gkpge.pl. Accessed May 2026.
- PGE Polska Grupa Energetyczna S.A. (2025b). Credit rating. https://www.gkpge.pl/en/for-investors/bonds/rating. Accessed May 2026.
- Varon, D. J., Jervis, D., McKeever, J., Spence, I., Gains, D., and Jacob, D. J. (2021). High-frequency monitoring of anomalous methane point sources with multispectral Sentinel-2 satellite observations. *Atmospheric Measurement Techniques*, 14:2771–2785.
- Vaughan, A., Mateo-García, G., Gómez-Chova, L., Růžička, V., Guanter, L., and Irakulis-Loitxate, I. (2024). CH4Net: A deep learning model for monitoring methane super-emitters with Sentinel-2 imagery. *Atmospheric Measurement Techniques*, 17:2583–2593.
- Worden, J., Cusworth, D., et al. (2025). Common practices for quantifying methane emissions from plumes detected by remote sensing. *NIST Interagency/Internal Report 8575*.

---

*Columbia University · Department of Industrial Engineering and Operations Research*
*Sponsored by the European Central Bank (ECB) and European Investment Bank (EIB)*
