# scripts/archive/

Scripts moved here were used during development but are not directly referenced in the paper or technical appendix. They are preserved for reproducibility and audit purposes.

## Subdirectories

| Folder | Contents |
|--------|----------|
| `legacy/` | Earliest-phase scripts from before the CEMF+IME pipeline was stabilised. Includes v0 quantification, phase-5 record writers, and the Sherwin (2023) baseline comparison. |
| `analysis/` | One-off exploratory scripts: TROPOMI side-queries for Groningen/Weisweiler, plot generation, max-data experiments, wind alignment checks. Superseded by production timeseries scripts. |
| `data_pipeline/` | Data download and preparation utilities (nonemitter tile downloads, ERA5 backfill, training/eval tile fetches). Run once to populate caches; not needed for result reproduction once data is cached. |
| `detection/` | Audit and diagnostic scripts for the detection rule (`audit_detections.py`, `audit_production_rule.py`). Results are captured in `results_analysis/production_rule_audit.json`. |
| `finance/` | `run_stress_test.py` — early wrapper around the transition-risk model; superseded by `scripts/finance/finance_transition_risk.py`. |
| `quantification/` | One-off recomputation scripts: MBSP upgrade rerun, Neurath/Weisweiler CEMF variants, annualisation recomputes. Outputs are captured in `results_analysis/`. `era5_utils.py` — early standalone ERA5 helper superseded by `src/ingestion/era5_client.py`. |
| `timeseries/` | Superseded by `scripts/timeseries/timeseries_builder.py` + `timeseries_backfill.py`. Archived: `belchatow_annual_timeseries.py`, `rybnik_chwalowice_annual_timeseries.py` (consolidated into `BaseTimeseriesBuilder`); `historical_backfill_download.py`, `historical_backfill_timeseries.py`, `repair_backfill_coverage.py` (consolidated into `BackfillDownloader`, `BackfillTimeseriesBuilder`, `BackfillCoverageRepairer`). Also includes deprecated power-station-coordinates variant and data migration scripts. |
