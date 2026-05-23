# Satellite Images for Emissions Tracking

## Meeting 6

__Date:__ Friday, April 24th, 2026

__Attendees__

Columbia University

1. Konstantina Mavrogianni
MSc in Financial Engineering
Email: __Konstantina.mavrogianni@columbia.edu__

1. Mihika Saraf
MSc in Business Analytics
Email: __ms7321@columbia.edu__

1. Vincent C. Cornelius
MSc in Operations Research
Email: __vcc2127@columbia.edu__

1. Yuhao Zhang
MSc in Operations Research
Email: __yz5075@columbia.edu__

Professor/TA

1. Dr. Ali Hirsa
Email: __ali.hirsa@columbia.edu__

1. Miao Wang
Email: __mw3302@columbia.edu__

European Central Bank / European Investment Bank

1. Dr. Oleg Reichmann
__oleg.reichmann@ecb.europa.eu__

1. Dr. Giuseppe Bonavolontà
Email: __g.bonavolonta@eib.org__

## Agenda

__Work Done Summary__ *(full technical detail in the supplemental report)*

1. Multi-Year Historical Backfill Across All 8 European Sites
Extended the pipeline from the two Netherlands point estimates shown at Meeting 5 into a 2019–2025 time series across all eight target sites: Weisweiler, Rybnik, Bełchatów, Lippendorf, Neurath, Boxberg (clean control), Groningen, and Maasvlakte. Each acquisition runs end-to-end — satellite ingest, CH4Net v8 inference, detection, and flow-rate retrieval with live ERA5 wind data — and writes to a single structured record. Bełchatów is the cleanest repeated signal (S/C = 849 in 2020, confirmed across 2020–2024); Boxberg correctly stays at non-detection. A class of records where the satellite tile did not actually cover the plant (partial-swath tiles) was identified as a silent failure mode; affected records were repaired and a pre-check was added to the pipeline to stop the issue recurring.

2. Statistically Rigorous Detection Threshold via Conformal Calibration
The old engineering heuristic for detection has been replaced with a modern distribution-free statistical method (split conformal prediction), calibrated against held-out non-emitter sites from CORINE Land Cover. The new production threshold carries a provable false-positive rate of ≤ 10% at finite sample size — i.e. a guarantee, not an approximation. This directly addresses the EIB's Meeting 5 question about calibrating detection against regulatory thresholds.

3. Quantification Pipeline Scaled to the 8 Sites, Sherwin Benchmarking Built
The flow-rate retrieval (CEMF spectral inversion + IME + ERA5 reanalysis wind) is now wired into the backfill loop — every above-threshold detection automatically produces a flow rate in kg/h, an uncertainty interval, an annualised tonnes figure, a 2026 IRA Waste Emissions Charge dollar number, and a full audit trail of inputs. In parallel, a benchmarking pipeline has been built against the Sherwin et al. (2024) single-blind controlled-release dataset — the only public multi-sensor known-truth methane benchmark. Five usable Sentinel-2 overpasses with ground-truth release rates between 5 and 1,496 kg/h have been identified; tile downloads are pending.

__Next Targets__

1. Model Validation & Testing on European Power Plant Sites
* Lock down backfill integrity before external validation — audit the 2019–2025 records and remove any from failed satellite passes where the tile did not actually cover the plant.
* Multi-date validation at Weisweiler and Bełchatów — run the pipeline on every cloud-free acquisition across 2019–2025 so each single-overpass detection becomes a multi-year persistence statistic.
* Parallel scale-up into the JRC-PPDB power plant database, prioritising Romanian lignite (Turceni, Rovinari) and Bulgarian Maritsa — depth over breadth: no new sites are added faster than existing ones acquire full uncertainty quantification.

2. Translate Pixels to Physical Flow Rates — Bundled WS1 × Sherwin Sprint
* Run the uncertainty decomposition script on all detections to replace the inherited ±40% Varon placeholder with a per-scene (σ_wind, σ_CEMF, σ_mask, σ_background) budget combined via 10k Monte Carlo.
* Replace the fixed 4×10⁻⁷ CEMF sensitivity scalar with a heteroscedastic MLP head trained against 6S/MODTRAN simulations.
* Train 5–10 deep-ensemble CH4Net heads for epistemic/aleatoric decomposition on the probability map.
* Fit conformalized quantile regression on Q against the Sherwin hold-out; target output "Q̂ = 338 kg/h, 80% conformal PI [245, 455]" with empirically verified coverage — the artifact that plugs directly into the existing Merton-KMV credit transmission.
* WS5 synthetic plume injection (formal LOD stress test) in parallel — fork the STARCOP codebase rather than rebuild.

3. GitHub Reproducibility Track
* Public-facing repository so collaborators and validators can reproduce every figure end-to-end from a clean machine: one-command conda setup, pinned requirements, Makefile-orchestrated pipeline, canonical JSON schemas, notebook walkthroughs, and a README framed in SR 11-7 three-pillar structure.

4. Five-Workstream Model Validation Framework
* Reframe CH4Net from "a detector we're improving" to "a live model under formal validation" — the same treatment a bank would apply to any model that feeds a risk calculation. Every output carries a traceable uncertainty budget and every design choice is documented against the three-pillar regulatory structure (SR 11-7 in the US, SS1/23 in the UK).
* Roadmap spans five workstreams: (1) uncertainty quantification, (2) foundation-model fine-tuning + out-of-distribution detection, (3) physics-informed plume transport, (4) multi-sensor fusion, (5) stress testing and controlled-release benchmarking.
* Workstreams 1 and 5 were seeded this cycle — the conformal detection threshold and the uncertainty-decomposition script both live here. Workstreams 2, 3, and 4 are the build-out priority over the next cycle.

__Questions for EIB / ECB__

1. Thank you for sharing the CSPP/PEPP holdings file. We cross-walked it against our eight monitored plants and found that four of the six repeatedly-emitting operators have bonds in the file — **RWE** (Weisweiler, Neurath), **EnBW** (Lippendorf), **Shell via NAM** (Groningen), and **Engie** (Maasvlakte). Given this overlap, what would be the most useful next step from your side? What shape does our emissions output need to take for it to plug into how your team actually uses data like this?

2. The other two repeatedly-emitting operators — **PGE** (Bełchatów and Rybnik, which are our strongest and most persistent detections) and **LEAG** (Boxberg, partial Lippendorf) — aren't in the CSPP/PEPP file. Is there a different dataset or exposure channel we should be looking at for these two? In particular, does the EIB have direct loan exposure to either of them, or should we frame them a different way in the analysis?

3. For internal model governance, does the Bank of England SS1/23 framework or the ECB guide on internal models map more directly onto how methane risk would be reviewed? Aligning vocabulary early avoids a painful documentation retrofit later.

4. For matching plumes back to specific facilities in the next sprint, does EIB have a preferred asset-identifier schema (EIC, GERD, or internal IDs) so that our emission trajectories integrate cleanly into existing portfolio systems without an extra join step?
