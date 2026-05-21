# Methane from Space: Detecting and Quantifying Fugitive Emissions at European Coal Mines Using Satellite Imagery and Deep Learning

**AI Applications in Finance (IEOR 4737)**
European Central Bank & European Investment Bank

*Academic Advisors: Professor Ali Hirsa, Miao Wang*
*Industry Advisors: Dr. Oleg Reichmann (ECB), Dr. Giuseppe Bonavolontà (EIB)*

---

## Abstract

European coal mine methane is an unquantified liability for institutions financing the sector's transition: operators self-report through activity-based estimates, and no independent, calibrated monitoring system with a verifiable false-positive rate exists for this asset class. We adapt CH4Net to European terrain through eleven retraining experiments, derive a conformal detection threshold with a finite-sample false-positive guarantee of ≤10%, and apply it with CEMF+IME inversion across eight candidate facilities. At KWB Bełchatów, 30 above-threshold detections and 26 quantification-supporting observations spanning 2019–2025 yield a detection-weighted annualised estimate of 16,486 t CH₄/yr (95% CI: 6,781–26,191 t/yr; this is a cloud-free-conditioned estimate under a continuous-emission assumption, not a full annual mass balance) — 56% of the Climate TRACE inventory, consistent with published Sentinel-2 recovery ranges for analogous industrial sources. Notably, these detections occur at a site the model was explicitly trained to treat as a non-emitter, making the result a direct test of spectral generalisation rather than label reproduction. At Rybnik — independently validated by TROPOMI and Carbon Mapper — the calibrated rule never fires; the cause is a training-distribution gap in Silesian industrial-fringe terrain, not a fundamental detection limit. Institutions relying on optical satellite monitoring for underground hard-coal operations should treat such signals as systematically underrepresented rather than zero-emitting. The Bełchatów detections translate into an illustrative annual carbon-cost exposure of €32.31 M (GWP100, EU ETS-equivalent pricing; 95% CI: €13.29–€51.33 M) — currently unpriced in any operator disclosure. *These figures are illustrative scenario estimates under stated policy assumptions, not forward projections of booked liabilities.*

---

## Supervisory Summary

*Self-contained overview for ECB, EIB, and institutional risk officers. Full methodology and validation in §1–§9 and the companion technical appendix.*

> **What this pipeline is.** An independent, satellite-based emission monitor built on freely available Sentinel-2 imagery — no operator cooperation required, no corporate disclosure assumed, updateable with every cloud-free pass.
>
> **What it found.** At KWB Bełchatów (Europe's largest coal mine), 30 above-threshold detections and 26 quantification-supporting observations spanning 2019–2025 yield a detection-weighted annual estimate of **16,486 t CH₄/yr**, translating to an illustrative carbon exposure of **€32.31 M/yr** under EU ETS-equivalent pricing.
>
> **What it cannot find — and why this matters.** At Rybnik — validated by both TROPOMI and Carbon Mapper — the calibrated rule never fires. Underground hard-coal operations in Silesian industrial terrain are currently below the detection horizon and should be treated as unverifiable by this approach, not as zero-emitting.

---

**Key Numbers**

| Metric | Value |
|---|---|
| Conformal threshold τ (α = 0.10) | 3.5796 |
| Bootstrap 90% CI on τ | [2.49, 4.34] |
| Calibration sites (n) | 35 |
| Above-threshold responses (Bełchatów, 2019–2025) | 30 above-threshold total; 26 quantification-supporting (1 TROPOMI cross-validated) |
| Quantification-supporting observations | 26 |
| Mean per-overpass flow rate | 1,882 kg CH₄/hr |
| Detection-weighted annualised estimate† | 16,486 t CH₄/yr |
| 95% CI on annual estimate† | [6,781, 26,191] t/yr |
| Climate TRACE reported total (2024) | 29,636 t CH₄/yr |
| Recovery ratio | 56% (95% CI: 23%–88%) |
| Implied annual carbon cost (central, GWP100) | €32.31 M at €70/tCO₂e |
| Monte Carlo Climate VaR 99 (GWP100) | €71.5 M |

*† Detection-weighted estimate conditioned on cloud-free observable overpasses and a continuous-emission assumption. Not a direct measurement of annual throughput — see §6.4 and §8.*

*‡ Climate TRACE note: Bełchatów has two separate asset entries — the open-pit coal mine (asset 16168, IPCC sector 1B1a, methane-dominant) and the adjacent power station (separate asset, sector 1A1a, near-zero methane). All comparisons in this paper use the mine entry. A lookup returning the power station will show no methane and incorrectly appear to contradict our detections — see §6.2 for full disambiguation.*

---

**Key Terms**

| Acronym | Definition |
|---|---|
| S/C | Signal-to-control ratio: facility crop mean ÷ mean of four offset control crops |
| CFAR | Constant False Alarm Rate: adaptive threshold that scales with local background heterogeneity |
| τ | Conformal detection threshold (τ = 3.5796 at α = 0.10 in this paper) |
| CEMF | Column-Enhancement Mass Flux: converts probability maps to column methane mass |
| IME | Integrated Mass Enhancement: inversion linking plume mass, wind speed, and emission rate |
| ERA5 | ECMWF Reanalysis v5 — hourly gridded atmospheric product used for wind retrieval |
| TROPOMI | Tropospheric Monitoring Instrument aboard Sentinel-5P (atmospheric methane sensor) |
| BT | Bitemporal differencing — seasonal reference subtraction; disabled at continuous emitters |
| MRV | Measurement, Reporting, and Verification (EU Methane Regulation framework) |

---

> ## Executive Brief: Five Things a Risk Officer Needs to Know
> *For ECB, EIB, and institutional risk officers — full methodology in Sections 2–8*
>
> ---
>
> **1. This pipeline monitors coal mine methane from space — independently of the operator.**
> It uses freely available Sentinel-2 satellite imagery with a statistically guaranteed false-positive rate of ≤10%. No operator cooperation, corporate disclosure, or proprietary data is required. It can be updated with every cloud-free satellite pass (~15–25/yr at European latitudes).
>
> **2. At Europe's largest coal mine, we found a confirmed multi-year emission signal.**
> KWB Bełchatów (PGE, Poland): **30 detections over 2019–2025**, every one in a month when Climate TRACE independently reports 1,700–3,000 t CH₄ emitted. Detection-weighted annual estimate: **16,486 t CH₄/yr** (95% CI: 6,781–26,191 t/yr). Cross-validated on the strongest date by TROPOMI (+12.7 ppb, same-day). Critically, these detections occur at a site the model was *explicitly trained to treat as a non-emitter* — the result is a test of generalisation, not memorisation.
>
> **3. The carbon exposure is material and currently unpriced.**
> At EU ETS-equivalent rates (GWP100, €70/tCO₂e): **€32.31 M/yr** illustrative exposure (95% CI: €13.29–€51.33 M). Under GWP20 — the relevant horizon for 5–10 yr investment decisions — the comparable figure is **€95.80 M/yr**. This is additional unpriced carbon risk on top of PGE's €2.54B existing EUA book. All figures are illustrative of the scale of unpriced exposure, not forward estimates of booked liabilities.
>
> **4. Underground hard-coal mines are a blind spot — treat as unverifiable, not zero.**
> Rybnik (Silesia) is confirmed by five TROPOMI enhancements and four Carbon Mapper quantifications at 1,150–2,019 kg/hr — the most externally validated site in this study. The pipeline never fires there. The cause is a training-distribution gap in complex industrial-fringe terrain. **Any portfolio methane risk assessment using optical satellite monitoring should treat underground hard-coal operations as currently unverifiable by this method, not as zero-emitting.**
>
> **5. The methodology has a citable false-positive guarantee and documented boundaries.**
> The conformal calibration provides a finite-sample FPR ≤10% — suitable for citation in compliance or supervisory documents. The Rybnik failure mode is explicitly mapped, giving supervisors a documented scope boundary. Copernicus data (the satellite source) is explicitly endorsed in ECB Banking Supervision's 2026 good practices compendium (Elderson, 2026).
>
> ---
>
> **Three immediate priorities:** *(i)* Retrain on Silesian underground-mine crops — fixable gap, not a detection limit. *(ii)* Expand the conformal calibration set from n=35 toward n≥40 for continent-scale deployment. *(iii)* Apply the financial scenario module across the peer portfolio of Central European coal operators, not just PGE.

---

## 1. Introduction

**The problem.** Coal mine methane is not directly measured. European operators report through activity-based estimates, national inventories aggregate rather than attribute, and lenders or supervisors have no independent view of what a given facility is actually emitting. The gap between reported and actual emissions is not merely a scientific uncertainty: for institutions financing or regulating fossil fuel assets, it is an unpriced liability. Undetected methane can imply unpriced carbon liabilities, misaligned ESG assessments, and potential regulatory exposure as the EU Methane Regulation (Regulation 2024/1787) and related MRV requirements come into force.

**What existing methods miss.** Satellite-based methane detection has advanced rapidly, but two problems remain unsolved at European coal mining sites. First, capable deep learning models were trained on arid oil-and-gas terrain — visually simple landscapes where plume contrast is large. European coal mines sit in temperate agricultural and industrial landscapes with high spectral variability; applying existing models directly produces false-positive rates of approximately 48% on verified non-emitter sites. Second, no published pipeline has derived a statistically calibrated detection threshold for European coal terrain. A threshold with a defensible, finite-sample false-positive guarantee — the kind a financial analyst or regulator could actually rely on — does not yet exist for this problem.

**What we built.** We adapt CH4Net, a deep learning model for Sentinel-2 methane detection, to European terrain through eleven documented retraining experiments, and derive a conformal-calibrated detection threshold with a quantified false-positive rate. The central finding is this: *calibrated Sentinel-2 methane detection can produce facility-level monitoring at large European open-pit coal mines that is illustrative of the scale of unpriced carbon exposure, but current optical pipelines systematically underperform in industrial-fringe underground coal terrain.* The first half of that claim is demonstrated at the KWB Bełchatów lignite mine in Poland; the second half — equally important — is demonstrated at the Rybnik underground hard-coal complex in Silesia.

**Main findings.** The model produces 30 above-threshold detections at KWB Bełchatów — a site it was explicitly trained to treat as a non-emitter (label_value = 0 on the background tile). The model is, in other words, disagreeing with its own training label on the basis of spectral evidence alone. This makes the multi-year detection record a direct test of generalisation rather than a test of memorisation, and it aligns with an independent inventory across every detection month. At Rybnik — which has stronger external validation than any other candidate site — the calibrated rule never fires; the cause is an identifiable training-distribution gap in Silesian industrial-fringe terrain, documented in Section 5. Full quantitative results are in Section 6.

**Contributions.** This paper makes six concrete contributions. First, a European domain adaptation of CH4Net with documented training-set shifts, eleven retraining experiments, and held-out validation. Second, a conformal-calibrated detection threshold replacing the heuristic 1.15 threshold — a methodologically sharper foundation than is typical in applied remote-sensing work. Third, operational handling for four real-world failure modes rarely addressed together: partial-swath data artifacts, denominator-collapse in S/C ratios, terrain-heterogeneity-driven CFAR adaptation, and cloud-driven temporal sparsity. Fourth, a facility-level quantification pipeline integrating Sentinel-2 output, ERA5 winds, and the IME/CEMF framework across 26 observations. Fifth, an empirical characterisation of where the current European detection frontier lies — including a systematic failure mode with direct implications for how institutions should interpret optical satellite monitoring of underground hard-coal operations. Sixth, direct evidence against training-set memorisation: the model produces 30 above-threshold responses at Bełchatów despite the site being labelled as a non-emitter in training, with zero date-overlap between training crops and evaluation acquisitions confirmed by formal leakage audit.

---

*[**Figure 1 — Methodology flow diagram** — Production brief for figure creator:*

*Two-column flowchart, left column = detection path, right column = quantification path, joined at the detection decision node.*

*Left column (detection):* **(1)** Copernicus CDSE API → Sentinel-2 L1C tile download → **(2)** Partial-swath fingerprint check: `site_mean == ctrl_mean`? → YES → reclassify as `no_coverage` (exit); NO → **(3)** CH4Net v8 inference (12-band, 100×100 px) → per-pixel probability map → **(4)** S/C ratio: facility crop mean ÷ mean of 4 offset control crops → **(5)** CFAR background test: `sc_cfar > τ = 3.5796`? (conformal threshold, α=0.10, n=35 non-emitter calibration sites) → NO → non-detection record; YES → **Detection confirmed.***

*Right column (quantification), branching from Detection confirmed:* **(6)** ERA5 CDS API → wind speed U at overpass time → **(7)** MBSP retrieval: scene-derived band-scaling factor c (Varon et al. 2021 Eq. 3) applied to OSM mine polygon boundary → dXCH₄ column enhancement map → **(8)** IME inversion: Q = (M × U_eff) / L → emission rate estimate (kg CH₄/hr) + per-source uncertainty budget → **(9)** Canonical record written to `belchatow_annual_timeseries_mbsp.json`.*

*Style notes: use rounded boxes for process steps, diamonds for decisions, cylinder for data stores (CDSE, CDS APIs), bold border on Detection confirmed node. Colour: blue for detection path, orange for quantification path. Target width: full column (≈140 mm for two-column journal). Font: 8 pt minimum.]*

---

## 2. Data: Sentinel-2 and What It Measures

Sentinel-2 (ESA Copernicus) produces multispectral imagery at 10–60 m resolution with a nominal ~5-day revisit at temperate latitudes from the 2A/2B constellation, and we access Level-1C tiles (top-of-atmosphere reflectance) via the Copernicus Data Space Ecosystem API.

Methane detection relies on two specific bands. **Band 11 (B11)** captures light in the 1,560–1,660 nanometer range, where methane is essentially transparent — it passes through without interacting. **Band 12 (B12)** captures light in the 2,090–2,290 nanometer range, where methane absorbs strongly. When a methane plume is present above a patch of ground, B12 is dimmer than it would otherwise be — the plume is blocking some of the sunlight that would normally reflect off the surface and reach the satellite. B11, being unaffected, tells the model what the surface should look like without methane. The contrast between B12 and B11 is the physical signature the detection model learns to recognize.

Our pipeline caches downloaded tiles as compressed NumPy arrays and runs detection on 100-by-100 pixel patches centered on the facility coordinates of interest, corresponding to roughly one square kilometer on the ground.

---

## 3. The Detection Model: CH4Net

### 3.1 What the Model Does

CH4Net is a U-Net convolutional neural network trained to identify methane plumes in Sentinel-2 imagery, producing a per-pixel probability map where values near 1 indicate likely plume pixels (Vaughan et al., 2024).

CH4Net takes all 12 Sentinel-2 bands rather than just B11/B12; the additional visible and near-infrared channels help distinguish methane absorption from other phenomena that also darken B12. Vaughan et al. demonstrated that the full-band configuration outperforms B11/B12-only on all detection metrics, which motivated our adoption of it.

CH4Net runs on a single image — no reference scene required. This matters for continuous emitters: any "clean" reference at a coal mine also contains methane absorption, so reference-subtraction methods partially cancel the signal they are designed to detect; we confirmed this empirically at Bełchatów (Appendix C).

### 3.2 Why the Model Needed Adaptation for Europe

CH4Net was originally trained on 23 super-emitter sites between 2017 and 2020, primarily in the Turkmenistan oil and gas region. The landscapes there are arid desert with sparse infrastructure — a visually simple environment where plume contrast against background is large. European coal mine sites are fundamentally different: Bełchatów sits in central Poland surrounded by agricultural fields, mixed forest, rivers, and road infrastructure whose surface spectral variability far exceeds the arid training distribution. The original model, applied directly, produced a false-positive rate of ~48% on verified European non-emitter sites.

We addressed this by fine-tuning CH4Net on a European-specific dataset — 14 TROPOMI-confirmed positive crops, 51 synthetic plume injections, and 22 verified negatives — across eleven retraining experiments. The eighth configuration (**v8**) correctly handled all identified failure modes without suppressing real signals and is the production model throughout this paper. To stress-test whether performance depends on the 14 real crops or merely on the synthetic injections, we retrained from the same base weights using only the 51 synthetic positives: training collapsed to an all-negative solution and the model produced zero detections on the 14 real positive crops, against v8's 14/14. Real positive crops are essential — synthetic augmentation amplifies them but cannot replace them. Full training details and per-experiment outcomes are in Appendix A (technical_appendix.md §1 and §5.3).

---

## 4. Building a Statistically Reliable Detection Rule

The detection pipeline reduces to a single question: does the facility stand out above its local background strongly enough to survive statistical calibration? Four mechanisms operationalise that question; full derivations and diagnostic tables are in Appendix B (technical_appendix.md §2).

**Signal-to-control ratio.** For each overpass we extract the mean CH4Net probability within a 100 × 100-pixel crop centred on the facility, then divide by the average of four identically sized control crops offset ~22 km in each cardinal direction. A ratio near 1 means the model is no more confident at the facility than in the surrounding landscape; a ratio well above 1 means the facility stands out.

*In plain terms: we ask whether the model is more confident about methane at the facility than in the surrounding countryside on the same day — not whether the model's output is high in absolute terms.*

**The problem with simple thresholds.** Applying the published S/C > 1.15 heuristic to 35 verified non-emitter sites across Europe produced a 46% false-positive rate — roughly half of all "detections" would be spurious. This is not a defensible basis for financial analysis.

*In plain terms: a smoke detector set this loosely would be triggering in nearly half the rooms with no fire.*

**Conformal calibration.** We replaced the heuristic with a threshold derived from conformal prediction (Angelopoulos and Bates 2021), which provides finite-sample guarantees on false-positive rates. Calibrating at α = 10% on the 35-site set yields τ = 3.5796; the empirical false-positive rate is 5.7%, and a 2,000-resample bootstrap gives a 90% CI of [2.49, 4.34] on τ.

*In plain terms: instead of guessing a threshold by inspection, we set it empirically from known-clean sites and obtain a mathematical guarantee on the maximum false-alarm rate — the same principle as calibrating a medical test on a control group before diagnostic use.*

**Leakage and stability audits.** A temporal leakage audit confirmed zero training/evaluation date pairs within a 14-day window at any candidate site. Leave-one-out analysis across all 27 above-threshold Bełchatów scenes showed detection-rate stability within 0.73 percentage points regardless of which scene is removed; one outlier (August 2022, sc_cfar = 515) influences the mean S/C ratio but not the binary classification outcome.

*In plain terms: the model hasn't seen the answer sheet — it generalises.*

---

## 5. Site Selection: Eight Candidates, One Primary

We evaluated eight European facilities spanning lignite and hard-coal mines, lignite power stations, and gas infrastructure, including two designated control sites (one clean non-emitter, one false-positive trap from the original model). Full per-site backfill results, partial-swath repair logs, and date-by-date detection outcomes are in Appendix D (technical_appendix.md §3).

| Site | Country | Type | Role in study |
|---|---|---|---|
| **KWB Bełchatów** | Poland | Open-pit lignite mine | Primary case study |
| **Rybnik** | Poland | Underground hard-coal complex (JSW/PGG) | Most externally validated; non-detection |
| **Neurath** | Germany | Lignite power station | Dual-sensor cross-validation anchor |
| **Lippendorf** | Germany | Lignite power station | Candidate; single isolated reading |
| **Weisweiler** | Germany | Lignite power station | Candidate; partial-swath gaps |
| **Boxberg** | Germany | Lignite power station | Clean control (designated non-emitter) |
| **Groningen** | Netherlands | Gas field | False-positive control |
| **Maasvlakte** | Netherlands | Gas terminal | False-positive control |

Bełchatów — signals in three distinct calendar years, multi-year persistence, Climate TRACE coverage, and facility scale — was selected as primary case study before intensive monitoring began. Neurath produced two above-threshold responses in 2024 (one TROPOMI-confirmed); Lippendorf a single isolated reading with no preceding signal across four years; Boxberg, Groningen, and Maasvlakte zero responses, the expected outcome for the clean control and false-positive traps. Of 139 Bełchatów acquisitions, 65 were partial-swath — reclassified as missing observations by a fingerprint check rather than counted as non-detections (Appendix D).

### 5.1 Rybnik: What Happens When You Apply Power-Station-Trained Detection to Underground Hard Coal

Rybnik is the most externally validated site in the candidate set — five TROPOMI enhancements (10.85–19.66 ppb above background), six Carbon Mapper overpasses with four quantified emission rates ranging 1,150–2,019 kg CH₄/hr. Two instruments, three organisations, persistent agreement that this complex emits at scale. The calibrated rule never fires.

The model is not blind to Rybnik. On March 22, 2025 — one day after Carbon Mapper measured 2,019 kg/hr — it produced a raw S/C of 5.48, well above the 1.15 naive threshold. A designed falsification confirms the response is not a static terrain artifact: the probability centroid bearing flipped 172° between March 21 and March 22, tracking a wind direction change. A fixed surface feature cannot do that. The CFAR score on March 22, however, was only 0.42 — well below τ = 3.5796 — because the extreme spectral heterogeneity of Silesian industrial-fringe background lifts the adaptive threshold above what the signal achieves. Full centroid-wind geometry is in technical_appendix.md §5.3; diagnostic acquisition tables (TROPOMI and Carbon Mapper) are in Appendix D (§3.3).

The failure is a training-distribution gap, not a fundamental detection limit. CH4Net v8 was fine-tuned primarily on power-station and oil-and-gas terrain. Underground hard-coal venting in complex industrial fringe terrain — Silesian collieries surrounded by urban infrastructure and legacy industry — was under-represented, the same way the original CH4Net was biased toward Turkmenistan desert terrain before domain adaptation. The fix is the same: retrain on confirmed Silesian scenes. Until then, **optical satellite monitoring of underground hard-coal operations in industrial-fringe terrain should be treated as systematically undercounting, not as zero-emission evidence.**

---

## 6. Results at Bełchatów

**Detection** and **quantification** operate in different uncertainty regimes. Detection is relatively strong: the conformal threshold provides a calibrated false-positive bound, every above-threshold date aligns with Climate TRACE's independent inventory, and the model fires at this site despite its negative training label. Annual quantification is weaker: it is conditioned on a non-representative sample of cloud-free overpasses, carries structural seasonal bias, and propagates wind and spatial-extent uncertainty through the CEMF inversion. The 16,486 t/yr figure should be read as a detection-conditioned estimate under a continuous-emission assumption, not a direct measurement of annual throughput.

### 6.1 From Power Station to Coal Mine: How the Analysis Evolved

Early analysis centred on the Bełchatów power station before facility-level inventory data confirmed the adjacent open-pit mine as the methane-dominant asset (IPCC sector 1B1a vs. sector 1A1a). Re-running the pipeline on the correct mine coordinates with an OSM polygon boundary produced two validating outcomes: the MBSP retrieval under the old 5 km power-station crop collapsed by 95% (6,739 → 320 kg/hr), confirming the prior heuristic had been inflating results over spectrally heterogeneous surface outside the mine boundary; and the detection record did not disappear when the crop shifted — it strengthened and aligned more cleanly with Climate TRACE's monthly inventory. Full five-variant crop-comparison data are in technical_appendix.md §4.1.

### 6.2 The Facility

KWB Bełchatów is an open-pit lignite mine in central Poland, operated by PGE Polska Grupa Energetyczna. It is the largest coal mine in Europe by excavated volume and one of the largest single-point sources of carbon dioxide on the continent, primarily through the adjacent Bełchatów power station which burns the extracted coal. For purposes of this analysis, the mine and the power station are distinct Climate TRACE assets. The mine — Climate TRACE asset 16168 at coordinates 51.242°N, 19.275°E *(verify asset ID against the current Climate TRACE inventory before submission, as IDs may be reassigned across inventory versions)* — is classified under IPCC sector 1B1a (fugitive emissions from coal mining) and is methane-dominant. The power station is classified under sector 1A1a (electricity generation) and is carbon dioxide-dominant, with near-zero reported methane. Any comparison of our results against Climate TRACE must use the mine entry, not the power station; a lookup that returns the power station will show no methane and appear to contradict our detections, which would be misleading.

### 6.3 The Detection Record

We ingested 139 Sentinel-2 acquisitions over Bełchatów spanning 2019 through 2025. Of these, 65 were reclassified as missing observations due to the partial-swath data quality issue described in Section 5 (Appendix D). The remaining 74 acquisitions entered the detection analysis as valid observations. Of those, 26 produced sufficient model output to support a CEMF+IME quantification estimate using the OSM mine polygon boundary. The 30 above-threshold responses in the combined record are composed of two groups: 27 detections from the 2021–2024 intensive monitoring period that satisfied both the calibrated S/C threshold (sc_cfar > τ) and the CFAR gate, plus 3 additional detections from the 2020–2024 candidate backfill evaluation (including the June 2020 detection at S/C = 849). Eight further records fell below the conformal threshold but carried enough model signal for a valid IME inversion. The September 9, 2021 acquisition — the highest-S/C date in the record — carries same-day TROPOMI cross-validation (§6.6 below); the remaining 29 above-threshold dates are calibrated-rule-confirmed only.

Every detection date from 2021 onward falls inside a month where Climate TRACE independently reports the mine emitting at the 1,700–3,000 tonne level. Non-detection dates cluster in winter and shoulder-season months — when cloud cover, reduced solar elevation, and occasional snow cover suppress Sentinel-2's SWIR sensitivity. The mine does not stop emitting in winter; the satellite cannot see clearly enough to detect it.

The above-threshold record also guards against label-reproduction: Bełchatów was included in v8 training as a negative example, labeled as a non-methane site on the August 2024 background tile. The model nonetheless produces 30 above-threshold responses — it is disagreeing with its own training label on the basis of spectral evidence, which is a stronger generalization argument than a held-out test provides. This is demonstrated at a single site; whether the behaviour holds across a broader set of European emitter types remains an open question.

### 6.4 Quantifying Emission Rates

To convert the spatial probability maps into physical emission rate estimates, we apply the Column-Enhancement Mass Flux (CEMF) method, following Varon et al. (2021). This approach integrates the mass of methane enhancement implied by the CH4Net probability map over a defined area, then divides by the estimated atmospheric transport time using wind speed data from the ERA5 reanalysis product (produced by the European Centre for Medium-Range Weather Forecasts, and available through the Copernicus Climate Data Store). The wind provides the key link between a snapshot of a plume at one moment and an estimate of how fast methane is flowing out of the source.

We applied the MBSP retrieval (Varon et al. 2021, scene-derived band-scaling factor *c*) to the OSM mine polygon boundary across all 26 quantification-supporting observations. ERA5 wind data was retrieved at the satellite overpass time for each date; all 26 records used ERA5 reanalysis winds rather than climatological fallback values, which would have required an additional uncertainty penalty.

Estimated emission rates across the 26 records ranged from 72 to 9,957 kg CH₄/hr, with a median of 612 kg/hr and a mean of 1,882 kg/hr. The wide range is physically meaningful: emissions from an open-pit mine vary with excavation activity, wind conditions, and atmospheric stability, and the model detects more easily on high-emission days, biasing the sampled distribution upward. Per-overpass uncertainty sources — wind retrieval (±20%), sensitivity coefficient (±15%), mask boundary (±5–7%), and temporal sampling bias — are quantified in Appendix H; the temporal sampling term dominates the annualised uncertainty.

The 95% CI on the mean rate (t-distribution, df=25) runs from 774 to 2,990 kg/hr. Under a continuous-emission assumption and conditional on the cloud-free overpasses in this sample, the annualised projection is **16,486 t CH₄/yr** (95% CI: 6,781–26,191 t/yr). This is a detection-weighted, cloud-free-conditioned estimate — not a full annual mass balance. It captures what the satellite could observe; it does not measure what the mine emitted on the ~350 days per year the sensor cannot see clearly.

### 6.5 Comparison to the Independent Inventory

Climate TRACE reports 29,636 tonnes of methane from asset 16168 for the full year 2024; our annualised mean estimate is 56% of that figure (95% CI: 23%–88%). The recovery fraction is not a discrepancy. The 56% ratio is consistent with published Sentinel-2 recovery ranges for analogous industrial point sources. Vaughan et al. (2024) document a 30–70% recovery band for global super-emitter ensembles under single-overpass multispectral retrieval — our 56% falls well within that range and is consistent with the physics of single-overpass SWIR sensing in temperate latitudes, where cloud cover, low solar elevation, and seasonal surface variability all compress recoverable signal relative to arid-terrain benchmarks. Ehret et al. (2022) report comparable fractions for oil-and-gas point sources. Neither paper addresses coal-mine fugitive sources specifically — coal-mine-specific Sentinel-2 recovery benchmarks are sparse in the published literature — so this comparison is treated as a cross-source plausibility check rather than a direct coal-mine calibration. Three structural factors drive the shortfall in the same direction: we detect preferentially on favorable atmospheric days, have zero observations from cloud-covered winter months, and the model's probability maps over-predict spatial extent in ways that push recovery estimates downward rather than upward. The 56% ratio reflects day-weighted flow rates compared to a full-year inventory; it is not evidence of missed plumes on observed days.

This is consistent with independent literature on Sentinel-2 recovery for industrial point sources and reflects the physics of temperate-latitude single-overpass sensing — cloud frequency, low solar elevation angle, and seasonal surface variability all compress recoverable signal relative to arid-terrain benchmarks — rather than a failure of the detection pipeline.

Climate TRACE rates its 2024 confidence for this asset as "low." Neither estimate is a ground-truth measurement; their order-of-magnitude agreement is the validation we can responsibly claim.

**Table — Multi-instrument validation summary**

| Site | CH4Net detections (above τ) | TROPOMI enhancements | Carbon Mapper overpasses | Climate TRACE alignment | Conformal threshold status |
|---|---|---|---|---|---|
| **KWB Bełchatów** | **30** (26 with full quantification; 1 TROPOMI cross-validated) | 1 same-day (+12.7 ppb, 2021-09-09) | None (no campaign) | Every detection month falls in a 1,700–3,000 t reported month | **Fires** — primary case study |
| **Rybnik** | **0** | 5 enhancements (+10.85–19.66 ppb, 2023–2024) | 6 overpasses; 4 quantified (1,150–2,019 kg/hr, 2023–2026) | N/A — never triggers | **Never fires** — training-distribution gap in Silesian industrial-fringe terrain |
| **Neurath** | **2** (2024-06-25, 2024-08-29) | 1 same-day (+12.2 ppb, 2024-06-25) | None | Limited (2024 only; no multi-year record) | Fires — dual-sensor cross-validation anchor |
| **Boxberg** | 0 | None | None | N/A — designated clean control | Correctly silent — clean control |

*Note: Rybnik is the most externally confirmed methane source in the candidate set yet the pipeline never fires there. The cause is a documented training-distribution gap, not a fundamental detection limit — see §5.1 and Appendix D.*

### 6.6 Independent Atmospheric Confirmation

We searched for coincident TROPOMI measurements on every above-threshold Bełchatów date. On September 9, 2021, TROPOMI recorded a +12.7 ppb enhancement at the Bełchatów coordinates against a background annulus of 0.25–1.0° — the same date our pipeline recorded its highest S/C ratio in the full record. Both instruments responding most strongly on the same day is the result expected if both are responding to methane, not surface artifacts.

The remaining above-threshold CH4Net dates did not yield usable TROPOMI retrievals — either the Sentinel-5P swath did not cover the tile or quality filtering removed too many cloud-contaminated pixels. Coincident retrievals are the exception at this latitude; the Climate TRACE inventory comparison is the primary cross-validation for the broader record.

---

**Key Findings in Plain Language** *(for non-technical readers)*

- **We can monitor coal mine methane from space — for free.** Using publicly available Sentinel-2 satellite imagery, the pipeline detects methane plumes above large open-pit coal mines with a statistically guaranteed false-alarm rate of ≤10%, requiring no operator cooperation or corporate disclosure.
- **At Europe's largest coal mine, we found a multi-year emission signal.** KWB Bełchatów produced 30 confirmed detections over 2019–2025. The detection-weighted annual estimate — 16,486 t CH₄/yr — is consistent with an independent inventory and was confirmed on its strongest date by a second, entirely different satellite sensor (TROPOMI).
- **The resulting carbon cost is material but currently unpriced.** At EU ETS-equivalent rates, the satellite-estimated methane translates to approximately €32 M/yr in carbon exposure — entirely absent from operator disclosures and from the credit models built on those disclosures.
- **Underground coal mines are a blind spot.** At Rybnik — independently confirmed as a large methane source by two other instruments — this pipeline never triggers. The cause is identifiable and fixable, but until it is fixed, optical satellite monitoring should be treated as systematically undercounting underground hard-coal operations, not as evidence they are zero-emitting.
- **The method is transparent about what it cannot see.** The 56% recovery ratio relative to an independent inventory is not a failure — it reflects the physics of single-pass optical sensing in temperate Europe, where cloud cover and winter darkness limit usable observations to roughly 15–25 per year.

---

## 7. Illustrative Transition Risk Translation (Scenario-Based)

> **Key terms for this section**
>
> | Term | Plain meaning |
> |---|---|
> | **GWP100** | Global Warming Potential over 100 years — the EU MRV regulatory conversion factor (1 t CH₄ = 28 t CO₂e) |
> | **GWP20** | Global Warming Potential over 20 years — a stricter metric (1 t CH₄ = 83 t CO₂e) reflecting near-term climate forcing; more relevant for 5–10 yr investment horizons |
> | **EU ETS / EUA** | EU Emissions Trading System; European Union Allowance — the carbon price instrument (currently ~€60–80/tCO₂e) |
> | **EU MRV** | EU Measurement, Reporting and Verification framework — the regulatory compliance standard that governs how emissions are counted and priced |
> | **Climate VaR** | Climate Value-at-Risk — the carbon-liability exposure at a given percentile of the Monte Carlo distribution (analogous to financial VaR) |
> | **Expected Shortfall (ES)** | Average loss in the worst-case tail beyond the VaR percentile; a more conservative risk measure |

ECB Banking Supervision has noted explicitly that climate and nature-related financial risks "may be underestimated within the financial system" due to "non-linear dynamics and compounding events" that remain poorly understood (Elderson, 2026). Unverified facility-level methane represents one concrete instance of that underestimation: it is a liability that exists in physical reality but is absent from operators' disclosed figures and therefore from the credit models built on those figures. The scenarios below illustrate the scale of that gap for a single asset under policy assumptions.

**This section is not a market prediction model.** It maps the methane emission estimates from Section 6 into carbon-price exposure figures under explicit policy assumptions and presents three illustrative stress scenarios for a hypothetical single-name position.

All results are conditional on two stated assumptions: (i) regulatory adoption of a methane pricing framework comparable to the EU ETS, and (ii) the detection pipeline's central estimate being representative of the facility's true emission rate. Neither is a conclusion of this study. In particular, the liability figures assume full methane pricing at EU ETS-equivalent rates — a policy outcome that does not currently exist for coal mine fugitive emissions and may not materialise on the timescales modelled. Readers should treat the figures as illustrative of the transmission channel, not as forecasts of actual financial exposure.

For ECB and EIB specifically, the primary contribution of this section is not the single-name figures: it is the demonstration that calibrated satellite methane data can be translated into the language of transition-risk stress testing under a defined and auditable set of assumptions. Three properties of the pipeline make it usable in an institutional context: (i) the conformal false-positive guarantee of ≤10% means its outputs can be cited in a compliance or supervisory document rather than treated as directional; (ii) the Rybnik finding explicitly maps where the methodology fails, giving supervisors a documented boundary on what the monitoring covers; and (iii) the pipeline uses only freely available data, imposing no access asymmetry between institution and operator.

> **Stylized stress analysis disclaimer.** Position sizes are hypothetical, shocks are illustrative tiers from the transition-risk literature, and carbon-cost figures are exposure proxies, not booked liabilities. All numbers are reproducible from `scripts/finance/finance_transition_risk.py`.

The issuer is **PGE Polska Grupa Energetyczna S.A.** (WSE:PGE), the state-controlled Polish utility that owns and operates KWB Bełchatów through its GiEK subsidiary. PGE carries investment-grade ratings (Moody's Baa1, Fitch BBB, stable as of 2025) and a 2026 market capitalisation of approximately USD 6.34 billion.

Key 2024 financials (PGE consolidated annual report): net loss EUR 717M (second consecutive year; EUR 1,083M in 2023); net debt EUR ~2.1B. PGE holds **EUR ~2.54B in CO₂ emission allowances** on its current balance sheet — already one of Europe's largest single-entity EUA exposures. The Conventional Generation segment posted a 2024 EBIT of −EUR 1.7B after EUR 2.1B in depreciation and write-downs, reducing segment PPE from EUR 4.9B to EUR 3.0B in a single year. The coal fleet is in structural balance-sheet decline; the satellite-estimated methane liability is additional unpriced carbon risk on top of an asset base already contracting. PGE's own CSRD/ESRS climate disclosures identify rising CO₂ allowance costs as a strategic threat — the same transmission channel this pipeline quantifies independently.

### 7.1 Stochastic Uncertainty Propagation — Methodology

We propagate uncertainty via 10,000 Monte Carlo simulations following Amundi (2024) and NIST IR 8575 (2025); full methodology in Appendix H.

### 7.2 Implied Carbon-Cost Exposure

Coal-mine methane is not currently covered under the EU ETS (Phase IV), but the Methane Regulation sets the trajectory. The figures below represent order-of-magnitude exposure if a comparable price regime applied. Converting the pipeline's central estimate (16,486 t CH₄/yr) via IPCC AR5 GWP100 (factor 28) gives 461,608 t CO₂e/yr. Sensitivity across EUA price scenarios:

| Price case | €/tCO₂e | Mean estimate | Lower CI bound | Upper CI bound |
|---|---:|---:|---:|---:|
| Low | 50 | €23.08 M | €9.49 M | €36.67 M |
| Central | 70 | €32.31 M | €13.29 M | €51.33 M |
| Upper | 95 | €43.85 M | €18.04 M | €69.67 M |

To place these figures in context: PGE's existing CO₂ allowance book is EUR ~2.54B (end-2024). The methane satellite estimate at the central EUA price adds approximately 1.3–3.8% to that existing carbon exposure depending on GWP metric — small as a fraction of the allowance book, but entirely unpriced and unaudited without independent monitoring. Under the central case the implied annual exposure is approximately €32.31 M at GWP100. The EU MRV framework uses GWP100 as its regulatory metric, making it the appropriate basis for compliance exposure estimates. GWP20 (factor 83, IPCC AR6) is the more relevant horizon for near-term transition risk: it reflects the actual climate forcing over the 5–10 year investment horizons within which most coal asset repricing is expected to occur.

**GWP Sensitivity — Carbon-Cost Exposure (Central EUA price: €70/tCO₂e)**

| | GWP100 (factor 28, EU MRV) | GWP20 (factor 83, IPCC AR6) |
|---|---:|---:|
| Annual CO₂e equivalent | 461,608 t | 1,368,338 t |
| Mean estimate | €32.31 M | €95.80 M |
| 95% CI lower bound | €13.29 M | €39.42 M |
| 95% CI upper bound | €51.33 M | €152.19 M |

The ~3× difference between GWP100 and GWP20 is not a modelling choice — it reflects which time horizon the exposure is being priced over. GWP100 is the regulatory compliance metric, governing EU MRV reporting and ETS-equivalent pricing. GWP20 is the investor-relevant metric: stranded-asset repricing unfolds over 5–10 year horizons driven by near-term physical forcing and policy action, not over a century. Supervisory exercises focused on near-term financial stability should present both.

**Monte Carlo Climate Value-at-Risk (10,000 simulations)**

The stochastic engine described in §7.1 propagates all five uncertainty layers jointly. The mean expected liability (€29.03M GWP100 / €86.04M GWP20) lies ~10% below the deterministic base case, reflecting the Beta(9,1) enforcement pass-through (mean 90%); this gap is intentional and documented in Appendix H.

**Uncertainty decomposition — what drives the tail** *(full parameter specs in Appendix H)*

| Source | Relative σ | Share of total variance |
|---|---:|---:|
| Carbon price trajectory (log-vol 30%) | 0.300 | **77%** |
| Emission sampling (26-obs CI) | 0.286 | 70% |
| ERA5 wind bias (±10%) | 0.100 | 9% |
| Plume spatial extent (±15%) | 0.087 | 7% |
| **Combined (quadrature)** | **0.435** | — |

The dominant driver is carbon price uncertainty, not satellite measurement precision. Improving emission estimates would reduce the second row but leave the dominant variance untouched. As a robustness check: even if the true annual emission rate were 2× our central estimate (32,972 t/yr), carbon-price trajectory would still account for 77% of total variance — the liability distribution widens but its shape is governed by regulatory uncertainty, not by the satellite data. **The wide tail in Climate VaR reflects whether and when a comparable price regime applies — not a measurement problem.**

| Risk Metric | GWP100 (M€) | GWP20 (M€) |
|---|---:|---:|
| Mean expected liability | 29.03 | 86.04 |
| Median | 26.65 | 79.01 |
| 95th percentile (Climate VaR 95) | 55.01 | 163.05 |
| 99th percentile (Climate VaR 99) | 71.50 | 211.96 |
| **99% Expected Shortfall** | **81.44** | **241.41** |

**Bottom line for a risk committee:** At the central EUA price (GWP100), the satellite-derived methane liability represents an unpriced exposure equivalent to approximately 1.3% of PGE's existing EUR 2.54B carbon allowance book — rising to 3.8% under GWP20. The figure is small relative to the allowance book but is entirely absent from operator disclosures and from the credit models built on those disclosures.

*Reproducibility: `python scripts/finance/finance_climate_var.py` — seed=42, n_sim=10,000. Full parameter table and simulation structure in Appendix H.*

### 7.3 Credit-Spread and Equity Stress

*The €10 million position size is chosen for arithmetic convenience; all P&L figures scale linearly.*

A hypothetical €10 million senior unsecured PGE position at 5-year modified duration (CR01 ≈ €5,000/bp) produces mark-to-market bondholder losses of €75K, €175K, and €250K under Mild (+15 bp), Moderate (+35 bp), and Severe (+50 bp) spread tiers respectively — corresponding qualitatively to an ESG-rating downgrade, a Methane-Regulation enforcement action, and a sustained inventory-vs-satellite gap entering public discussion. A hypothetical €10 million long-equity position under the same tiers (−3%, −7%, −10%) produces P&L outcomes of −€300K, −€700K, and −€1.00 M.

### 7.4 Combined Sensitivity Grid

| Tier | ΔEquity | ΔSpread | Equity P&L (€10M long) | Bond P&L (€10M IG) | Combined P&L |
|---|---:|---:|---:|---:|---:|
| Mild | −3% | +15 bp | −€300 K | −€75 K | **−€375 K** |
| Moderate | −7% | +35 bp | −€700 K | −€175 K | **−€875 K** |
| Severe | −10% | +50 bp | −€1.00 M | −€250 K | **−€1.25 M** |

The combined-channel P&L (−€375K to −€1.25M) is modest relative to PGE's capitalisation but meaningful at portfolio-position scale. The three channels are not necessarily independent: in a scenario where the inventory-vs-satellite gap becomes policy-salient, ESG-sensitive funding costs and equity valuations could move together rather than sequentially. Treating them additively may understate drawdown risk under that specific scenario.

**A note on coverage.** The Rybnik finding implies that optical-satellite estimates will undercount underground coal methane. For PGE this bias is limited — Bełchatów is open-pit lignite — but any continent-wide application to Polish, Czech, or German hard-coal operators should pair satellite signals with conservative inventory-based estimates for the underground subset.

More broadly, systematic deployment across European coal operators would create the data-availability conditions Reinders, Schoenmaker, and van Dijk (2025) identify as a prerequisite for coordinated asset reassessment under regulatory disclosure. Whether markets would respond materially is conditional on regulatory adoption and investor reaction; this study does not model either.

---

*[**Figure — What works / what fails / why** — Production brief for figure creator (three horizontal panels, full-page width):*

***Panel A — Bełchatów: confirmed multi-year signal*** *(left panel, ~40% width)*
*Bar chart or scatter plot: x-axis = date (2019–2025, quarterly ticks); y-axis = sc_cfar score (log scale); horizontal dashed red line at τ = 3.5796 (labelled "conformal threshold α=0.10"). Plot all 30 above-threshold sc_cfar values as filled circles; plot sub-threshold valid observations as open circles at their actual sc_cfar. Annotate: (i) 2021-09-09 with a star and "TROPOMI +12.7 ppb"; (ii) 2020-06-01 with "sc_cfar = 600 (strongest signal)". Colour: above-threshold = dark blue; sub-threshold = light grey. Inset mini-table or annotation strip below x-axis: shade Q1/Q4 months light grey labelled "structurally low visibility." Data source: `results_analysis/belchatow_annual_timeseries_mbsp.json` + `results_analysis/production_rule_audit.json`.*

***Panel B — Rybnik: confirmed source, zero detections*** *(centre panel, ~35% width)*
*Two-part panel. Top: map or schematic (≈60% of panel height) showing the Rybnik source pin (Carbon Mapper, 50.0781°N, 18.5451°E) as a red star; the CH4Net probability-weighted centroid on 2025-03-22 (2.56 km SW, bearing 223.65°) as a blue dot with arrow; ERA5 wind vector on 2025-03-22 (from east, 4.84 m/s) as a labelled grey arrow; previous day Carbon Mapper wind (from 215.7°, bearing 35.7° predicted plume) as a dashed orange arrow labelled "CM wind 2025-03-21". Label the 172° centroid flip between the two arrows. Bottom: small bar or table showing sc_cfar = 0.42 vs. τ = 3.5796 (CFAR gate prevents detection despite raw S/C = 5.48). Data source: `results_analysis/rybnik_centroid_vs_wind.json`.*

***Panel C — Structural ceiling*** *(right panel, ~25% width)*
*Stacked or grouped bar chart: x-axis = calendar quarter (Q1–Q4); y-axis (left) = number of above-threshold detections across 2019–2025; y-axis (right, secondary) = estimated cloud-free acquisition frequency (% of passes) drawn as a line. Bars: Q1 = 0 detections (all years); Q2 = peak detections; Q3 = high detections; Q4 = near-zero detections. Line: cloud cover climatology at 51°N, showing Q1/Q4 at 65–75% cloud cover and Q2/Q3 at 45–55%. Caption should note: "The detection record is a mirror image of cloud-free availability — the mine does not stop emitting in winter; the satellite cannot see clearly enough to detect it." Data source: detection dates from `production_rule_audit.json`; cloud climatology from ERA5 or Copernicus Climate Atlas for 51°N, 19°E.*

*Style notes: three panels in a single row separated by thin vertical rules; shared caption below. Panel letters A/B/C in bold top-left of each panel. Target: full text width (≈180 mm two-column or ≈85 mm single-column). Font 8 pt minimum. Colourblind-safe palette (blue/orange/grey). No fill colours in map panel — use outline boundaries only for the facility footprint.]*

### 7.5 Policy Relevance and Operational Use Cases

The following three use cases describe how the pipeline's outputs could be operationalised by ECB, EIB, or institutional risk officers without requiring modification to the methodology. Each is distinct in scope and requires no additional modelling beyond what is demonstrated in this paper.

| # | Use case | Primary actor | What the pipeline provides | Scope boundary |
|---|---|---|---|---|
| 1 | **MRV validation support** | National competent authority; ECB/EIB supervisory teams | Independent satellite-derived emission estimate, calibrated FPR ≤10%, citable in a compliance audit context | Flagging mechanism, not a definitive measurement; Copernicus data explicitly endorsed in ECB 2026 good practices (Elderson, 2026) |
| 2 | **Portfolio screening for high-emission coal assets** | Credit risk officers; supervisory portfolio teams | Multi-year, multi-instrument detection record independent of operator disclosure | Qualitative screen only; does not replace financial due diligence or credit ratings |
| 3 | **Prioritisation for targeted high-resolution sensing** | Institutions seeking ground-truth quantification | Triage signal meeting Carbon Mapper / GHGSat tasking documentation requirements | Prior probability, not a rate estimate; routes facilities to targeted campaigns rather than requiring blanket coverage |

- **MRV validation support under EU Regulation 2024/1787.** The EU Methane Regulation requires active coal mines to submit continuous methane monitoring data and member states to establish independent verification mechanisms. The pipeline described here is directly usable as a source-independent verification layer: it produces facility-level emission estimates from freely available satellite data, with a calibrated false-positive rate of ≤10% that can be cited in a compliance audit context. Where an operator's self-reported figure diverges materially from the satellite-derived estimate, the pipeline identifies the discrepancy for follow-up — not as a definitive measurement, but as a flagging mechanism within an MRV framework. Notably, ECB Banking Supervision's 2026 good practices compendium explicitly identifies Copernicus programme data — the same satellite infrastructure underlying our Sentinel-2 pipeline — as an endorsed public tool for quantifying physical risks at individual asset level (Elderson, 2026, footnote 4).

- **Portfolio screening signal for high-emission coal assets.** A lender or supervisory authority can use the detection record as a qualitative screening input: sites with multi-year, multi-instrument-confirmed above-threshold responses represent a different category of transition-risk exposure than sites with no observable signal. The pipeline does not replace financial due diligence. It does provide an independent, updateable data source not derived from operator disclosure — closing a gap that credit ratings and ESG scores currently leave open. ECB Banking Supervision's 2026 good practices compendium identifies facility-level transition risk assessment — "assessing transition risks at the individual client level" rather than relying on sectoral averages — as an emerging supervisory good practice (Elderson, 2026). This pipeline operationalises that approach using satellite data rather than client-reported figures.

- **Prioritisation tool for targeted high-resolution sensing.** Carbon Mapper's Tanager programme and GHGSat's commercial constellation both accept tasking requests for specific facilities, but tasking capacity is limited and requires documented evidence of likely methane activity. The pipeline's Sentinel-2 outputs serve directly as that prior: the September 9, 2021 TROPOMI confirmation at Bełchatów, combined with the CH4Net detection record, constitutes the signal quality that Carbon Mapper's public-data programme requires before scheduling a quantification overpass. For any institution seeking ground-truth confirmation at a coal asset, this pipeline provides the triage layer that routes facilities toward targeted high-resolution sensing rather than requiring blanket coverage.

---

## 8. Limitations

The most important limitation is the Rybnik finding: the pipeline cannot currently confirm detections at the most externally validated coal mine methane source in the candidate set. The undercount is not random — it is concentrated in underground hard-coal, which is the dominant mining sector in Poland and elsewhere in Central Europe. The cause is a training-distribution gap in Silesian industrial-fringe terrain. It is fixable, but unfinished. Training-distribution analysis is in technical_appendix.md §5.3; extended seasonal analysis is in Appendix G.

The calibration set at n=35 remains moderate. The conformal guarantee holds at any finite n, and the bootstrap CI on τ has tightened to [2.49, 4.34] as the set expanded from 25 to 35, with all five ecoregion strata now at n≥6. A continent-scale deployment would require n≥40 with fuller Atlantic and Continental coverage — the recommended next step. The pilot results are valid within this scope.

The temporal coverage of the detection record is near the structural ceiling for Sentinel-2 in central Poland — approximately 15–25 cloud-free acquisition opportunities per year at 51°N, with Q1 and late Q4 structurally near-invisible to SWIR optical sensors. The 16,486 t/yr annualised figure should be read as a detection-conditioned estimate, not a direct measurement of annual throughput. Seasonal variation, partial-fix options (Sentinel-2C, cloud masking), and instrument complementarity are discussed in Appendix G. A natural follow-on is multi-instrument fusion: combining Sentinel-2 detections with TROPOMI column enhancements (which operate independently of solar illumination constraints) and, where available, Sentinel-2C's reduced revisit interval could meaningfully extend winter coverage and tighten the annual uncertainty band without requiring new model development.

We deliberately chose not to apply bitemporal differencing at coal mine sites. At a continuously emitting mine, any reference scene also contains methane absorption; subtracting it removes part of the signal in an amount that depends unpredictably on the atmospheric state of the reference date. The direction of the effect is not interpretable as evidence for or against methane. Full empirical confirmation — opposite BT outcomes on the two strongest Bełchatów dates using the identical model and reference scene — is in Appendix C.

---

## 9. Conclusion

Calibrated Sentinel-2 methane detection can produce financially relevant facility-level monitoring at large European open-pit coal mines, but current optical pipelines systematically underperform in industrial-fringe underground coal terrain. Both halves of that conclusion are empirically grounded. At Bełchatów, the approach yields a four-year above-threshold response record consistent with an independent emissions inventory, cross-validated by TROPOMI on the highest-signal date, and robust to a leakage audit and training-label contradiction test. At Rybnik — the most externally confirmed methane site in the candidate set — the calibrated rule never fires; the reason is identifiable: a training-distribution gap in Silesian industrial-fringe terrain.

The Rybnik finding is as important as the Bełchatów result: underground hard-coal operations in complex industrial terrain are currently below the detection horizon of this approach, and any institution using optical satellite monitoring for European coal methane risk should treat the underground hard-coal sector as systematically underrepresented rather than zero-emitting. The 44% gap between our Bełchatów estimate and the Climate TRACE inventory is an honest characterisation of what single-overpass optical sensing can see — consistent with the published literature — not a pipeline failure. What the pipeline cannot see is as important to document as what it can.

Three immediate next steps follow. First, expand the model's training coverage for Silesian underground mines — the gap is identifiable and fixable. Second, grow the conformal calibration set from n=35 toward n≥40 with additional Atlantic and Continental candidates to tighten the bootstrap CI toward a continent-scale deployment threshold. Third, broaden the transition-risk translation from the single-issuer PGE case to a peer portfolio of Central European lignite and hard-coal operators; Section 7 demonstrates the transmission channel and the portfolio extension is the natural follow-up.

An open empirical question beyond these steps is whether multi-instrument fusion — combining Sentinel-2 detections with TROPOMI column enhancements and Sentinel-2C's higher revisit frequency — can meaningfully close the seasonal gap that leaves Q1 and late Q4 structurally unobservable. This remains to be demonstrated at European coal latitudes.

---

> **Reproducibility**
>
> All code, weights, and outputs: **https://github.com/vincentcorn2/methane-emission-tracker**
>
> | Step | Command |
> |---|---|
> | Detection + production rule audit | `python scripts/detection/apply_bitemporal_diff.py --sites belchatow` |
> | Conformal threshold (n=35) | `python scripts/calibration/run_mac_inference.py --phase 3` |
> | Bełchatów annual timeseries (MBSP) | `caffeinate -i python scripts/timeseries/belchatow_annual_timeseries.py` |
> | Monte Carlo Climate VaR | `python scripts/finance/finance_climate_var.py` |
> | Transition-risk stress scenarios | `python scripts/finance/finance_transition_risk.py` |
> | Bootstrap AUROC/AP (real crops only) | `python scripts/validation/bootstrap_auroc_ap.py` |
>
> **Canonical result files** (all §6–7 numbers sourced from):
> `results_analysis/belchatow_annual_timeseries_mbsp.json` · `results_analysis/finance_climate_var.json` · `results_analysis/finance_transition_risk.json`
>
> Dependencies: `pip install torch torchvision numpy scipy rasterio pydantic requests cdsapi` · Requires free Copernicus Data Space account and CDS API key for ERA5.

## Data and Code Availability

All pipeline code, model weights, detection records, and financial outputs are available at **https://github.com/vincentcorn2/methane-emission-tracker**.

> **Data recency note.** The analysis uses publicly available data through early 2026, including Carbon Mapper Tanager overpasses at Rybnik through March 2026 and PGE market data anchored to May 2026. Readers reviewing this paper after that date should check whether updated Carbon Mapper campaign data or revised PGE financial disclosures have been released; the pipeline methodology and detection record are unaffected by these updates. The file map below identifies every result file referenced in the paper. Files marked **[canonical]** are the authoritative inputs for all numbers reported in §6 and §7; files marked [archive] are retained for audit and crop-comparison purposes only.

### Bełchatów detection and quantification records

All files are under `results_analysis/` in the repository unless otherwise noted.

| File | Coverage | Physics | Crop | Role |
|---|---|---|---|---|
| `belchatow_annual_timeseries_mbsp.json` | 2019–2025 | MBSP (Varon 2021, scene-derived *c*) | OSM mine polygon (51.242°N, 19.275°E) | **[canonical] Primary result — all §6 numbers** |
| `belchatow_annual_timeseries.json` | 2019–2025 | Old heuristic (*c* = 0.5) | OSM mine polygon | [archive] Pre-MBSP run at correct site |
| `timeseries/belchatow/04_belchatow_mine_polygon_allacq_2019-2025_snapshot.json` | 2019–2025 | Old heuristic | OSM mine polygon | [archive] Snapshot before MBSP re-run |
| `timeseries/belchatow/03_belchatow_mine_coords_singleacq_2021-2024.json` | 2021–2024 | Old heuristic | 750 px square, correct coords | [archive] Intermediate crop |
| `timeseries/belchatow/03_belchatow_mine_coords_singleacq_2021-2024_mbsp.json` | 2021–2024 | MBSP | 750 px square, correct coords | [archive] MBSP requant of intermediate |
| `timeseries/belchatow/01_belchatow_powerstation_coords_750px_crop_2019-2024.json` | 2019–2024 | Old heuristic | 750 px square, **wrong site** (51.266°N, 19.315°E) | [archive] Old wrong-site run |
| `timeseries/belchatow/01_belchatow_powerstation_coords_750px_crop_2019-2024_mbsp.json` | 2019–2024 | MBSP | 750 px square, wrong site | [archive] MBSP requant of wrong site |
| `timeseries/belchatow/02_belchatow_powerstation_5km_crop_2024.json` | 2024 | Old heuristic | 5 km square, wrong site | [archive] −95% under MBSP (spurious surface signal) |
| `timeseries/belchatow/02_belchatow_powerstation_5km_crop_2024_mbsp.json` | 2024 | MBSP | 5 km square, wrong site | [archive] Collapses to 320 kg/hr under MBSP |

**Note on the wrong-site archive.** The `01_` and `02_` files used power-station coordinates (51.266°N, 19.315°E) rather than the mine centroid. They are retained as they demonstrate the internal validation described in §6: under MBSP physics the 5 km power-station crop collapses by 95%, confirming that old estimates over that boundary were driven by surface heterogeneity, not methane signal.

### Rybnik detection records

| File | Coverage | Role |
|---|---|---|
| `rybnik_chwalowice_annual_timeseries_mbsp.json` | 2019–2025 | **[canonical]** Current-site run, MBSP physics |
| `rybnik_chwalowice_annual_timeseries.json` | 2019–2025 | [archive] Current-site run, old heuristic |
| `timeseries/rybnik/02_rybnik_chwalowice_cm_pin_allacq_2019-2025_snapshot.json` | 2019–2025 | [archive] Snapshot, CM pin polygon |
| `timeseries/rybnik/01_rybnik_wrong_site_pge_heat_centroid_2023.json` | 2023 | [archive] Old wrong-site run — 0 detections |

### Financial outputs

| File | Contents | Inputs used |
|---|---|---|
| `finance_climate_var.json` | **[canonical]** Monte Carlo CVaR, 10,000 simulations, all §7.1–7.2 tables | MBSP emission params (16,486 t/yr, n=26) |
| `finance_transition_risk.json` | **[canonical]** Carbon-cost exposure table, PGE issuer stress grid, §7.2–7.4 | MBSP emission params (16,486 t/yr) |

### Quantification and retrieval scripts

| Script | Function |
|---|---|
| `src/quantification/cemf.py` | MBSP retrieval: scene-derived *c*, Varon Eq. 3, dXCH4 integration |
| `src/quantification/ime.py` | IME inversion: Q = mass × U_eff / L, uncertainty bounds |
| `scripts/quantification/requant_mbsp_upgrade.py` | Re-applies MBSP to all five historical crop variants for comparison |
| `scripts/timeseries/belchatow_annual_timeseries.py` | Production Bełchatów pipeline (mine polygon, MBSP) |
| `scripts/timeseries/rybnik_chwalowice_annual_timeseries.py` | Production Rybnik pipeline (CM pin polygon, MBSP) |
| `scripts/finance/finance_climate_var.py` | Monte Carlo CVaR engine |
| `scripts/finance/finance_transition_risk.py` | Transition-risk scenario module |
| `scripts/analysis/annualise_belchatow.py` | QC-filtered annualisation with three framings |
| `scripts/analysis/requant_mbsp_upgrade.py` | Crop comparison re-quantification |

### External data sources

Sentinel-2 imagery: Copernicus Data Space Ecosystem (https://dataspace.copernicus.eu). ERA5 reanalysis winds: Copernicus Climate Data Store (https://cds.climate.copernicus.eu). Climate TRACE facility inventory: https://climatetrace.org (asset 16168 for Bełchatów mine; note: do not use the adjacent power station asset, which reports near-zero methane). Carbon Mapper overpass data: https://carbonmapper.org.

---

## References

- Elderson, F. (2026). Good practices for advancing climate and nature-related risk management. *The Supervision Blog*, ECB Banking Supervision, 8 May 2026. *(URL placeholder — resolve live ECB permalink before submission; cite as ECB Banking Supervision blog post or replace with forthcoming official compendium reference if published.)*
- Desnos, A., Le Guenedal, T., Morais, G., & Roncalli, T. (2024). From climate stress testing to climate value-at-risk: A stochastic approach. *Amundi Institute Working Paper*. *(No persistent DOI at time of writing — verify current availability via Amundi Institute or SSRN before submission.)*
- Ehret, T. et al. (2022). Global tracking and quantification of oil and gas methane leaks from multispectral satellite data. *Environmental Science & Technology*, 56(14), 10226–10235. https://doi.org/10.1021/acs.est.1c07201
- Worden, J. et al. (2025). Common practices for quantifying methane emissions from plumes detected by remote sensing. *NIST Interagency Report 8575*. National Institute of Standards and Technology, Gaithersburg, MD. https://doi.org/10.6028/NIST.IR.8575
- Angelopoulos, A.N. & Bates, S. (2021). A gentle introduction to conformal prediction and distribution-free uncertainty quantification. *arXiv:2107.07511 [cs.LG]*. https://doi.org/10.48550/arXiv.2107.07511
- Karacan, C.Ö., Ruiz, F.A., Cotè, M., & Phipps, S. (2011). Coal mine methane: A review of capture and utilization practices with benefits to mining safety and to greenhouse gas reduction. *International Journal of Coal Geology*, 86(2–3), 121–156. https://doi.org/10.1016/j.coal.2011.02.009
- PGE Polska Grupa Energetyczna S.A. (2025). Consolidated Financial Statements of the PGE Capital Group for the Year 2024 (in accordance with EU IFRS). Warsaw: PGE S.A.
- Reinders, H.J., Schoenmaker, D., & van Dijk, M. (2025). Climate risk stress testing: A critical survey and classification. *Journal of Climate Finance*, 10, 100061. https://doi.org/10.1016/j.jclimf.2025.100061
- Sherwin, E.D., El Abbadi, S.H., Burdeau, P.M., Zhang, Z., Chen, Z., Rutherford, J.S., Chen, Y., and Brandt, A.R. (2024). Single-blind test of nine methane-sensing satellite systems from three continents. *Atmospheric Measurement Techniques*, 17, 765–782. https://doi.org/10.5194/amt-17-765-2024
- Varon, D.J. et al. (2021). Quantifying time-averaged methane emissions from individual coal mine vents with GHGSat-D satellite observations. *Atmospheric Measurement Techniques*, 14, 2771–2785. https://doi.org/10.5194/amt-14-2771-2021
- Vaughan, A., Mateo-García, G., Gómez-Chova, L., Růžička, V., Guanter, L., and Irakulis-Loitxate, I. (2024). CH4Net: a deep learning model for monitoring methane super-emitters with Sentinel-2 imagery. *Atmospheric Measurement Techniques*, 17, 2583–2593. https://doi.org/10.5194/amt-17-2583-2024

---

The following appendices are contained in the companion technical document (technical_appendix.md). Section references below map directly to that document.

| Appendix | Contents | technical_appendix.md section |
|---|---|---|
| **A** | Model architecture, parameter counts, U-Net layer structure, training dataset composition, retraining experiments v1–v11, held-out test outcomes, synthetic plume validation | §1 (§1.1–§1.5) |
| **B** | Conformal calibration full results: per-site sc_cfar scores, Mondrian per-ecoregion thresholds, bootstrap CI, comparison to legacy 1.15 threshold | §2.1 |
| **C** | Bitemporal differencing: mechanism, production decision logic, empirical experiment at Bełchatów (2020 vs 2021, opposite outcomes with identical reference) | §2.2, §5.1 |
| **D** | Full per-site backfill record: all eight candidate sites, date-by-date detection outcomes, partial-swath repair log | §3 (§3.1–§3.5) |
| **E** | External validation detail: Climate TRACE monthly comparison, TROPOMI co-location methodology and full date table, Carbon Mapper overpass record | §4 (§4.1–§4.3) |
| **F** | Data independence and leakage audit: training/evaluation overlap check, calibration set exclusion radius, threshold independence verification | Final appendix section ("Appendix — Data Independence and Leakage Audit") |
| **G** | Temporal sampling analysis: quarterly breakdown, structural ceiling calculation, instrument complementarity options | §5.5 |
| **H** | Transition-risk scenario module: Monte Carlo engine, five uncertainty layers, GWP100/GWP20 VaR tables, reproducibility parameters | §6.2 (technical appendix) |
| **I** | Leave-one-out scene stability: methodology, per-scene influence table (27 scenes × 19 months), stability verdict, honest disclosure of August 2022 outlier | §2.4 |
