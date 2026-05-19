# Methane from Space: Detecting and Quantifying Fugitive Emissions at European Coal Mines Using Satellite Imagery and Deep Learning

**AI Applications in Finance (IEOR 4737)**
European Central Bank & European Investment Bank

*Academic Advisors: Professor Ali Hirsa, Miao Wang*
*Industry Advisors: Dr. Oleg Reichmann (ECB), Dr. Giuseppe Bonavolontà (EIB)*

---

## Abstract

Methane from European coal mines represents an unquantified liability for institutions financing the sector's transition: operators self-report through activity-based estimates, and no independent, calibrated monitoring system with a verifiable error rate currently exists for this asset class. We close part of that gap. Adapting CH4Net to European terrain through eleven retraining experiments, we derive a conformal detection threshold with a finite-sample false-positive guarantee of ≤10% and apply CFAR background testing and CEMF+IME inversion to eight candidate facilities. Three findings follow. At KWB Bełchatów, we identify 30 above-threshold detections spanning 2020–2024, yielding a detection-weighted annualised estimate of 11,481 t CH₄/yr (95% CI: 6,563–16,400 t/yr), approximately 39% of the Climate TRACE inventory and consistent with published Sentinel-2 recovery ranges of 30–60%. At Rybnik, externally validated by both TROPOMI and Carbon Mapper, the calibrated rule never fires despite methane-consistent model responses. The failure is attributable to training-set under-representation of Silesian industrial-fringe terrain, and implies that current optical monitoring cannot yet confirm underground hard-coal emissions at this class of site. We demonstrate how these measurements translate into illustrative transition-risk exposure under EU ETS-equivalent pricing.

---

**Key Numbers**

| Metric | Value |
|---|---|
| Conformal threshold τ (α = 0.10) | 3.5796 |
| Bootstrap 90% CI on τ | [2.49, 4.34] |
| Calibration sites (n) | 35 |
| Above-threshold responses (Bełchatów, 2020–2024) | 30 (1 TROPOMI cross-validated; 29 calibrated-rule only) |
| Quantification-supporting observations | 37 |
| Detection-weighted annualized estimate | 11,481 t CH₄/yr |
| 95% CI on annual estimate | [6,563, 16,400] t/yr |
| Climate TRACE reported total (2024) | 29,636 t CH₄/yr |
| Recovery ratio | 39% (95% CI: 22%–55%) |
| Estimated gap | 18,155 t/yr |

---

> ## Policy & Finance Takeaways
> *For ECB, EIB, and institutional risk officers — key findings without the methodology*
>
> ---
>
> **What this pipeline is.** An independent, satellite-based, per-overpass emission monitoring system for coal-mine methane, built on freely available Sentinel-2 imagery and a deep learning model calibrated against a verified European non-emitter reference set. It produces a facility-level time series that can be updated with every cloud-free satellite pass — approximately 15–25 times per year at European latitudes — without any cooperation from the emitting entity.
>
> **What it found.** At KWB Bełchatów — Europe's largest coal mine, operated by PGE — the analysis identified 30 above-threshold detections spanning 2020–2024, all in months when Climate TRACE independently reports the mine emitting at the 1,700–3,000 tonne level. The detection-weighted annualized emission estimate is **11,481 t CH₄/yr** (95% CI: 6,563–16,400 t/yr), approximately 39% of the Climate TRACE inventory figure of 29,636 t/yr, consistent with published Sentinel-2 recovery ranges of 30–60% (see §6.4 for interpretation). One detection date — September 9, 2021 — carries independent confirmation from the TROPOMI atmospheric sensor on the same day.
>
> **What it cannot find — and why this matters more.** The Rybnik underground hard-coal complex in Silesia, the most externally validated site in the candidate set (five TROPOMI enhancements ranging 10–20 ppb, six Carbon Mapper overpasses with four quantified emissions at 1,150–2,019 kg CH₄/hr), never triggers the calibrated detection rule. The model responds to Rybnik spectrally — it is not blind — but the industrial-fringe terrain exceeds the current training distribution's coverage. **A monitoring implication follows: portfolio-level methane risk assessments relying solely on optical satellite detection would undercount underground hard-coal operations if this gap is not addressed. The underground sector's methane exposure should be treated as currently unverifiable by this approach, not as zero.**
>
> **Financial exposure — single issuer.** Under EU ETS-equivalent pricing and the EU MRV metric (GWP100, factor 28), the Bełchatów detection record implies an annual carbon-cost exposure of approximately **€22.5 M** at €70/tCO₂e (95% CI: €12.9–€32.1 M). Under GWP20 (factor 83, IPCC AR6) — more appropriate for near-term transition risk over 5–10 year investment horizons — the comparable figure is **€66.8 M** (CI: €38.1–€95.3 M). Credit-spread and equity stress scenarios for a hypothetical €10 M PGE position are developed in Section 7.
>
> **Scenario: regulatory disclosure and coordinated reassessment.** If satellite-derived emission monitoring were deployed systematically and linked to mandatory disclosure frameworks, it could contribute to a coordinated reassessment of carbon-linked asset values across coal-sector portfolios — one mechanism the CRST literature identifies as a prerequisite for information-driven repricing (Reinders, Schoenmaker & van Dijk, 2025). This pipeline demonstrates the sensing capability that would underpin such a scenario at a single facility. Whether markets would respond to improved emission transparency through material repricing is conditional on regulatory adoption, disclosure coordination, and investor response — none of which this study models or predicts.
>
> **Three immediate next steps.** *(i)* Expand the model's training coverage for Silesian underground mines — a fixable gap, not a fundamental detection limit. *(ii)* Continue expanding the conformal calibration set — now at n=35 with all five ecoregions represented and small-sample warnings cleared — toward n≥40 with additional Continental and Boreal candidates to tighten the bootstrap CI further toward a continent-scale deployment threshold. *(iii)* Apply the financial scenario module across the peer portfolio of Central European coal operators rather than a single issuer.

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

## 1. Introduction

**The problem.** Coal mine methane is not directly measured. European operators report through activity-based estimates, national inventories aggregate rather than attribute, and lenders or supervisors have no independent view of what a given facility is actually emitting. The gap between reported and actual emissions is not merely a scientific uncertainty: for institutions financing or regulating fossil fuel assets, it is an unpriced liability. Undetected methane can imply unpriced carbon liabilities, misaligned ESG assessments, and potential regulatory exposure as the EU Methane Regulation (Regulation 2024/1787) and related MRV requirements come into force.

**What existing methods miss.** Satellite-based methane detection has advanced rapidly, but two problems remain unsolved at European coal mining sites. First, capable deep learning models were trained on arid oil-and-gas terrain — visually simple landscapes where plume contrast is large. European coal mines sit in temperate agricultural and industrial landscapes with high spectral variability; applying existing models directly produces false-positive rates of approximately 48% on verified non-emitter sites. Second, no published pipeline has derived a statistically calibrated detection threshold for European coal terrain. A threshold with a defensible, finite-sample false-positive guarantee — the kind a financial analyst or regulator could actually rely on — does not yet exist for this problem.

**What we built.** We adapt CH4Net, a deep learning model for Sentinel-2 methane detection, to European terrain through eleven documented retraining experiments, and derive a conformal-calibrated detection threshold with a quantified false-positive rate. The central finding is this: *calibrated Sentinel-2 methane detection can produce financially relevant facility-level monitoring at large European open-pit coal mines, but current optical pipelines systematically underperform in industrial-fringe underground coal terrain.* The first half of that claim is demonstrated at the KWB Bełchatów lignite mine in Poland; the second half — equally important — is demonstrated at the Rybnik underground hard-coal complex in Silesia.

**Main findings.** Bełchatów — included in training as a negative example — produces a multi-year above-threshold detection record consistent with an independent emissions inventory, providing evidence of generalizable spectral learning rather than site memorization. Rybnik produces no calibrated detections despite stronger external validation than any other candidate site; the cause is an identifiable training-distribution gap documented in Section 5. Full quantitative results are in Section 6.

**Contributions.** This paper makes six concrete contributions. First, a European domain adaptation of CH4Net with documented training-set shifts, eleven retraining experiments, and held-out validation. Second, a conformal-calibrated detection threshold replacing the heuristic 1.15 threshold — a methodologically sharper foundation than is typical in applied remote-sensing work. Third, operational handling for four real-world remote-sensing failure modes rarely addressed together: partial-swath data artifacts, denominator-collapse in S/C ratios, terrain-heterogeneity-driven CFAR adaptation, and cloud-driven temporal sparsity. Fourth, a facility-level quantification pipeline integrating Sentinel-2 output, ERA5 winds, and the IME/CEMF framework across 37 observations. Fifth, an empirical characterization of where the current European detection frontier lies, including a systematic failure mode with direct implications for how institutions should interpret optical satellite monitoring of underground hard-coal operations. Sixth, direct evidence against training-set memorization: the model produces 30 above-threshold responses at Bełchatów despite this site being labeled as a non-emitter in training, with zero date-overlap between training crops and evaluation acquisitions confirmed by formal temporal leakage audit — a stronger generalization argument than a held-out test provides.

---

*[**Figure 1 — Methodology flow diagram** to be inserted here: Sentinel-2 tile download → partial-swath check → CH4Net inference → S/C ratio → CFAR background test → conformal threshold τ=3.5796 → detection decision → ERA5 wind retrieval → CEMF/IME inversion → emission rate estimate.]*

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

A critical feature of CH4Net is that it runs on a single image. It does not compare today's image to a previous reference image. This distinguishes it from the family of methods — known as multi-band multi-pass, or MBMP — that subtract a clean reference scene from the current scene to isolate what has changed. For a coal mine, which emits methane year-round without "clean days," a reference-subtraction approach is particularly problematic: the reference scene itself contains methane absorption, so the subtraction partially cancels the very signal you are trying to detect. We examined this issue carefully and confirmed it empirically on our own data (see Section 6 and Appendix C for detail). CH4Net's single-image design avoids this problem entirely, at the cost of working harder to separate methane from surface features using spectral shape alone.

### 3.2 Why the Model Needed Adaptation for Europe

CH4Net was originally trained on 23 super-emitter sites between 2017 and 2020, primarily in the Turkmenistan oil and gas region. The landscapes there are arid desert with sparse infrastructure — a visually simple environment where plume contrast against background is large. European coal mine sites are fundamentally different: Bełchatów sits in central Poland surrounded by agricultural fields, mixed forest, rivers, and road infrastructure whose surface spectral variability far exceeds the arid training distribution. The original model, applied directly, frequently mistook European surface features for methane.

We addressed this by fine-tuning the model on a European-specific training dataset. This involved collecting 14 confirmed positive training crops — image patches from European sites where TROPOMI atmospheric measurements independently confirmed methane was present on that date — along with 51 synthetic positive crops generated by digitally injecting simulated plumes into real European imagery, and 22 verified-negative crops from sites with no known methane emissions. We ran eleven retraining experiments and evaluated each against a consistent set of control outcomes. The eighth version — which we call **v8**, or interchangeably "our fine-tuned model" or "our production model" — was the configuration that correctly handled all of the failure modes we had identified without suppressing real signals at other sites. Throughout this paper, "CH4Net" refers to the base architecture; "v8" refers specifically to our European fine-tuned version. Full architecture specification, training dataset composition, and per-experiment retraining outcomes are in Appendix A (technical_appendix.md §1).

One of those failure modes deserves specific mention. Our pipeline initially produced persistent false positives over the polder landscape near Groningen in the Netherlands — the distinctive pattern of drainage canals and reclaimed farmland produced SWIR reflectance patterns that the model consistently misread as plume-shaped. We added ten Groningen scenes to the negative training set specifically to teach the model to ignore that landscape. This worked, but it had a side effect we did not fully anticipate: the model became somewhat overfit to suppressing polder-like industrial fringe terrain, which reduced its sensitivity at the Silesian underground mining sites we discuss in Section 5.

---

## 4. Building a Statistically Reliable Detection Rule

The detection pipeline reduces to a single question: does the facility stand out above its local background strongly enough to survive statistical calibration? The metrics that follow — S/C ratio, CFAR, conformal threshold τ — are operational expressions of that one question.

Detecting a plume is not simply a matter of asking the model whether the probability is high. High probabilities can occur over interesting surfaces on non-emission days, and a site visited 50 times per year will accumulate false detections if the threshold is set too loosely. Before we could use any of the model's outputs for financial or policy analysis, we needed a detection rule with a calibrated, defensible false-positive rate.

### 4.1 Signal-to-Control Ratio

Our detection metric is the signal-to-control ratio, or S/C ratio. For each satellite overpass, we extract the mean CH4Net probability within a 100-by-100 pixel crop centered on the facility we are monitoring. We then extract the same metric from four control crops of identical size, each offset approximately 22 kilometers in a cardinal direction from the site. The S/C ratio is simply the site crop's mean probability divided by the average of the four control crops. A ratio of 1.0 means the model is no more confident at the facility than in the surrounding landscape; a ratio well above 1 means the facility stands out.

The advantage of this ratio approach is that it adapts to local conditions. On a day when the model is generally noisy — perhaps because of atmospheric scattering or an unusual surface state — both the site mean and the control means will be elevated, and the ratio remains near 1. The signal-to-control ratio therefore reflects facility-specific elevation above the local background, not absolute model output.

*In plain terms: we ask whether the model is more confident about methane at the facility than in the surrounding countryside on the same day — not whether the model's output is high in absolute terms.*

### 4.2 The Problem with Simple Thresholds

The natural instinct is to declare a detection when S/C exceeds some fixed threshold — Vaughan et al. used 1.15 as a starting point. We applied this rule to 35 verified non-emitter sites across Europe, sampled from European Environment Agency land-cover classifications with a minimum 50 kilometer separation from any known methane source. The result was striking: the 1.15 threshold fired on 16 of those 35 non-emitter sites, a false-positive rate of 46%. Applied at that rate, roughly half of the "detections" our pipeline would report at any new site would be spurious. This is not a defensible threshold for financial analysis.

*In plain terms: a smoke detector set this loosely would be going off in half the rooms with no fire. You cannot base a risk assessment on it.*

### 4.3 Conformal Calibration

*In plain terms: instead of guessing a threshold by inspection, we set it empirically from known-clean sites and get a mathematical guarantee on the maximum false-alarm rate — the same principle as calibrating a medical test on a control group before using it diagnostically.*

We replaced the heuristic threshold with one derived from conformal prediction, a statistical framework developed by Angelopoulos and Bates (2021) that provides finite-sample guarantees on error rates rather than asymptotic approximations. The procedure is straightforward: given a set of verified non-emitter observations, it computes the minimum detection threshold such that the probability of a false positive is guaranteed to be at most some chosen level, regardless of sample size.

We chose a false-positive rate target of 10% — meaning that when our pipeline reports a detection at a new site, there is at most a 1-in-10 chance that site is actually a non-emitter. The resulting threshold, denoted τ, is 3.5796. Applied to the full 35-site calibration set, this threshold fires on only 2 of 35, an empirical false-positive rate of 5.7%. We also computed a bootstrap confidence interval by resampling the calibration set 2,000 times; the 90% interval on τ is [2.49, 4.34], confirming that the threshold is not pinned to a single outlier observation.

The practical implication is that detection calls made by our pipeline carry a statistical guarantee that was absent from the original model. A financial analyst using these detections to flag an emitter can interpret each positive result as a one-in-ten chance of a false alarm — not an unknown and unquantifiable chance.

This should be read as a well-grounded pilot calibration rather than a continent-scale production threshold. The guarantee is mathematically valid at n=35, and the bootstrap CI on τ ([2.49, 4.34]) has tightened meaningfully compared to earlier calibration rounds — the lower bound has risen from 2.12 to 2.49 as the set expanded. All five ecoregion strata now have n≥6 with no small-sample warnings, and even at n=35 the conformal threshold already outperforms the heuristic 1.15 threshold by a large margin. The Pannonian and Mediterranean strata, where the heuristic fired at rates above 30%, now carry meaningful indicative thresholds. The path to a production-grade continental threshold is clear: expand to 40 or more calibration sites with fuller Atlantic and Continental coverage.

### 4.4 Scene-Level Detection Stability

A concern common in machine-learning–assisted monitoring studies is that reported performance metrics are dominated by a small number of anomalously strong observations. We tested this directly by running a leave-one-out (LOO) analysis on every above-threshold scene in the Bełchatów record: for each of the 27 cfar_detect=True scenes spanning 2021–2024, we removed it from the detection set and recomputed the detection rate and mean S/C ratio on the remaining observations.

The detection rate is stable across all 27 removals. Excluding any single scene shifts the scene-level detection rate from 27/102 to 26/101, a change of 0.73 percentage points — 2.8% of the baseline rate — identical regardless of which scene is removed. The month-level rate (counting a month as detected if any scene over it fired) shifts by 1.31 percentage points per exclusion, again constant. These results are a structural consequence of the binary detection framework: no single scene can exert outsized influence on a proportion when N=102 and positives number 27. The LOO confirms this mathematically rather than assuming it.

One observation does carry disproportionate influence on the mean S/C ratio: August 2022 produced sc_cfar=515, 9× the next-highest value and 140× the median (57.4). Removing it shifts the mean from 97.6 to 81.5 — a 16% change — while leaving the detection rate and detection count unchanged. We disclose this directly: the August 2022 acquisition is the single strongest response in the record, consistent with a high-loading summer operating day, and we do not trim or down-weight it. Its influence is concentrated in the mean of a right-skewed distribution, not in the binary classification outcome that underlies all financial and regulatory conclusions in the paper. Full per-scene LOO results are reported in Appendix I (technical_appendix.md §2.4).

### 4.5 Leakage Audit

Because certain evaluation sites — including Bełchatów and Neurath — appeared in the v8 training set as negative examples, we conducted a formal temporal leakage audit before interpreting any results. The concern is whether near-duplicate scenes (same site, close acquisition date) in both training and evaluation could cause the model to appear to generalize when it is actually memorizing. The audit checked every evaluation date against every training-crop date at the same site within a 14-day window. The result: zero pairs fell within that window across all eight candidate sites. The 27-site conformal calibration set was verified to have zero spatial overlap with any candidate site under a 50 kilometer exclusion radius. The threshold τ was computed from calibration-set scores alone, with no exposure to candidate-site outcomes at any stage. The full audit log is in Appendix F (technical_appendix.md Appendix).

The implication is important for interpreting Section 6: when the model produces above-threshold responses at Bełchatów despite a negative training label, that is not a leakage artifact. The training crop used the August 2024 background tile; the evaluation acquisitions are spread across 2021–2024 on different dates, at a different atmospheric state. The model is encountering genuinely novel scenes and responding to their spectral content, not pattern-matching to a memorized image.

*In plain terms: we checked that the model isn't passing the exam because it saw the answer sheet. It hasn't — it generalises.*

---

## 5. Site Selection: Eight Candidates, One Primary

We evaluated eight European facilities as candidates for the primary case study. The set was chosen to span major coal types (lignite and hard coal), include gas infrastructure, and include two sites we knew in advance to be control cases — either clean (expected non-emitters) or false-positive traps (sites where the original CH4Net model had misfired before our retraining). For each site, we downloaded all available cloud-free Sentinel-2 acquisitions between 2019 and 2024 and ran the full pipeline.

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

### 5.1 What the Comparison Showed

After applying our calibrated detection rule to the candidate backfill, three sites produced at least one above-threshold response. Bełchatów produced three methane-consistent signals, spread across three distinct calendar years: June 2020, June 2021, and July 2024. This temporal spread matters. A single high-confidence response at a site could be a transient surface anomaly — a flooded pit section, a flash of unusual reflectance from exposed coal. Three above-threshold responses across three different years, each independently exceeding our calibrated threshold, are much harder to attribute to coincidental surface effects.

Neurath produced two above-threshold responses, both in 2024, and is notable because one coincides with an independent atmospheric enhancement measured by the TROPOMI sensor on the same day — making it the only site in our candidate set with a same-day dual-instrument confirmation during the candidate evaluation window. Lippendorf produced a single very high signal reading in September 2024, but showed no signal at all across the preceding four years of acquisitions, which makes the reading look more like an isolated surface event than a persistent emission pattern. Weisweiler produced one above-threshold response in 2021 but suffered five partial-swath data gaps in its record (see the next section). The three remaining sites — Boxberg, Groningen, and Maasvlakte — produced zero above-threshold responses under our calibrated rule, which is the correct outcome: Boxberg is the clean control, and Groningen and Maasvlakte are the false-positive controls that our retraining was designed to suppress.

We selected Bełchatów as the primary case study on the strength of its multi-year temporal persistence, the availability of independent monthly inventory data from Climate TRACE for cross-validation (Climate TRACE rates its confidence for this facility as "low" — its estimates are activity-based rather than measurement-based, which is precisely why independent satellite verification is valuable), and its status as Europe's largest lignite mine — making it both the highest-priority emission source in the candidate set and the one with the most publicly available contextual data. Critically, this selection was made on the basis of the candidate backfill results alone — before any intensive monitoring, quantification, or annual emission estimation was carried out. The selection criteria (multi-year persistence, Climate TRACE coverage, facility scale) were defined in advance of running the full pipeline, and no annual quantification results existed at the time of selection. The intensive monitoring record in Section 6 was produced after and independently of the site selection decision.

### 5.2 A Note on Data Quality: Partial-Swath Tiles

During the backfill analysis, we identified a data quality issue that could easily have gone unnoticed. Some Sentinel-2 products listed in the catalog as covering a given location do not actually contain imagery at that location — the tile catalog record exists, but the satellite's imaging swath did not extend to the relevant coordinates on that pass. When the pipeline ran inference on these mostly-zero arrays, it produced an S/C ratio of exactly 1.0, which would appear to be a clean non-detection. In reality the model had no usable data. We built a fingerprint check into the pipeline to detect this condition and reclassify affected records as missing observations rather than non-detections (technical details in Appendix D, technical_appendix.md §2.3). The initial backfill caught seven affected records across three sites. The Bełchatów intensive monitoring period, discussed in the next section, produced 47 such records out of 111 total acquisitions — a non-trivial fraction that would have meaningfully inflated our apparent non-detection rate had it gone uncorrected.

### 5.3 Rybnik: The Most Externally Validated Site the Model Cannot Detect

The most instructive finding from the candidate comparison is what happened at Rybnik. The Rybnik mining area in Silesia, which encompasses several underground hard-coal mines operated by JSW and PGG, is by far the most externally validated site in our candidate set. TROPOMI has recorded five independent methane enhancements above this area between February 2023 and June 2024, ranging from 10.85 to 19.66 parts per billion above local background. Carbon Mapper — an airborne and satellite program operated by Planet Labs and NASA-JPL that uses a different physical measurement principle entirely — has made six overpasses between August 2023 and March 2026 and measured quantified emission rates on four of them, ranging from 1,150 to 2,019 kilograms of methane per hour. Two independent instruments, one atmospheric and one imaging, from three independent organizations (ESA, Planet Labs, NASA-JPL), are all pointing at the same location and agreeing that it is emitting large amounts of methane persistently.

The model does respond to Rybnik. On March 22, 2025 — one day after Carbon Mapper measured 2,019 kg/hr at these coordinates — CH4Net produced a raw S/C ratio of 5.48, well above the naive 1.15 threshold. The probability-weighted centroid of high-confidence pixels sat 2.56 kilometers southwest of the Carbon Mapper source pin, broadly consistent with ERA5 winds blowing from the east at 4.8 m/s — a physically plausible plume displacement, not a random artifact.

The distinction between "model responds" and "calibrated rule confirms" is the central point of the Rybnik result. The table below summarises the two most diagnostic acquisition dates:

| Date | External context | Raw S/C | cv_ctrl | Naive threshold (>1.15) | Calibrated outcome |
|---|---|---|---|---|---|
| 2023-06-01 | — | 309 | 1.62 | Would fire | ✗ Suppressed — high background inflates CFAR threshold to ~6.0 |
| 2025-03-22 | Carbon Mapper 2,019 kg/hr (Mar 21) | 5.48 | 0.978 | Would fire | ✗ Suppressed — CFAR score 0.42, well below τ |

On the highest-signal date (June 2023), the raw S/C of 309 clears the naive threshold by two orders of magnitude — and is still suppressed. On the Carbon Mapper-adjacent date (March 2025), the raw ratio clears the naive threshold by nearly 5×, and is still suppressed. In both cases the mechanism is the same: background heterogeneity in Silesian industrial terrain raises the calibrated bar above what the signal achieves.

**The 172° wind reversal test.** The centroid displacement is not just plausible — it passes a designed falsification. The logic has three parts:

*Hypothesis to falsify:* the model is responding to a static surface feature (railway yard, industrial rooftop, SWIR-bright spoil heap). If true, the centroid of high-confidence pixels must remain fixed regardless of atmospheric conditions on any given overpass.

*Prediction under the null:* Carbon Mapper's previous overpass (March 21, 2025) reported wind from 215.7°, which would have driven a methane plume centroid toward bearing 35.7° northeast. A static terrain artifact would produce that same centroid bearing on March 22 — because static features do not move when wind direction changes.

*Observation:* the CH4Net centroid on March 22 sits at bearing 172° from that prediction — almost exactly opposite, and broadly aligned with the day-of ERA5 wind from the east. The centroid did not stay fixed; it flipped by 172° between consecutive days, tracking the change in wind direction.

A fixed terrain artifact cannot produce a 172° centroid shift between consecutive days. The directional evidence is inconsistent with the static-feature hypothesis and consistent with atmospheric transport of a real emission. The calibrated rule cannot confirm a detection — the Silesian background is too heterogeneous — but the spatial behavior of the model's output is behaving as a plume, not as terrain. The highest raw S/C across the full Rybnik backfill was 309 in June 2023 (full geometry analysis in technical_appendix.md §5.3). The signal is consistent with a physical plume response; the calibrated rule cannot confirm it at the current threshold.

The same CFAR mechanism that suppresses false positives at Groningen applies here: background heterogeneity in Silesian industrial terrain is high enough that the adaptive threshold rises above what the signal achieves, which is the correct statistical behavior.

The Rybnik non-detection reflects a training coverage gap, not a fundamental limit of the optical sensing approach. We had one underground mine industrial scene in our positive training set and ten Groningen polder scenes in our negative training set. The fine-tuning overfit to suppressing polder-like complexity at the cost of sensitivity to a different kind of complex industrial landscape. Expanding the training set with confirmed Silesian scenes is the most important single improvement available to the pipeline. We have documented this limitation explicitly because it matters for any institution considering how much of European coal mine methane is currently detectable with this approach — the answer, at least for underground hard-coal operations in industrial-fringe terrain, is that current tools are probably undercounting.

---

**Context for Section 6.** The Rybnik result documents where the pipeline currently fails. Section 6 documents where it succeeds: at a site within the model's training distribution, the pipeline produces a four-year quantification record with calibrated uncertainty, inventory-consistent recovery, and same-day TROPOMI confirmation. The two findings are distinct and both reported without preference.

---

## 6. Results at Bełchatów

A note on epistemic register before the results: **detection** and **quantification** operate in fundamentally different uncertainty regimes in this pipeline, and the paper's claims should be read accordingly. Detection — whether the model's calibrated rule fires — is relatively strong: the conformal threshold provides a calibrated false-positive bound, every above-threshold response date aligns with Climate TRACE's independent inventory, and the model produces above-threshold responses at this site despite a negative training label. Annual quantification is substantially weaker: it is conditioned on the highly non-representative sample of cloud-free observable overpasses, carries structural seasonal bias (zero winter observations in any year), and propagates significant wind and spatial-extent uncertainty through the CEMF inversion. The 11,481 t/yr figure and the 39% recovery ratio should be read as detection-conditioned estimates under a continuous-emission assumption, not as direct measurements of annual throughput.

### 6.1 The Facility

KWB Bełchatów is an open-pit lignite mine in central Poland, operated by PGE Polska Grupa Energetyczna. It is the largest coal mine in Europe by excavated volume and one of the largest single-point sources of carbon dioxide on the continent, primarily through the adjacent Bełchatów power station which burns the extracted coal. For purposes of this analysis, the mine and the power station are distinct Climate TRACE assets. The mine — Climate TRACE asset 16168 at coordinates 51.242°N, 19.275°E — is classified under IPCC sector 1B1a (fugitive emissions from coal mining) and is methane-dominant. The power station is classified under sector 1A1a (electricity generation) and is carbon dioxide-dominant, with near-zero reported methane. Any comparison of our results against Climate TRACE must use the mine entry, not the power station; a lookup that returns the power station will show no methane and appear to contradict our detections, which would be misleading.

### 6.2 The Detection Record

We ingested 111 Sentinel-2 acquisitions over Bełchatów spanning 2021 through 2024, averaging slightly over three overpasses per month. Of these, 47 were reclassified as missing observations due to the partial-swath data quality issue described in Section 5.2. The remaining 64 acquisitions entered the detection analysis as valid observations. Of those, 37 produced sufficient model output to support a CEMF+IME quantification estimate at a consistent 7.5 km crop window. The 30 above-threshold responses in the combined record are composed of two groups: 27 detections from the 2021–2024 intensive monitoring period that satisfied both the calibrated S/C threshold (sc_cfar > τ) and the CFAR gate, plus 3 additional detections from the 2020–2024 candidate backfill evaluation (including the June 2020 detection at S/C = 849). Eight further records fell below the conformal threshold but carried enough model signal for a valid IME inversion. The September 9, 2021 acquisition — the highest-S/C date in the record — carries same-day TROPOMI cross-validation (§6.5); the remaining 29 above-threshold dates are calibrated-rule-confirmed only.

Every detection date from 2021 onward falls inside a month where Climate TRACE independently reports the mine emitting methane at the 1,700 to 3,000 tonne level. Non-detection dates cluster in winter and shoulder-season months — December through February and November — when cloud cover, reduced solar elevation, and occasional snow cover reduce Sentinel-2's effective sensitivity in the shortwave-infrared bands. These are not months when the mine stops emitting; they are months when the satellite cannot see clearly enough to detect it.

The record of above-threshold responses also provides evidence against a label-reproduction artifact. Bełchatów was included in the v8 training data as a negative example: we labeled it as a non-methane site on the background tile from August 2024. The model nonetheless produces 30 above-threshold responses — the pipeline is responding to spectral signatures it was trained to ignore. This is consistent with the model having learned a generalizable methane signature rather than memorizing site identities, and it is a stronger indicator than a held-out test would provide, because the model is disagreeing with its own training label on the basis of the spectral evidence.

### 6.3 Quantifying Emission Rates

To convert the spatial probability maps into physical emission rate estimates, we apply the Column-Enhancement Mass Flux (CEMF) method, following Varon et al. (2021). This approach integrates the mass of methane enhancement implied by the CH4Net probability map over a defined area, then divides by the estimated atmospheric transport time using wind speed data from the ERA5 reanalysis product (produced by the European Centre for Medium-Range Weather Forecasts, and available through the Copernicus Climate Data Store). The wind provides the key link between a snapshot of a plume at one moment and an estimate of how fast methane is flowing out of the source.

We applied this method to a consistent 7.5 kilometer crop window centered on the mine coordinates across all 37 quantification-supporting observations. ERA5 wind data was retrieved at the satellite overpass time for each date; all 37 records used ERA5 reanalysis winds rather than climatological fallback values, which would have required an additional uncertainty penalty.

Estimated emission rates across the 37 records ranged from 82 to 8,218 kilograms of methane per hour, with a median of 618 kg/hr and a mean of 1,311 kg/hr. The wide range is expected and physically meaningful: methane emissions from an open-pit mine are not constant. They vary with excavation activity, wind conditions that affect how the plume disperses, and atmospheric stability that affects how much methane the satellite can sense above the background. High emission days tend to produce higher S/C ratios, which is why the distribution of detection-date flow rates skews upward relative to the overall emission average — the model detects more easily on high-emission days, introducing a selection bias that our 30–60% recovery expectation accounts for. The dominant uncertainty sources on individual per-overpass estimates, in approximate order of contribution, are: wind retrieval (~±20%, propagated from ERA5 accuracy at coarse resolution), sensitivity coefficient (~±15%, traceable to the Varon 2021 calibration), mask threshold sensitivity (~±5–7% at Bełchatów, based on formal jackknife decomposition showing near-zero background contribution), and temporal sampling bias — the largest structural uncertainty on the annualised figure, arising because detected days may not be representative of the continuous emission distribution across all cloud-free opportunities.

The 95% confidence interval on the mean rate, based on a t-distribution with 36 degrees of freedom, runs from 749 to 1,872 kg/hr. Under a continuous-emission assumption and conditional on the cloud-free observable overpasses that produced this sample, the detection-weighted mean flow rate corresponds to an annualized estimate of 11,481 tonnes of methane per year. The 95% confidence interval on this annualized projection is 6,563 to 16,400 tonnes per year. This figure reflects what was detectable and measurable under the sampling conditions described in Section 8; it is not a direct measurement of annual throughput.

### 6.4 Comparison to the Independent Inventory

Climate TRACE reports 29,636 tonnes of methane from asset 16168 for the full year 2024, making our annualized mean estimate approximately 39% of the inventory figure. The 95% confidence interval spans 22% to 55% of the Climate TRACE value.

This recovery fraction is not a discrepancy — it is the expected behavior of satellite-based single-overpass detection applied to a continuously emitting source. The published literature on Sentinel-2-based methane quantification at coal mines documents recovery ratios in the 30 to 60% range (Varon et al. 2021; Sherwin et al. 2024). Our estimate sits just below the lower bound of that range, which reflects two structural factors specific to our pipeline. First, we detect preferentially on favorable atmospheric days — low cloud cover, adequate solar illumination, stable atmospheric conditions — and these are not representative of the full year. On days the satellite cannot see clearly or the model cannot detect, we have no flow rate estimate, and those gaps tend toward periods of lower modeled sensitivity rather than lower actual emissions. Second, the model's probability maps over-predict spatial extent relative to the true plume boundary, which makes the area integration in the CEMF calculation sensitive to the crop window size. We used a consistent 7.5 kilometer crop for all 37 records to avoid introducing size-dependent bias across the time series, but the absolute level of the estimate varies with this choice: a narrower window (5 km) would capture less of the over-predicted plume area and likely reduce the mean estimate, while a wider window (10 km) risks incorporating background pixels that inflate it. The direction of the dominant bias is toward overestimation of spatial extent, which would push our recovery ratio downward relative to the true value — meaning 39% should be interpreted as a lower bound on the fraction of total emissions we are detecting, not an upper bound. A useful analogy: measuring a river's flow rate on 37 cloud-free days per year and averaging does not "see only 37/365 of the river" — it measures the flow on those specific days and extrapolates. The 39% ratio reflects how that day-weighted average compares to an inventory covering all 365 days; it is not evidence that the pipeline misses 61% of plumes on the days it does observe.

Climate TRACE itself rates its 2024 confidence for this asset as "low" on its published confidence scale, reflecting the well-documented difficulty of quantifying fugitive coal mine emissions from activity data alone. Neither estimate is a ground-truth measurement. Both are independent estimates built from different methodologies, and their order-of-magnitude agreement is the validation we can responsibly claim.

*[**Table — Validation summary** to be inserted here: rows = Bełchatów, Rybnik, Neurath; columns = CH4Net confirmed detections, TROPOMI co-detections, Carbon Mapper overpasses, Climate TRACE alignment, conformal threshold status. Allows reader to grasp cross-instrument agreement at a glance.]*

### 6.5 Independent Atmospheric Confirmation

We searched for coincident measurements by the TROPOMI sensor aboard Sentinel-5P on every Bełchatów observation date where our model produced an above-threshold response. TROPOMI measures the total column of methane in the atmosphere at approximately 5.5 by 3.5 kilometer pixel resolution; an enhancement above the facility coordinates on the same day that our model fires would provide independent confirmation that elevated methane concentrations were present in the atmosphere.

On September 9, 2021, TROPOMI recorded a +12.7 parts per billion enhancement at the Bełchatów coordinates against a background estimated from a 0.25 to 1.0 degree annulus around the facility. This is the same date on which our pipeline recorded its highest S/C ratio across the entire intensive monitoring record. Having both instruments respond on the same day — and respond most strongly on the same day — is the result we would expect if both instruments are responding to methane rather than surface artifacts — evidence against a purely artifact-driven explanation.

The remaining 32 above-threshold CH4Net dates did not yield usable TROPOMI retrievals. On some dates the Sentinel-5P orbital swath did not cover the tile; on others, the quality filtering (designed to exclude cloud-contaminated columns) removed too many pixels for a valid enhancement estimate. TROPOMI's revisit frequency and spatial coverage at this latitude make coincident retrievals the exception rather than the rule, which is why the Climate TRACE inventory comparison provides the primary cross-validation for the broader record and why the single September 9, 2021 agreement is informative rather than conclusive on its own.

---

**Key Findings in Plain Language** *(for non-technical readers)*

- We can reliably detect methane at large open-pit coal mines using freely available satellite imagery with statistical calibration that bounds the false-alarm rate.
- Underground coal systems in complex European terrain are currently under-detected by this method — a documented limitation, not a claim of absence.
- Where detection succeeds, the resulting emission estimates align with independent inventories at the order-of-magnitude level expected from single-overpass optical sensing.

---

## 7. Illustrative Transition Risk Translation (Scenario-Based)

ECB Banking Supervision has noted explicitly that climate and nature-related financial risks "may be underestimated within the financial system" due to "non-linear dynamics and compounding events" that remain poorly understood (Elderson, 2026). Unverified facility-level methane represents one concrete instance of that underestimation: it is a liability that exists in physical reality but is absent from operators' disclosed figures and therefore from the credit models built on those figures. The scenarios below illustrate the scale of that gap for a single asset under policy assumptions.

**This section is not a market prediction model.** It maps the methane emission estimates from Section 6 into carbon-price exposure figures under explicit policy assumptions, and presents three illustrative stress scenarios for a hypothetical single-name position. All results are conditional on (i) regulatory adoption of a methane pricing framework comparable to the EU ETS, and (ii) the detection pipeline's central estimate being representative of the facility's true emission rate — both of which are stated assumptions, not conclusions. No claim is made about actual market pricing, credit spread movements, or equity valuation changes at PGE. The purpose is to demonstrate the transmission channel from satellite-based measurement to financial exposure, not to predict how or when that channel would activate.

For ECB and EIB specifically, the primary contribution of this section is not the single-name figures: it is the demonstration that calibrated satellite methane data can be translated into the language of transition-risk stress testing under a defined and auditable set of assumptions. Three properties of the pipeline make it usable in an institutional context: (i) the conformal false-positive guarantee of ≤10% means its outputs can be cited in a compliance or supervisory document rather than treated as directional; (ii) the Rybnik finding explicitly maps where the methodology fails, giving supervisors a documented boundary on what the monitoring covers; and (iii) the pipeline uses only freely available data, imposing no access asymmetry between institution and operator.

> **Stylized stress analysis disclaimer.** Position sizes are hypothetical, shocks are illustrative tiers from the transition-risk literature, and carbon-cost figures are exposure proxies, not booked liabilities. All numbers are reproducible from `scripts/finance_transition_risk.py`.

The issuer is **PGE Polska Grupa Energetyczna S.A.** (WSE:PGE), the state-controlled Polish utility (State Treasury is the only shareholder holding ≥5% of votes) that owns and operates KWB Bełchatów through its GiEK subsidiary. PGE carries investment-grade ratings (Moody's Baa1, Fitch BBB, both stable as of 2025) and a 2026 market capitalisation of approximately USD 6.34 billion. Key 2024 financials from the consolidated annual report: net loss EUR 717M (second consecutive year of losses; EUR 1,083M in 2023); gross debt EUR ~3.1B; net debt EUR ~2.1B after EUR 1.0B cash. PGE already holds **EUR ~2.54B in CO₂ emission allowances** on its current balance sheet (PLN 10,913M per note 15 of the 2024 financial statements) — making it already one of Europe's largest single-entity holders of EUA exposure. The Conventional Generation segment, which contains Bełchatów, posted a 2024 EBIT of −EUR 1.7B after approximately EUR 2.1B in depreciation and asset write-downs that reduced the segment's PPE from EUR 4.9B to EUR 3.0B in a single year. The coal fleet is already in structural decline on the balance sheet; the methane liability is additional unpriced carbon risk stacked on top of an asset base whose book value is contracting at pace. PGE's own 2024 climate risk disclosures (prepared under CSRD/ESRS) explicitly identify rising CO₂ allowance costs as a strategic financial threat — the precise transmission channel our satellite pipeline quantifies from the outside.

### 7.1 Stochastic Uncertainty Propagation — Methodology

We implement a Monte Carlo uncertainty propagation engine (10,000 simulations) to translate the satellite-derived methane emission record into a full probability distribution of carbon liabilities. The framework follows the stochastic climate risk approach of Desnos, Le Guenedal, Morais and Roncalli (Amundi, 2024) and incorporates the IME plume quantification uncertainty budget recommended in Worden et al. (NIST IR 8575, 2025).

In each simulation we draw: (i) an annual CH₄ emission rate from a zero-truncated normal consistent with the 95% sampling CI on the 37 quantification-supporting observations; (ii) a systematic ERA5 wind bias ~N(0, 0.10), justified by NIST IR 8575 §4.2 benchmarking of reanalysis grid-to-point interpolation error against local meteorological tower observations; (iii) a plume spatial-extent factor ~Uniform(0.85, 1.15), following Varon et al. (2021, AMT §2.3) sensitivity tests on the IME integration boundary; (iv) a carbon price from LogNormal centred at €70/tCO₂e with 30% log-volatility, consistent with historical EU ETS dynamics; and (v) a regulatory enforcement probability from Beta(9,1) (mean 90%), reflecting a conservative compliance stress scenario under the EU Methane Regulation. The truncated normal in step (i) — rather than a log-normal — preserves the exact accounting identity E[Liability] = E[Q] × GWP × E[Price] × E[β], making the model directly auditable against the deterministic base case. The stochastic mean lies approximately 10% below the deterministic figure, reflecting residual enforcement uncertainty under the Beta(9,1) pass-through distribution; this gap is documented, not a defect.

We report the mean expected liability, 95% and 99% Climate Value-at-Risk, and 99% Expected Shortfall — separately under GWP100 (EU MRV regulatory metric) and GWP20 (near-term transition risk horizon). All parameters are declared as module-level constants in `scripts/finance_climate_var.py`; results are serialised to `results_analysis/finance_climate_var.json`.

### 7.2 Implied Carbon-Cost Exposure

Coal-mine methane is not currently covered under the EU ETS (Phase IV), but the Methane Regulation sets the trajectory. The figures below represent order-of-magnitude exposure if a comparable price regime applied. Converting the pipeline's central estimate (11,481 t CH₄/yr) via IPCC AR5 GWP100 (factor 28) gives 321,468 t CO₂e/yr. Sensitivity across EUA price scenarios:

| Price case | €/tCO₂e | Mean estimate | Lower CI bound | Upper CI bound |
|---|---:|---:|---:|---:|
| Low | 50 | €16.07 M | €9.19 M | €22.96 M |
| Central | 70 | €22.50 M | €12.86 M | €32.14 M |
| Upper | 95 | €30.54 M | €17.46 M | €43.62 M |

To place these figures in context: PGE's existing CO₂ allowance book is EUR ~2.54B (end-2024). The methane satellite estimate at the central EUA price adds approximately 0.9–2.6% to that existing carbon exposure depending on GWP metric — small as a fraction of the allowance book, but entirely unpriced and unaudited without independent monitoring. Under the central case the implied annual exposure is approximately €22.5 M at GWP100. The EU MRV framework uses GWP100 as its regulatory metric, making it the appropriate basis for compliance exposure estimates. GWP20 (factor 83, IPCC AR6) is the more relevant horizon for near-term transition risk: it reflects the actual climate forcing over the 5–10 year investment horizons within which most coal asset repricing is expected to occur.

**GWP Sensitivity — Carbon-Cost Exposure (Central EUA price: €70/tCO₂e)**

| | GWP100 (factor 28, EU MRV) | GWP20 (factor 83, IPCC AR6) |
|---|---:|---:|
| Annual CO₂e equivalent | 321,468 t | 953,923 t |
| Mean estimate | €22.50 M | €66.77 M |
| 95% CI lower bound | €12.86 M | €38.13 M |
| 95% CI upper bound | €32.14 M | €95.28 M |

The ~3× difference between GWP100 and GWP20 is not a modeling choice — it reflects a genuine scientific distinction about which time horizon the exposure is being priced over. GWP100 is the regulatory compliance metric: it governs EU MRV reporting and ETS-equivalent pricing frameworks. GWP20 is the investor-relevant metric for transition risk: stranded-asset repricing, if it occurs, unfolds over 5–10 year horizons driven by near-term physical forcing and policy action — not over a century. Supervisory exercises focused on near-term financial stability should present both.

**Monte Carlo Climate Value-at-Risk (10,000 simulations)**

The stochastic engine described in §7.1 propagates all five uncertainty layers jointly. The mean expected liability (€20.3M GWP100 / €60.3M GWP20) lies approximately 10% below the deterministic base case, reflecting the Beta(9,1) enforcement pass-through (mean 90%) — this gap is documented and intentional, not a model defect. The uncertainty decomposition reveals that carbon price trajectory (log-vol 30%, contributing 77% of total σ) dominates the tail; improving the satellite emission precision would have less impact on the liability distribution than reducing uncertainty about whether and when a comparable price regime applies.

| Risk Metric | GWP100 (M€) | GWP20 (M€) |
|---|---:|---:|
| Mean expected liability | 20.34 | 60.29 |
| Median | 18.80 | 55.74 |
| 95th percentile — Climate VaR 95 | 36.36 | 107.78 |
| 99th percentile — Climate VaR 99 | 48.11 | 142.60 |
| **99% Expected Shortfall (ES)** | **54.47** | **161.45** |

*Parameters: n_sim = 10,000; seed = 42; carbon price central €70/tCO₂e, log-vol 30%; pass-through Beta(9,1) mean 90%; ERA5 systematic σ = 10%; mask jitter Uniform(0.85,1.15). Full results in `results_analysis/finance_climate_var.json`.*

### 7.3 Credit-Spread Stress

*The €10 million position size used throughout §7.2–7.4 is chosen for arithmetic convenience only. All P&L figures scale linearly — divide or multiply for any actual exposure size.*

A hypothetical €10 million senior unsecured PGE position at 5-year modified duration produces a CR01 of approximately €5,000/bp. Three illustrative spread tiers — Mild (+15 bp), Moderate (+35 bp), Severe (+50 bp) — correspond qualitatively to an ESG-rating downgrade, a Methane-Regulation enforcement action, and a sustained inventory-vs-satellite gap entering public discussion, yielding mark-to-market bondholder losses of €75K, €175K, and €250K respectively.

### 7.4 Equity-Repricing Scenario

A hypothetical €10 million long-equity position under the same three tiers (Mild −3%, Moderate −7%, Severe −10%) produces P&L outcomes of −€300K, −€700K, and −€1.00 M.

### 7.5 Combined Sensitivity Grid

| Tier | ΔEquity | ΔSpread | Equity P&L (€10M long) | Bond P&L (€10M IG) | Combined P&L |
|---|---:|---:|---:|---:|---:|
| Mild | −3% | +15 bp | −€300 K | −€75 K | **−€375 K** |
| Moderate | −7% | +35 bp | −€700 K | −€175 K | **−€875 K** |
| Severe | −10% | +50 bp | −€1.00 M | −€250 K | **−€1.25 M** |

The combined-channel P&L (−€375K to −€1.25M) is modest relative to PGE's capitalisation but meaningful at portfolio-position scale — the relevant frame for an EIB credit officer or ECB supervisory analyst. The three channels are not necessarily independent: in a scenario where the inventory-vs-satellite gap becomes policy-salient, ESG-sensitive funding costs and equity valuations could move together rather than sequentially. Treating them additively may understate drawdown risk under that specific scenario, though the degree of correlation is not empirically estimated here.

**A note on coverage.** The Rybnik finding (Section 5.3) implies that any portfolio-level exposure estimate built solely from optical-satellite signal will undercount the underground coal sector. For PGE the bias is limited — its principal coal asset is open-pit lignite at Bełchatów — but for a continent-wide application to Polish, Czech, or German hard-coal operators, the satellite-detection horizon should be paired with conservative inventory-based estimates for the underground subset rather than treated as a complete view. More broadly, systematic deployment across European coal operators would create the data-availability conditions that Reinders, Schoenmaker, and van Dijk (2025) identify as a prerequisite for coordinated asset reassessment under regulatory disclosure — a scenario in which emission signals not previously incorporated into asset prices become observable under a defined policy framework. Whether and how markets would respond is conditional on regulatory adoption and investor reaction; this study does not model either.

---

*[**Figure — What works / what fails / why** to be inserted here: a three-panel or three-row summary. Panel 1 — Bełchatów: confirmed detections, S/C time series, TROPOMI co-validation on Sept 9 2021. Panel 2 — Rybnik: model S/C 5.48, CFAR score 0.42/4.11, centroid-wind geometry diagram showing the 172° falsification. Panel 3 — Structural ceiling: quarterly detection counts vs. cloud climatology bar chart showing Q1/Q4 gap. This figure is the five-minute read the paper currently lacks.]*

### 7.6 Policy Relevance and Operational Use Cases

The following three use cases describe how the pipeline's outputs could be operationalised by ECB, EIB, or institutional risk officers without requiring modification to the methodology. Each is distinct in scope and requires no additional modelling beyond what is demonstrated in this paper.

- **MRV validation support under EU Regulation 2024/1787.** The EU Methane Regulation requires active coal mines to submit continuous methane monitoring data and member states to establish independent verification mechanisms. The pipeline described here is directly usable as a source-independent verification layer: it produces facility-level emission estimates from freely available satellite data, with a calibrated false-positive rate of ≤10% that can be cited in a compliance audit context. Where an operator's self-reported figure diverges materially from the satellite-derived estimate, the pipeline identifies the discrepancy for follow-up — not as a definitive measurement, but as a flagging mechanism within an MRV framework. Notably, ECB Banking Supervision's 2026 good practices compendium explicitly identifies Copernicus programme data — the same satellite infrastructure underlying our Sentinel-2 pipeline — as an endorsed public tool for quantifying physical risks at individual asset level (Elderson, 2026, footnote 4).

- **Portfolio screening signal for high-emission coal assets.** A lender or supervisory authority maintaining exposure to Central European coal operators can use the detection record as a qualitative screening input: sites producing multi-year, multi-instrument-confirmed above-threshold responses represent a different category of transition-risk exposure than sites with no observable signal. The pipeline does not replace financial due diligence, but it provides an independent, updateable data source that is not derived from operator disclosure — closing a gap that credit ratings and ESG scores currently leave open. ECB Banking Supervision's 2026 good practices compendium identifies facility-level transition risk assessment — "assessing transition risks at the individual client level" rather than relying on sectoral averages — as an emerging supervisory good practice (Elderson, 2026). This pipeline operationalises that approach using satellite data rather than client-reported figures.

- **Prioritisation tool for targeted high-resolution sensing.** Carbon Mapper's Tanager programme and GHGSat's commercial constellation both accept tasking requests for specific facilities, but tasking capacity is limited and requires documented evidence of likely methane activity. The pipeline's Sentinel-2 outputs serve directly as that prior: the September 9, 2021 TROPOMI confirmation at Bełchatów, combined with the CH4Net detection record, constitutes the signal quality that Carbon Mapper's public-data programme requires before scheduling a quantification overpass. For any institution seeking ground-truth confirmation at a coal asset, this pipeline provides the triage layer that routes facilities toward targeted high-resolution sensing rather than requiring blanket coverage.

---

## 8. Limitations

We have tried throughout this paper to be specific about where the pipeline falls short, because vague acknowledgments of limitation are less useful to a reader making decisions than precise ones.

The most important limitation is the Rybnik finding. Our model cannot currently confirm detections at the most externally validated coal mine methane source in our candidate set. The reason is identifiable and fixable — the training set underrepresents Silesian industrial-fringe terrain — but the implication is that the current pipeline is probably undercounting European coal mine methane, and the undercount is not random. It is concentrated in the underground hard-coal sector, which in Poland and other Central European countries is the dominant form of coal mining. Bełchatów, where we do have above-threshold responses with Climate TRACE inventory alignment, is a large open-pit lignite mine — a geographically distinctive and relatively large target. Smaller or underground operations may be systematically harder to detect with this pipeline in its current form.

The second limitation is that n=35 remains a moderate calibration set, and we want to be direct about what that means rather than paper over it. The conformal guarantee holds at any finite n — that is its mathematical appeal — and the bootstrap CI on τ has tightened to [2.49, 4.34] (see Key Numbers table) as the set expanded from 25 to 35. All five ecoregion strata now have n≥6 with no small-sample warnings; Boreal and Pannonian went from n=3 to n=6, meaningfully reducing single-observation sensitivity. The Pannonian and Mediterranean Mondrian thresholds, previously near-zero and unreliable, are now 3.5796 and 1.4088 respectively. A continent-scale deployment would require 40 or more calibration sites with fuller Atlantic and Continental coverage — still the recommended next step, now closer than it was. None of this invalidates the pilot results; it defines the scope within which they should be applied.

Third, the temporal coverage of our detection record is near the structural ceiling for Sentinel-2 in central Poland — and that ceiling is lower than one might assume. Sentinel-2's nominal revisit is five days, implying roughly 73 acquisition opportunities per year. But SWIR methane detection requires both clear skies and adequate solar illumination. Central Poland sits at 51°N with 55–70% annual cloud cover, concentrated in the November–March period. This reduces effective cloud-free acquisition opportunities to approximately 15–25 per year. Our intensive monitoring over 2021–2024 achieved 15 valid observations per year — consistent with that ceiling, not a sparse sample of a larger accessible set.

The seasonal consequence is concrete: we have zero above-threshold responses from November through March in any year in the record. Q1 (January–March) and late Q4 (November–December) are structurally near-invisible to this approach. Climate TRACE's monthly inventory shows the mine emitting continuously through winter at comparable rates to summer. Whether the mine's actual emission rate varies seasonally cannot be fully assessed from our data, but the coal mine methane literature offers a tentative directional inference for open-pit operations. Methane desorption from coal is temperature-dependent and inversely correlated with barometric pressure: cold, high-pressure winter conditions are generally associated with lower fugitive emission rates than warmer summer months (Karacan et al. 2011; IPCC 2006 Energy Guidelines §4.2). If this pattern holds at Bełchatów — a reasonable prior for an open-pit mine at 51°N, though published seasonal flux measurements for this specific facility are not available to our knowledge — our detection-weighted mean, drawn predominantly from April–October overpasses, would modestly overestimate the true annual continuous average. This directional effect partially offsets the spatial-extent over-prediction bias discussed in Section 6.4, but the net magnitude remains unquantified. For underground mines, where active ventilation continuously removes methane regardless of temperature and season, this directional inference is weaker and the seasonal pattern less predictable.

Partial fixes exist: Sentinel-2C (launched 2024) reduces revisit to roughly three days when operating alongside Sentinel-2A and 2B, adding perhaps 20–30% more opportunities per year. Better cloud masking can recover some borderline passes. But the core limitation — that cloud-covered winter months in temperate Europe are structurally opaque to SWIR optical sensors — is not solvable within the Sentinel-2 paradigm. SAR-based approaches (cloud-independent radar instruments) exist but would require a different detection model not developed here. Commercial hyperspectral instruments (GHGSat, Carbon Mapper's Tanager) can be tasked specifically on cloud-optimal days but do not provide the time series density or free data access that Sentinel-2 does. This limitation is not unique to our pipeline; it applies to any Sentinel-2-based methane monitoring approach in temperate climates and should be understood by any institution using such estimates as a basis for financial analysis.

Finally, one methodological choice deserves transparency. We deliberately chose not to apply bitemporal differencing — a technique that subtracts a reference scene from the current scene to isolate what has changed — at coal mine sites. This technique is useful at gas field sites where the landscape is stable and methane events are episodic. At a continuously emitting coal mine, however, any reference scene also contains methane absorption, and subtracting it removes some of the signal we are trying to measure. We confirmed this empirically: applying the reference subtraction to our two strongest Bełchatów detection dates produced opposite outcomes — the 2020 signal collapsed entirely, while the 2021 signal amplified by a factor of 12,696 — depending purely on which reference scene was used and what the atmospheric state happened to be on the reference date. This confirmed that reference subtraction is unreliable at continuous emitters, and we disabled it for all seven coal and lignite sites in our candidate set. We cite the Varon et al. (2021) MBMP framework because our CEMF quantification uses its mathematical formulation, but we depart from it on this specific preprocessing decision.

---

## 9. Conclusion

Calibrated Sentinel-2 methane detection can produce financially relevant facility-level monitoring at large European open-pit coal mines, but current optical pipelines systematically underperform in industrial-fringe underground coal terrain. Both halves of that conclusion are empirically grounded. At Bełchatów, the approach yields a four-year above-threshold response record consistent with an independent emissions inventory, cross-validated by TROPOMI on the highest-signal date, and robust to a leakage audit and training-label contradiction test. At Rybnik — the most externally confirmed methane site in the candidate set — the calibrated rule never fires; the reason is identifiable: a training-distribution gap in Silesian industrial-fringe terrain.

The Rybnik finding is as important as the Bełchatów result. It establishes that underground hard-coal operations in complex industrial terrain are currently below the detection horizon of this approach, and any institution using optical satellite monitoring to assess European coal methane risk should treat the underground hard-coal sector as systematically underrepresented rather than zero-emitting. The 61% gap between our Bełchatów estimate and the Climate TRACE inventory (18,155 t/yr in absolute terms) is not a pipeline failure — it is an honest characterization of what single-overpass optical sensing can see, consistent with the published literature. What the pipeline cannot see is equally important to document.

The immediate next steps are expanding the model's training coverage for Silesian underground mines, growing the conformal calibration set from n=35 toward n≥40 with additional Atlantic and Continental candidates to further tighten the bootstrap CI, and broadening the illustrative transition-risk translation (Section 7) from the single-issuer PGE case to a peer-portfolio basis covering other Central European lignite and hard-coal operators. Section 7 demonstrates how the quantification time series maps to carbon-price exposure and stylized stress scenarios under explicit regulatory assumptions; the portfolio extension is the natural follow-up.

---

## References

- Elderson, F. (2026). Good practices for advancing climate and nature-related risk management. *The Supervision Blog*, ECB Banking Supervision, 8 May 2026. https://www.bankingsupervision.europa.eu/press/blog/2026/html/ssm.blog260508~example.en.html
- Desnos, A., Le Guenedal, T., Morais, G., & Roncalli, T. (2024). From climate stress testing to climate value-at-risk: A stochastic approach. *Amundi Institute Working Paper*.
- Worden, J. et al. (2025). Common practices for quantifying methane emissions from plumes detected by remote sensing. *NIST Interagency Report 8575*. National Institute of Standards and Technology, Gaithersburg, MD. https://doi.org/10.6028/NIST.IR.8575
- Angelopoulos, A.N. & Bates, S. (2021). A gentle introduction to conformal prediction and distribution-free uncertainty quantification. *arXiv:2107.07511 [cs.LG]*. https://doi.org/10.48550/arXiv.2107.07511
- Karacan, C.Ö., Ruiz, F.A., Cotè, M., & Phipps, S. (2011). Coal mine methane: A review of capture and utilization practices with benefits to mining safety and to greenhouse gas reduction. *International Journal of Coal Geology*, 86(2–3), 121–156.
- PGE Polska Grupa Energetyczna S.A. (2025). Consolidated Financial Statements of the PGE Capital Group for the Year 2024 (in accordance with EU IFRS). Warsaw: PGE S.A.
- Reinders, H.J., Schoenmaker, D., & van Dijk, M. (2025). Climate risk stress testing: A critical survey and classification. *Journal of Climate Finance*, 10, 100061. https://doi.org/10.1016/j.jclimf.2025.100061
- Sherwin, E.D., El Abbadi, S.H., Burdeau, P.M., Zhang, Z., Chen, Z., Rutherford, J.S., Chen, Y., and Brandt, A.R. (2024). Single-blind test of nine methane-sensing satellite systems from three continents. *Atmospheric Measurement Techniques*, 17, 765–782. https://doi.org/10.5194/amt-17-765-2024
- Varon, D.J. et al. (2021). Quantifying time-averaged methane emissions from individual coal mine vents with GHGSat-D satellite observations. *Atmospheric Measurement Techniques*, 14, 2771–2785.
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
| **F** | Data independence and leakage audit: training/evaluation overlap check, calibration set exclusion radius, threshold independence verification | Appendix (technical_appendix.md) |
| **G** | Temporal sampling analysis: quarterly breakdown, structural ceiling calculation, instrument complementarity options | §5.5 |
| **H** | Transition-risk scenario module: GWP100 conversion, EUA price grid, illustrative stress tiers, hypothetical position assumptions, reproducibility command | §6 |
| **I** | Leave-one-out scene stability: methodology, per-scene influence table (27 scenes × 19 months), stability verdict, honest disclosure of August 2022 outlier | §2.4 |
