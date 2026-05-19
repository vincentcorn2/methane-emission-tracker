# Meeting 6 — Background Notes for Vincent

*A personal study guide. Everything you need to speak to each term fluently. Structured so you can skim on the day.*

---

## 0. The 60-second mental model

Sentinel-2 is a European satellite that takes a photo of every place on Earth roughly every 5 days. We run a deep-learning model called **CH4Net** on those photos to pick out methane plumes — the model outputs, for every pixel, "the probability there's methane here." That gives a probability map.

Two jobs follow:

1. **Detection.** Is there actually a plume, or is the model reacting to terrain / clouds / noise? We compare the probability at the plant vs. at a nearby clean area ("control"), get a ratio, and fire a detection if the ratio is high enough. This is where *threshold calibration* happens.
2. **Quantification.** Given a detection, how big is the leak in kg/h? This is where *CEMF + IME + ERA5 wind* come in.

Everything else on the agenda is either fixing data problems that corrupt step 1, or building the statistical machinery that makes the numbers defensible to a regulator.

---

## 1. Satellite plumbing

**Sentinel-2 / S2.** European Space Agency optical satellite pair. Revisits each location every ~5 days. Produces multi-band images (visible + near-infrared + shortwave infrared).

**S2 overpass date.** The specific date/time the satellite passed overhead and took a usable image of a location. Not every overpass is usable — clouds, sensor saturation, or the plant falling outside the actual image swath all invalidate a date. When we say "five valid S2 overpass dates with ground truth between 5 and 1,496 kg/h," we mean five dates where (a) Sentinel-2 was looking, (b) the Sherwin team had released a known amount of methane, and (c) the image is usable.

**Why this matters for us.** The 5-day revisit is the floor on how fresh our signal can be. For financial risk, a quarterly update is enough; for real-time operations monitoring it wouldn't be.

---

## 2. Detection — turning probability maps into yes/no

### S/C ratio (signal over control)

For every acquisition, we take the average CH4Net probability in a ring of pixels around the plant (**signal**) and in a ring around a nearby clean area (**control**), then divide. If S/C ≈ 1, nothing unusual at the plant. If S/C = 849 (our Bełchatów 2020-06-01 result), the signal is 849× louder than the local background — that's a huge plume.

### CFAR — the old detection rule

**CFAR = Constant False Alarm Rate.** A classical rule from radar engineering. The idea: set a threshold that scales with the local noise level, so the fraction of false alarms stays constant across different conditions. Our old rule was "call it a detection if S/C > 1.15 + 3σ_control" — i.e., the signal has to be both higher than the control and higher than the typical variation in the control by ≥3 standard deviations.

**Why we replaced it.** CFAR assumes the noise is roughly Gaussian. Satellite imagery isn't — terrain variability, cloud fragments, and seasonality all introduce non-Gaussian noise. The CFAR false-positive rate was an approximation under assumptions that don't hold. On our 14-site calibration set, the legacy 1.15 threshold fires on **11 of 14 known-clean sites** (78.6% false-positive rate in practice, despite the theoretical ~0.3% under Gaussian assumptions). So it was producing many spurious "detections."

### Split conformal threshold — the new rule

**Conformal prediction** is a modern distribution-free statistical framework. It gives you a probabilistic guarantee — "FPR ≤ 10%" — that holds **at any sample size** without assuming the noise follows any particular distribution.

The mechanics are simple:

1. Collect a set of **known non-emitter sites** (places that you're confident are clean).
2. Run them through the pipeline and record each site's S/C value.
3. To guarantee FPR ≤ α, use the (1 − α)-quantile of that empirical distribution as your threshold.

"Split" just means we used a **held-out** sample (different from training data) for calibration — which is what makes the guarantee valid.

**τ (tau)** is the threshold value we end up with. Our production number is **τ = 4.1052**. Any acquisition with S/C > 4.1052 clears the bar.

**α (alpha)** is the false-positive rate we targeted. α = 0.10 means "at most 10% of genuinely clean sites will falsely fire." We pick α for policy reasons: α = 0.05 would be stricter but currently there's not enough calibration data to pick a finer quantile (single-observation dominated).

**The sentence to remember:** *"We replaced an engineering heuristic whose false-positive rate was approximately 0.3% under Gaussian assumptions that don't hold — it was firing on 78% of our verified non-emitters in practice — with a distribution-free conformal threshold that carries a provable ≤ 10% FPR at finite sample size."*

### What "finite-sample false-positive rate ≤ 10%" means in plain English

If we apply this threshold to 100 genuinely-clean sites, at most ~10 will fire. That's a guarantee, not an average — and it holds no matter how weird the underlying noise distribution is.

---

## 3. Quantification — turning detection into kg/h

Once we know a plume exists, we need to estimate how much methane it contains and how fast it's leaking. This is a three-step physical retrieval.

### CEMF — Column-Enhanced Matched Filter

**What it does.** Converts the CH4Net probability mask into an estimate of methane **mass** (in kg) in the atmosphere above each pixel at the moment of the overpass.

**How it works.** Sentinel-2 has two shortwave-infrared bands that are close to each other in wavelength: B11 (~1610 nm) and B12 (~2190 nm). Methane absorbs light at ~2190 nm about **5× more strongly** than at ~1610 nm. That differential absorption is the "signature" — by comparing the reflectance in the two bands pixel by pixel, you can back out how much methane was in the column that sunlight passed through.

The "matched filter" part is the statistical technique: you project the spectral anomaly onto the known methane absorption signature to isolate the CH₄ contribution from other things that could darken the image (soil, water, shadows).

**Key parameter: 4×10⁻⁷ reflectance per ppb·m.** This is the sensitivity coefficient from Varon et al. 2021 — says "for every ppb of methane integrated over 1 meter of atmosphere, the reflectance drops by 4×10⁻⁷." It's a lookup value we're currently using as a fixed scalar and that one of the near-term tasks is to replace with a scene-dependent estimate.

### IME — Integrated Mass Enhancement

**What it does.** Converts the instantaneous plume mass (from CEMF) into a flow rate (kg/h).

**The formula:** Q = (M × U) / L

- **M** = total plume mass in kg (from CEMF)
- **U** = wind speed (m/s)
- **L** = plume length (m) — estimated from the geometry of the detection mask

**Intuition.** If the plume is 5 km long and the wind is 5 m/s, the wind flushes through the plume in 1000 seconds. Whatever mass is in the plume right now is what the source emitted in that 1000 seconds. Multiply out → emission rate per hour.

**Uncertainty note.** Literature value for the total uncertainty on IME retrievals is ±40% (single overpass, multispectral). That's a placeholder we're working to decompose and tighten.

### ERA5 — the wind data

**What it is.** A global atmospheric **reanalysis** product from ECMWF (the European weather forecasting center), distributed via the Copernicus CDS API. It gives hourly wind data at 0.25° resolution globally, back to 1940.

**Why it matters.** The plume flow rate is linear in wind speed. If you use the wrong wind, your Q is off by the same ratio. Before ERA5, we were using a fixed 3.5 m/s climatological fallback — the average European wind speed — which could be off by factor of 2+ on any given day. That was the dominant error source in the Meeting-5 Netherlands numbers. ERA5 pulls the actual wind at the plant location at the exact time the satellite took the image.

**The provenance story.** Every quantification record logs which wind source was used — ERA5 or fallback — so a reviewer can see at a glance whether a specific flow rate is trustable. Records on fallback get a governance flag (`WIND_FALLBACK`) and have their wind uncertainty inflated by +30 percentage points. That's a deliberate pessimistic bias when we know the input is degraded.

### Provenance (general)

For every number we output, we record: which wind source, which mask file, the model version, the CEMF sensitivity coefficient, the acquisition timestamp, the centroid lat/lon, whether it was excluded and why. This is regulatory-grade auditability — if someone asks "where does this 426.5 kg/h come from?" we can reconstruct it end-to-end from the JSON record.

### IRA liability

**IRA = US Inflation Reduction Act.** In 2026 it imposes a **Waste Emissions Charge of $1,500 per tonne of methane** on petroleum and natural gas facilities above a threshold.

**Why we report it.** The EU Methane Regulation penalties haven't been finalized in a numeric form yet, so there's no euro-per-tonne we can use as a regulatory benchmark. The IRA $1,500/tonne is currently the only enacted statutory methane liability rate globally, so we use it as a financial proxy — a concrete dollar number that shows the scale of exposure if a similar scheme were applied to European operators. When the EU Methane Regulation / CBAM methane calibration is finalized, we swap in those numbers.

---

## 4. Data quality — what "degenerate record" and "backfill integrity" mean

### The failure mode

Sentinel-2 tiles the earth into fixed ~100km grid squares. Sometimes an orbit clips a tile at its edge — the tile exists in the catalog, but most of its pixels are zero because the satellite wasn't actually looking there. We call these **partial-swath tiles**.

If the plant location happens to fall in the zero-pixel region of a partial-swath tile, CH4Net still runs on the (mostly blank) data and produces a uniform probability map. Both the signal region and the control region end up reading the **same probability** — which gives **S/C = 1.0 exactly, to many decimal places**. That's a **degenerate record**: it looks like a valid "no plume detected" observation, but actually no observation happened at all.

### Why the audit matters

A reviewer who looks at the raw JSON and notices 7 records where `site_mean == ctrl_mean` down to 6 decimal places will immediately recognize that as a pipeline artifact, not a physical measurement. If they find that in our deliverable, our credibility on the whole time series takes a big hit from one grep command.

The fix has already been applied — 7 records were repaired and a valid-pixel coverage pre-check was added to the pipeline. **Backfill integrity** just means re-auditing the entire 2019–2025 historical series to make sure no remaining degenerate records slipped through before the deliverable leaves the room.

---

## 5. Model validation framework — SR 11-7, SS1/23, and the five workstreams

### The regulatory framing

**SR 11-7** is a US Federal Reserve / OCC supervisory letter from 2011 that defines how banks must manage the risk from models used in decision-making. It established the **three-pillar structure** that's now standard globally:

- **Pillar 1 — Conceptual soundness.** Is the model mathematically and scientifically defensible?
- **Pillar 2 — Process verification / independent validation.** Do the numbers check out? Are the inputs and outputs audited?
- **Pillar 3 — Outcomes analysis / ongoing monitoring.** Does it keep working correctly once deployed?

**SS1/23** is the UK Bank of England's 2023 equivalent — same three-pillar structure, tuned to UK bank governance.

**Why this matters for us.** A methane-detection model being used as an input to credit risk calculations is, formally, a bank model under SR 11-7 / SS1/23. If we build our workstreams to map onto the three-pillar structure from the start, we avoid a painful documentation retrofit later. That's what "aligned to the SR 11-7 / SS1/23 three-pillar structure" means in the agenda.

### The five workstreams

**WS1 — Uncertainty quantification.** Every number output by the pipeline gets a decomposed uncertainty budget: how much comes from wind error, how much from the CEMF retrieval, how much from the plume mask, how much from background noise. Combined via Monte Carlo. This is what replaces the flat ±40% placeholder we inherited from Varon. Pillar 2.

**WS2 — Foundation model + OOD.** CH4Net was originally trained on Central Asian oil and gas imagery. European coal plants look different — different terrain, different climate, different backgrounds. **OOD = out-of-distribution detection** is a family of methods that flags when the input image is "outside" what the model was trained on. We plan to replace CH4Net's backbone with a fine-tuned **foundation model** (Prithvi-EO-2.0 via LoRA) to get better generalization. Pillar 1.

**WS3 — Physics-informed transport.** Build a lightweight physical simulator of how plumes disperse (advection + diffusion, solved with a Fourier Neural Operator and simulation-based inference). Cross-check the CEMF/IME retrievals against this physics model as a conceptual-soundness anchor. Pillar 1.

**WS4 — Multi-sensor fusion.** Combine Sentinel-2 (our main source) with TROPOMI (a different methane satellite with different strengths) and potentially others, so a detection is cross-confirmed by independent instruments. Pillar 3.

**WS5 — Stress testing and benchmarking.** Inject synthetic plumes of known magnitude into real images to measure the pipeline's detection limit; benchmark against the Sherwin et al. controlled-release dataset where ground truth is known. Pillar 3.

WS1 and WS5 are the two we've made progress on this cycle (conformal threshold is WS5; uncertainty decomposition script is WS1). WS2–WS4 are the priority for next cycle.

---

## 6. The EIB/ECB question — what to actually ask

### CSPP / PEPP / ISIN

**CSPP = Corporate Sector Purchase Programme.** Since 2016, the ECB has bought corporate bonds on the open market as part of its monetary policy. Those bonds sit on the Eurosystem balance sheet.

**PEPP = Pandemic Emergency Purchase Programme.** A COVID-era expansion of the same idea.

**ISIN = International Securities Identification Number.** A unique identifier for each specific bond (different ISINs for different maturities, different coupons, etc.).

**What we found.** Of our 8 monitored plants, 6 are repeated emitters. Of those 6, **4 have parent companies whose bonds are held by the ECB** in CSPP/PEPP:

| Plant | Operator | Bonds in ECB portfolio |
|---|---|---|
| Weisweiler, Neurath | RWE | 5 ISINs |
| Lippendorf (50%) | EnBW | 8 ISINs |
| Groningen | Shell (via NAM JV) | 10+ ISINs |
| Maasvlakte | Engie | 20+ ISINs |

The other 2 emitters — **Bełchatów and Rybnik (PGE, Polish state-owned)** and **Boxberg/Lippendorf (LEAG, private EPH-owned)** — are **not in CSPP**. PGE/LEAG bond issuance is generally not CSPP-eligible.

### Merton-KMV, PD, MTM

**Merton-KMV.** A standard structural credit-risk model. It treats a company's equity as a call option on its assets — if asset value falls below debt, the firm defaults. From asset volatility and the debt level, you derive a **probability of default**.

**PD = Probability of Default.** How likely the company is to miss a bond payment within some horizon (usually 1 year). This is the single most important number in credit risk.

**MTM = Mark-to-Market.** The market value of a bond isn't its face value — it shifts as investors' assessment of the issuer's creditworthiness changes. If PD rises, the bond's yield spread widens and its market price falls. MTM impact = how many euros the bond's market value moved.

### Why this matters to ECB

If our emissions trajectory can be translated into a PD shift, and from PD into a spread-widening, and from spread-widening into MTM impact — then our satellite data is literally a dollar number on the Eurosystem balance sheet. That's the climate-to-credit transmission the ECB has been asked to quantify for its Pillar 2 climate stress tests.

### The question in the simplest form

**Do you want the analysis at the company level (one PD per operator, applied across all their bonds), or at the bond level (per ISIN, which captures that longer-maturity bonds are more sensitive to credit-quality changes than shorter ones)?**

Company-level is faster, gives one headline number per operator. Bond-level is more work but gives a euro MTM impact per ISIN that the ECB can roll directly into their balance-sheet stress test.

### Question 2 in the simplest form

**The two remaining emitters (PGE and LEAG) aren't in CSPP. Does EIB have direct loan exposure to either of them?** If yes, our methane signal should feed EIB Group credit monitoring on those loans. If no, we treat them as "non-CSPP reference cases" used in the broader ECB Pillar 2 stress test rather than tied to a specific holding.

---

## 7. Likely pushback and scripted replies

**"Why α = 0.10? Why not α = 0.05?"**
*At n = 14 calibration sites, both α = 0.05 and α = 0.10 give the same τ = 4.1052 — the sample size is too small to distinguish. α = 0.10 is the reported figure because that's the tightest nominal coverage the current sample supports. Expanding the calibration set to n ≥ 30 will let us offer α = 0.05 with a meaningful separation.*

**"Your τ is basically the maximum of your calibration set — isn't that single-observation-dominated?"**
*Yes, exactly — at n = 14 the conformal quantile lands on the maximum by construction. That's a known property, not a bug: the guarantee is still valid at any n, but it's conservative. A single pathological site (nonemit_003, Moselle Valley, with an anomalously low control mean) dominates. Expanding to n ≥ 30 is the stated next step for that reason.*

**"What's the difference between your S/C ratio and a p-value?"**
*S/C is a physical ratio, not a significance measure. It says "the signal is X times stronger than the local control." The p-value interpretation comes after conformal calibration — once we apply τ, the statement becomes "under the null of no emission, the probability of seeing S/C > τ is ≤ α." So S/C is the test statistic; τ calibrates it into a hypothesis test with a known error rate.*

**"Why not use TROPOMI directly since its methane sensitivity is better?"**
*TROPOMI has 7 km resolution vs. Sentinel-2's ~20 m. It can confirm a region is emitting but can't pinpoint which facility. Sentinel-2 gets us to the actual plant and lets us estimate plume geometry for IME. The plan is to use TROPOMI as a cross-validation in WS4, not as the primary source.*

**"Is ±40% uncertainty really acceptable for a credit-risk input?"**
*No — and that's why the WS1 uncertainty decomposition is the near-term priority. The ±40% is an inherited literature placeholder. Replacing it with a per-scene, source-attributed budget (σ_wind, σ_CEMF, σ_mask, σ_background combined via Monte Carlo) is the script we have built and ready to run. Target is tighter bounds wrapped in conformalized quantile regression against Sherwin ground truth.*

**"What does your pipeline miss?"**
*Three main blind spots at present: (1) plumes below ~100 kg/h are at or below our detection limit; (2) scenes under heavy cloud cover are unusable; (3) nighttime emissions are invisible (Sentinel-2 is optical). The WS5 synthetic plume injection is what will quantify (1) rigorously.*

---

## 8. One-line cheat sheet for each term

- **S/C ratio** — signal at the plant ÷ signal at a nearby clean area.
- **CFAR** — old detection rule, heuristic, fires on ~78% of known-clean sites.
- **Split conformal threshold** — new detection rule, distribution-free, provably ≤ 10% false positives.
- **τ (tau)** — the threshold value (= 4.1052 in production).
- **α (alpha)** — the false-positive rate we're targeting (0.10).
- **CEMF** — spectral retrieval that converts pixels to plume mass.
- **IME** — Q = (M × U) / L, converts plume mass to flow rate.
- **ERA5** — European wind reanalysis, plugs into IME.
- **Provenance** — the audit log that lets anyone reconstruct a number from inputs.
- **IRA liability** — US $1,500/tonne methane charge, used as a financial proxy until EU rate is fixed.
- **S2 overpass date** — date of a usable Sentinel-2 image.
- **Degenerate record** — a pipeline output where the satellite didn't actually see the plant (S/C = 1.0 exactly).
- **Backfill integrity** — auditing the historical time series for those degenerate records.
- **OOD** — "out-of-distribution," flags when an input is outside the model's training envelope.
- **SR 11-7 / SS1/23** — US and UK regulatory frameworks for model risk management, three pillars.
- **CSPP / PEPP** — ECB corporate bond purchase programmes.
- **ISIN** — unique bond identifier.
- **Merton-KMV** — structural credit model; asset vol → PD.
- **PD** — probability of default.
- **MTM** — mark-to-market impact in euros.
