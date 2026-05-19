# CH4Net Validation Research Handoff
## Context for Deep-Research LLM: What We've Done and What Remains Open

**Project:** Satellite-derived methane emission tracking system for ECB/EIB climate risk assessment
**Course:** Columbia MSOR IEOR 4742 (Prof. Ali Hirsa)
**Date of handoff:** 2026-03-26
**Author:** Vincent Cornelius (vincentcorn2@gmail.com)

---

## 1. What This Document Is

The `CH4Net_Experimental_Results.docx` file covers the experimental record chronologically — what we ran, what the numbers were, and what immediate interpretations we drew. This document is different: it focuses on the *why*, the *unresolved tensions*, the *methodological subtleties*, and the *open research questions* that should inform the next phase of work. It is intended to give another LLM the richest possible context to reason about what should happen next.

---

## 2. Full Technical Stack and Data Flow

### 2.1 Ingestion
- **Source:** Copernicus Data Space Ecosystem (CDSE) — Sentinel-2 Level 1C tiles in SAFE format
- **Client:** `src/ingestion/copernicus_client.py` handles OAuth2 token refresh and STAC-based tile search by bounding box + date + cloud cover threshold
- **Preprocessing:** `src/ingestion/preprocessing.py` — tiles the full Sentinel-2 scene (10980×10980 px at 10m native res, resampled to 20m for bands 11/12) into overlapping or non-overlapping patches. Training and inference both use `tile_size=160, overlap=0`. The `tile_scene()` function returns `TilePatch` objects with pixel-coordinate metadata; `stitch_predictions()` reassembles them into a full-scene probability map. All 12 Sentinel-2 bands (B1–B12) are used as model input, not just the SWIR bands.
- **Cache format:** Scenes are saved as `.npy` files (H, W, 12) in uint8 format alongside a `_geo.json` sidecar containing tile_id, acquisition_date, CRS string, and 6-element Affine transform for georeferencing.

### 2.2 Model Architecture
- **Paper:** Vaughan et al. 2024, *Atmospheric Measurement Techniques* — "CH4Net: A deep learning model for methane plume detection in Sentinel-2 imagery"
- **Architecture:** Standard U-Net with 4 encoder stages and 4 decoder stages, skip connections, bilinear upsampling in decoder. Input: (B, 12, 160, 160). Output: (B, 160, 160, 1).
- **Key parameter:** `div_factor` controls channel width. `div_factor=8` gives channels [8, 16, 32, 64, 64] → ~214K parameters. `div_factor=1` gives [64, 128, 256, 512, 512] → ~13.5M parameters. The paper specifies div_factor=8.
- **Critical implementation detail:** The named submodule structure matters for weight loading. The training script (`approach_c_retrain.py`) uses named `_DoubleConv(nn.Module)`, `_Down(nn.Module)`, `_Up(nn.Module)` classes, producing state_dict keys like `inc.net.0.weight`, `down1.net.1.net.0.weight`. Any inference code that uses anonymous `nn.Sequential` objects instead produces keys like `inc.0.weight` and will fail with a load_state_dict key mismatch. This was a bug that was fixed in `src/detection/ch4net_model.py`.
- **Inference wrapper:** `CH4NetDetector` in `ch4net_model.py` — handles weight loading (both raw state_dict and `{"model_state_dict": ...}` wrapper formats), sigmoid activation, threshold application, connected-component labeling for blob detection.

### 2.3 Training
- **Dataset:** `av555/ch4net` on HuggingFace (official Vaughan et al. release) — 8,255 train, 255 val, 2,473 test. Variable-size crops (roughly 160–230px in each dimension), center-cropped/zero-padded to 160×160 for batching.
- **Label format:** Binary float64 masks — 1.0 = confirmed methane plume pixel, 0.0 = background. The `mbmp` (RGBA visualization) field in the dataset is NOT the ground truth mask.
- **Training script:** `approach_c_retrain.py` — Adam optimizer, lr=1e-3 with ReduceLROnPlateau, BCEWithLogitsLoss (model outputs raw logits during training, sigmoid applied separately at eval), batch_size=16, 50 epochs. Best checkpoint saved by val F1 at threshold=0.5.
- **Infrastructure:** Google Colab free tier. Critical I/O bottleneck: training from Google Drive directly was extremely slow (~hours per epoch) due to small file I/O overhead across ~11K individual .npy files. Fix: `shutil.copytree()` to Colab's local SSD first, then train. Copy took ~16 minutes for 9GB.
- **Training results:**
  - Run 1: best val F1 = 0.2733 at epoch 34 of 50
  - Run 2: best val F1 = 0.2679 at epoch 23 of 50
  - Both runs plateau well below the paper's reported ~0.35 F1. Root cause unclear — candidates include: (a) insufficient epochs, (b) the 160×160 fixed-crop standardization losing context from larger natural crops, (c) class imbalance in the training set not being addressed (no positive-sample oversampling or weighted loss), (d) no data augmentation (flips, rotations, intensity jitter).
- **Weights file:** `weights/ch4net_div8_retrained.pth` — raw state_dict, 128 weight tensors, 214,385 parameters.

### 2.4 Quantification Pipeline
- **IME (Integrated Mass Enhancement):** `src/quantification/ime.py` — currently a geometric proxy. Full formula: Q = (IME × U_eff) / L where IME = Σ ΔX_CH4,i × A_i. Phase 1 uses an assumed enhancement of ~0.003 kg/m² as a proxy; Phase 3 would replace this with Beer-Lambert radiative transfer retrieval using Band 11/12 ratios.
- **CEMF (Column Enhancement Mass Flux):** `src/quantification/cemf.py` — spectral retrieval of total_mass_kg from SWIR absorption signature, feeds into `CEMFIntegratedMassEnhancement.estimate_from_cemf()`.
- **ERA5 wind integration:** Completed by teammate. `era5_client.py` pulls reanalysis wind vectors (u, v components at 10m) for each scene's location and time. These wind speeds feed into IME as U_eff. Emission records are logged to `results/emission_timeseries.jsonl` via `emission_logger.py`, with fields including `q_cemf_kg_per_hour`, `wind_speed_ms`, `wind_source`, and `ira_waste_charge_usd_2026` (applying the 2026 IRA Waste Emissions Charge of $1,500/metric ton).
- **Entity resolution:** `src/entity_resolution/resolver.py` — matches detected plume centroids to known facility databases (JRC-PPDB-OPEN, GEM, GFED) to attribute emissions to specific industrial entities for ESG/regulatory reporting.

---

## 3. The European Power Plant Validation in Detail

### 3.1 Site Selection Rationale

We chose European validation sites specifically to test **geographic generalization**: the model was trained on data dominated by Turkmenistan/Central Asian terrain. If it only fires on sandy desert terrain and misses or suppresses European signals, it cannot be used for the ECB/EIB's primary interest area (European infrastructure).

Three categories of European sites were tested:

**JRC-PPDB-OPEN v1.0 (Joint Research Centre Power Plant Database):**
This is the EU's official database of power plants, publicly available. We used it because it provides GPS-precise coordinates for large fossil fuel facilities with known operational capacity in MW. We selected:
- **Emsland gas power plant, Germany** (52.481°N, 7.306°E): 1,820 MW combined-cycle gas turbine, Lingen, Lower Saxony. Confirmed as JRC ID entry. Chosen because it's one of the largest gas plants in Germany and is in a predominantly flat agricultural area (Münsterland) — relatively "clean" terrain with few other strong SWIR signals.
- **Eemshaven gas power plant, Netherlands** (53.437°N, 6.881°E): 1,410 MW, coastal northern Netherlands. Chosen because it's in the same Sentinel-2 tile as Groningen, allowing cross-validation.

**TROPOMI-confirmed field:**
- **Groningen gas field** (53.252°N, 6.682°E): Europe's largest natural gas production area. Known CH4 emitter with confirmed TROPOMI Level 2 detections. We derived the test coordinates from clustering prior CH4Net detection centroids from that region. This is the strongest positive control in our European set — unlike the JRC plants (which may or may not be visibly emitting on the specific acquisition date), the Groningen field has documented continuous elevated CH4.

### 3.2 Control Site Design

Each emitter was paired with a "background" control in the same tile:
- **Emsland control:** Rural farmland ~20km SW of the plant (52.300°N, 7.050°E), in the same T32ULC tile. Selected specifically because the CH4Net model under the old (broken) weights was *already firing here* — which gave us a diagnostic: if the new model also fires more at the control than the emitter, the terrain-detection problem persists.
- **Eemshaven control:** Rural Groningen province farmland ~20km south (53.250°N, 6.600°E), no known sources.
- **Groningen field control:** North Sea coastal area ~10km NW of the field (53.340°N, 6.500°E). Chosen as minimal surface reflectance area.

The 200×200 pixel (2km×2km) crop size was chosen to match a plausible satellite detection footprint at 10m Sentinel-2 resolution — large enough to capture a diffuse plume, small enough to be spatially diagnostic.

### 3.3 Results and Their Implications

| Site | Signal/Control Ratio | Interpretation |
|------|---------------------|----------------|
| Groningen gas field (TROPOMI) | **2.95×** | Strong: model preferentially activates at Europe's largest documented CH4 emitter vs. coastal background |
| Eemshaven gas plant (JRC) | **1.38×** | Moderate: above control but not the 1.5× threshold we used for "strong" |
| T7 (Darvaza, Turkmenistan) | **1.14×** | Weak: barely above control in in-domain data |
| T6 (Darvaza, Turkmenistan) | **0.955×** | Suppressed: control fires more than emitter |
| Emsland gas plant (JRC) | **0.694×** | Suppressed: model fires more at rural farmland than at the 1,820 MW gas plant |
| T14, T17 (Balkanabat) | Out of bounds | These coordinates fall in tile T40RBJ, not T40SBJ — adjacent tile not yet cached |

**The Groningen result is the most important finding.** The ratio of 2.95× means the retrained model is spatially specific at a TROPOMI-confirmed European CH4 emitter. This cannot be dismissed as terrain noise — the control is coastal North Sea, which is actually more "interesting" terrain (water-land boundary) than rural Netherlands. The fact that the emitter wins suggests the model has learned something about plume signatures.

**The Emsland suppression is a significant open question.** Emsland (S/C = 0.694) is a large industrial gas plant in flat agricultural terrain. Several competing hypotheses:
1. The plant was not actively emitting on the acquisition date (2024-06-27). Gas turbines cycle on/off — without knowing the plant's operational schedule, we cannot confirm it was generating at the time.
2. The model's training distribution does not include European industrial emissions — it was trained on Vaughan et al.'s dataset which is dominated by Turkmenistan/Middle East super-emitters. European gas plant plumes are smaller and more diffuse.
3. The agricultural terrain in Münsterland has enough spectral complexity (mixed crops, bare soil, irrigation) that it genuinely produces false positives at the control site.

The Emsland result illustrates a key limitation: **JRC coordinates tell us where a plant is, not whether it was emitting on a specific day.** Groningen is different because it's a producing gas field with documented continuous emissions.

### 3.4 What We Have NOT Done Yet (European Validation)

- **TROPOMI co-location for Groningen on 2024-06-28:** We have the Sentinel-2 acquisition but not the TROPOMI Level 2 CH4 product for the same date. Cross-validating that TROPOMI shows elevated XCH4 on that exact day over Groningen would transform our 2.95× result from "suggestive" to "confirmed." TROPOMI data is freely available via the Copernicus Open Access Hub for mission S5P product `L2__CH4___`.
- **Emsland operational schedule lookup:** Checking whether the Emsland plant was dispatched on 2024-06-27 would disambiguate hypothesis 1 vs. 3 above.
- **Broader JRC site sampling:** We tested only 2 JRC plants. The database has hundreds of large gas/coal facilities in Europe. A systematic sweep of 10–20 large facilities across multiple dates would give statistical power to understand whether the Emsland result is an outlier or representative.
- **Negative control tiles (no large facilities):** We have never run a tile that is purely agricultural/forest with no known emitters and tested whether the model is quiet. This is needed to estimate false positive rate on European terrain specifically.

---

## 4. The Turkmenistan Validation in Detail

### 4.1 The In-Domain Failure Problem

The most alarming finding across all experiments is that the retrained model performs **worse in-domain (Turkmenistan) than out-of-domain (Europe)** on the metrics that matter for spatial specificity:
- T6 (paper-confirmed emitter): S/C = 0.955 → suppressed
- T7 (paper-confirmed emitter): S/C = 1.14 → weakly above
- Approach B at threshold 0.18: E/C ratio = 0.59 (barely better than broken model's 0.58)

This is paradoxical: the model was trained on Turkmenistan data. Why does it fail to be specific at the very sites Vaughan et al. identified as the ground truth?

The most likely explanation: **the training dataset is not organized by geographic location of emission.** The `av555/ch4net` HuggingFace dataset contains 8,255 train samples that are 160×160 crops taken from TROPOMI-guided sampling across multiple years and geographic regions. The T6/T7/T14/T17 sites from Table 1 of the Vaughan et al. paper are specific acquisition dates and pixel coordinates that may or may not be well-represented in the training crops. The model learns to recognize the *spectral signature* of a plume but the validation approach (comparing mean probability at a GPS point vs. 2km north) may not be the right test.

An alternative framing: maybe the F1=0.27 model is working but the control placement (+2km north on the same acquisition date) is too similar spectrally — if the atmosphere is uniformly methane-enriched over a large area (which TROPOMI suggests for the Turkmenistan Balkanabat region), there may be no local spatial contrast to detect.

### 4.2 Seasonal Control

One tile we have cached but did not fully exploit: `T40SBJ 2021-01-29` — the same Balkanabat tile in winter. This is a powerful control because:
- January temperatures in Karakum are 0–10°C, dramatically different surface albedo
- Gas production often continues year-round but emissions patterns may differ
- If the model fires equally on the 2021 winter image as the 2024 summer image, it's detecting terrain not plumes

We ran inference on this tile but did not systematically compare probabilities between dates.

---

## 5. Approach B: The Emission/Clean Comparison

### 5.1 Design
- **Emission polygon:** POLYGON((53.6 39.3, 54.0 39.3, 54.0 39.7, 53.6 39.7)) — covers TROPOMI-dense red CH4 area west of Balkanabat. Uses combined pixel counts from T40TBK and T40SBJ (adjacent tiles covering the same polygon).
- **Clean polygon:** POLYGON((56.0 38.3, 56.4 38.3, 56.4 38.7, 56.0 38.7)) — Magtymguly, Turkmenistan, TROPOMI-confirmed clean.
- **Logic:** If the model learned anything about methane, the E/C ratio (emission pixel count / clean pixel count at each threshold) should exceed 1.0.

### 5.2 Result
The ratio never exceeded 1.0 at any threshold from 0.05 to 0.90. It improved slightly from 0.58 (broken model) to 0.59 (retrained model). This is the most concerning result quantitatively.

### 5.3 Why This Might Not Be Fatal

The Approach B test is essentially asking: "across an entire 10980×10980 tile, does the emission region have more activated pixels than the clean region?" This is a very hard test because:
1. The "emission tile" (T40SBJ) has ~110M pixels, of which only a small fraction overlie actual plumes. The vast majority of both tiles are background.
2. The Magtymguly clean tile may have surface features (dunes, evaporite flats, settlements) that share spectral properties with the emission detection signal.
3. If the model learned a narrow plume signature, it might be very specific at the plume scale but meaningless at the tile scale.

The more meaningful test would be to restrict comparison to the TROPOMI-positive sub-polygon. We have the polygon coordinates but did not implement pixel-level polygon masking in the Approach B analysis — it just counts all pixels above threshold across the entire tile.

---

## 6. The Terrain Detector Problem: Root Cause Analysis

### 6.1 What "Terrain Detector" Means Here

The broken div_factor=1 model (~13.5M params, trained on a subset of ~925 samples) fired on everything — it identified Karakum desert sand as "methane plume" because the B11/B12 SWIR reflectance of bare sand in certain conditions resembles the absorption profile of methane. The old model had 138,824 "plume pixels" on the known emission tiles AND 239,395 "plume pixels" on the clean tile. It was simply detecting sandy desert.

### 6.2 Has Retraining Fixed This?

Partially. The key evidence is the terrain generalization section:
- **European tiles (Netherlands/Germany):** mean probability ≈ 0.0082
- **Turkmenistan (Karakum desert):** mean probability ≈ 0.022
- **Ratio:** 2.68× — the model still fires ~3x more on Karakum terrain than European terrain

This is progress — the old model fired indiscriminately. But the 2.68× ratio means the model retains significant terrain sensitivity. The question is whether this is a feature (Karakum genuinely has more CH4) or a bug (terrain spectral similarity to training-data positives).

### 6.3 What Would Actually Fix It

The training dataset (`av555/ch4net`) includes "hard negatives" — non-emitting scenes from diverse terrain types. But if the dataset is geographically biased toward Karakum and Central Asian scenes (where the paper's ground truth sites are), the model will still learn Karakum-specific texture features. Options:
1. **Data augmentation:** Random brightness/contrast jitter on B11/B12 to break terrain-spectral correlation
2. **Hard negative mining from European tiles:** Take confirmed-clean European scenes, label them as 100% background, add to training. This directly teaches the model that European farmland is not methane.
3. **Band dropout:** Train with random channel masking to prevent over-reliance on specific band combinations correlated with terrain type
4. **Two-stage training:** Fine-tune on European tiles with confirmed labels (Groningen positive, known-clean areas negative) after initial training on the full Vaughan et al. dataset

---

## 7. Other Methodological Avenues Considered

### 7.1 CEMF (Column Enhancement Mass Flux)
Spectral retrieval approach where Band 12 / Band 11 ratio anomalies are used to directly estimate column CH4 enhancement via Beer-Lambert law: `T_plume = L12/L11 = exp(-AMF × σ_CH4 × ΔX_CH4)`. The code stub exists in `src/quantification/cemf.py`. This is Phase 3 work — it requires a pre-computed lookup table (LUT) of AMF (Air Mass Factor) values as a function of solar zenith angle, view angle, and surface albedo. Not yet implemented.

### 7.2 Cross-Sectional Flux (CSF)
Alternative quantification method: integrate mass flux across perpendicular transects downwind of the source. Requires wind direction to define the "downwind" axis. The ERA5 integration now provides this. CSF is more robust than IME when the plume has a clear elongated structure. Stub exists in `ime.py` as `CEMFIntegratedMassEnhancement`.

### 7.3 TROPOMI-CH4Net Fusion
TROPOMI (on Sentinel-5P) has 5.5km × 7km pixel resolution but global daily coverage and is directly calibrated in XCH4 (ppb). Sentinel-2 has 10–20m resolution but only ~5-day revisit. The natural fusion: use TROPOMI to identify areas/dates of elevated CH4, then pull Sentinel-2 for fine-grained plume localization. We use this logic in the validation design (TROPOMI-confirmed sites as ground truth) but have not yet implemented an operational TROPOMI-query step in the ingestion pipeline.

### 7.4 T40RBJ Tile (Balkanabat Sites T14/T17)
The Vaughan et al. Table 1 sites T14 (38.55747°N, 54.20049°E) and T17 (38.49393°N, 54.19764°E) have row coordinates (row ~12865, ~13569) that exceed the 10980-row height of tile T40SBJ. These sites are in the adjacent tile T40RBJ (one tile south). We have never downloaded T40RBJ. This means 2 of the 4 Turkmenistan paper-confirmed emitter sites have never been tested.

### 7.5 Temporal Consistency Testing
We have two acquisitions of T31UET (Rotterdam/Amsterdam area): 2024-06-26 and 2024-06-28. We have not compared the probability maps between these dates. If the model is detecting real signals (industrial sources, agricultural CH4), we'd expect some spatial consistency. If it's detecting atmospheric artifacts or clouds, maps should be uncorrelated. This is a free diagnostic we haven't run.

### 7.6 Threshold Calibration
The detection threshold of 0.18 was chosen empirically from earlier sessions based on the ratio of false positives to detections. The training script uses threshold=0.5 for val F1 calculation. There may be a threshold gap — the model was optimized at 0.5 but we're evaluating at 0.18. Running the val set through at threshold 0.18 to get a calibrated F1 would tell us if 0.18 is appropriate. The plateau in val F1 at training time (0.27) is measured at 0.5; at 0.18, the true F1 might be meaningfully different.

---

## 8. What the API Layer Looks Like

The pipeline has a fully implemented FastAPI layer (`src/api/main.py`) with endpoints:
- `POST /detect` — accepts a scene_id, triggers Copernicus download, runs CH4Net inference, returns detection results
- `POST /quantify` — takes a detection result and runs IME/CEMF to produce kg/h estimates
- `GET /emissions/{entity_id}` — queries the emission time-series log for a specific entity

The Pydantic schemas (`src/api/schemas.py`) include fields for IRA Waste Emissions Charge calculations, which is the financial output the ECB/EIB is interested in.

---

## 9. Open Research Questions for Next Phase

These are the specific questions where deep research would be most valuable:

**Q1: How do we get TROPOMI Level 2 XCH4 data for Groningen on 2024-06-28?**
The Copernicus Open Access Hub (scihub.copernicus.eu) provides S5P L2 products. The product is `L2__CH4___`. We need to understand the API/download procedure and data format (NetCDF4, variable `methane_mixing_ratio_bias_corrected`) to extract a TROPOMI overpass collocated with our Sentinel-2 acquisition.

**Q2: What training modifications would most efficiently raise val F1 above 0.35?**
Specifically: is the bottleneck (a) data volume, (b) class imbalance, (c) augmentation absence, (d) architecture capacity (div_factor=8 may be too small — the paper reports 0.35 F1 but that may have been with a different training setup), or (e) the 160×160 fixed crop size discarding context from the natural variable-size crops?

**Q3: Is Groningen 2.95× a real signal or an artifact of the specific control placement?**
The control was placed at 53.340°N, 6.500°E (North Sea coastal). If there is something spectrally unusual about that specific coastal pixel patch (e.g., glint, shallow water, shipping lanes) that the model predicts as "non-plume," the 2.95× might reflect the control being anomalously quiet rather than the emitter being anomalously hot.

**Q4: What is the detection threshold for Sentinel-2 CH4 plumes in terms of emission rate?**
Vaughan et al. 2024 report detection rates as a function of emission strength. What is their stated minimum detectable emission rate (in t/h) at Sentinel-2 spatial resolution? The Groningen gas field emits at a rate that should be easily above this threshold; the Emsland plant may emit at a rate that is marginal depending on operational mode.

**Q5: How should we handle the class imbalance in training?**
The dataset has 8,255 train samples but the positive:negative pixel ratio within each image is probably <<1% (plumes are small features in large scenes). BCEWithLogitsLoss without pos_weight treats this imbalance naively. What is the effect of setting `pos_weight = (total_pixels / plume_pixels)` in the loss function? The paper methodology section should specify what Vaughan et al. did.

**Q6: Is there a public benchmark for CH4 detection from Sentinel-2 that we can compare against?**
Beyond the Vaughan et al. paper itself, are there other published approaches (e.g., matched filter, multiband anomaly detection, transformer-based architectures) with published F1 scores on the same `av555/ch4net` dataset that would give us an external benchmark?

**Q7: What does "best val F1 = 0.27 vs. paper's ~0.35" actually mean in practice?**
In binary segmentation, F1 is computed at the pixel level. A F1 of 0.27 vs. 0.35 sounds bad but may or may not matter operationally — the question is whether the *spatial ranking* of probabilities is correct even if the absolute threshold calibration is off. ROC-AUC might be a better metric to determine if the model has learned useful representations.

---

## 10. Summary of What the Retrained Model Actually Is

At this point, the retrained CH4Net (div_factor=8, best val F1=0.27, 214K params) is best characterized as:

- A model that fires ~3× more on Karakum desert terrain than European agricultural terrain (terrain bias significantly reduced but not eliminated)
- Spatially specific at Groningen gas field (Europe's largest documented CH4 field) with 2.95× S/C ratio
- Weakly specific at Eemshaven gas plant (1.38×)
- Suppressed at Emsland gas plant (0.694×) — unknown whether due to model failure or actual non-emission on acquisition date
- Completely failing to be globally calibrated — the "emission tile fires more than clean tile" test (Approach B) still fails at all thresholds
- Still producing its best F1 ~0.08 below the paper's reported performance, suggesting undertraining

It is not yet a production-ready methane detector. It is a research prototype that shows one compelling signal (Groningen) and several diagnostic failures. The next phase should be targeted at understanding and resolving the Approach B failure specifically, since that is the most operationally relevant test.
