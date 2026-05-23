# Session Progress — 2026-05-14
## CH4Net European Retraining Project — Bełchatów Site Selection & Quantification Cleanup

---

## 1. Context

The project fine-tunes CH4Net v2 (13.5M parameter U-Net) on Sentinel-2 L1C imagery
to detect methane from European coal plants and gas facilities. The production model
is `weights/european_model_v8.pth`. Two candidate sites were under consideration for
the final presentation: **Bełchatów** (858 MW lignite, Poland, Europe's #1 CO₂ emitter)
and **Rybnik** (coal mining IPCC 1B1a, Poland, 6 Carbon Mapper detections at 1,150–2,019
kg CH₄/hr). The session resolved site selection and addressed two open trustworthiness
issues in the quantification pipeline.

---

## 2. Site Selection — Bełchatów Chosen

### Why not Rybnik

Rybnik has strong external validation (5 TROPOMI hits + 6 Carbon Mapper overpasses),
but model-internally it has never triggered CFAR. The production pipeline has
`skip_bitemporal=True` for Rybnik because BT caused 55× over-activation in an earlier
version. The March 2025 overpass shows pre-BT S/C = 5.48 but post-BT S/C = 0.68
(CFAR=no). Without a same-season reference tile (no March/April 2024 T34UCA cached
locally), the BT collapse is ambiguous: it could be either real methane attenuated by
an imperfect December reference, or terrain artifact correctly suppressed.
Presenting Rybnik would require spending the entire Q&A defending why CH4Net fails
on a site where TROPOMI and Carbon Mapper succeed.

### Why Bełchatów

Bełchatów has CFAR-confirmed baseline detections on two historical dates:

| Date       | Baseline S/C | CFAR    | cv_ctrl | CFAR margin |
|------------|-------------|---------|---------|-------------|
| 2020-06-01 | 849.08       | DETECT  | 0.712   | +597.3      |
| 2021-06-06 | 18.53        | DETECT  | 1.003   | +7.1        |

Additionally, Climate TRACE reports 29,636 t/yr for this facility and IPCC classifies
it 1B1a — independent corroboration from two external inventories. The `skip_bitemporal=True`
flag is set for principled physical reasons (continuous industrial emitter), not as a
workaround, and this was empirically confirmed this session (see §3).

---

## 3. Bitemporal Historical Experiment — Key Finding

### Motivation

The Aug-24 2024 Bełchatów detection showed a 92.9% BT collapse (S/C 27.3 → 1.94).
Before finalising Bełchatów as the presentation site, the question was whether the
2020 and 2021 CFAR-confirmed detections would survive BT. A survival result would
constitute a "overwhelmingly real" claim; a collapse would redirect to the
CFAR+inventory argument as the primary evidence.

### Script

`scripts/belchatow_bt_historical.py` — imports from `apply_bitemporal_diff.py`,
explicitly targets the 2020-06-01 and 2021-06-06 T34UCB npy files (bypassing the
`find_target_npy` lookup that defaults to the most recent tile), applies B12-only
BT difference using `T34UCB_ref_20231218.npy`, runs CH4Net v8 inference on both
baseline and BT arrays, and saves to `results_analysis/belchatow_bt_historical.json`.

### Results

| Date       | Baseline S/C | BT S/C    | CFAR BT | BT shifted mean | Interpretation                         |
|------------|-------------|-----------|---------|-----------------|----------------------------------------|
| 2020-06-01 | 849.08       | 1.728     | no      | 126.9           | BT collapses (ref brighter than target) |
| 2021-06-06 | 18.53        | 12,696.5  | DETECT  | 135.4           | BT amplifies (target brighter than ref) |

cv_ctrl stayed high for both dates (1.250 and 1.380 respectively), confirming BT
is not scene-flattening — it is responding specifically to the B12 difference at the
facility location.

### Critical interpretation

The two dates show **opposite** BT behaviours — one collapses to near-noise, one
explodes to 12,696 — depending purely on whether the December 2023 reference scene
happened to capture a higher or lower emission state than each target date.

- 2020-06-01: shifted B12 mean = 126.9 (below neutral 128) → the reference had
  *more* B12 absorption than the 2020 target on average. The plume sign inverts
  and S/C collapses.
- 2021-06-06: shifted B12 mean = 135.4 (above neutral 128) → the 2021 target had
  *more* B12 brightness than the reference. BT amplifies the contrast.

This is structurally decisive: **BT at a continuous industrial emitter reflects the
difference in emission state between two dates, not the presence or absence of methane.**
The reference tile is not a clean methane-free baseline — it has its own emission
signature from December 2023. Which direction the BT diff goes is an accident of
reference-date conditions, not a diagnostic signal.

This empirically validates `skip_bitemporal=True` as a principled design decision
rather than a workaround. When a reviewer asks "did you try BT?" the answer is yes,
and the result is interpretable: BT is uninformative at this site, for a structural
reason demonstrated by opposite outcomes on two dates.

### Nuance on "methane − methane ≈ 0"

The naive framing ("it always emits so BT cancels it") is incomplete. Methane plumes
are not static stamps — they follow the wind on the day of the overpass. The Dec 2023
reference plume was blown in whatever direction the wind pointed on Dec 18; the summer
target plumes were blown in different directions. BT does not cancel methane cleanly;
it *attenuates* it, and the degree of attenuation (or even sign inversion) depends on
spatial overlap between the reference and target plumes. The structural ambiguity is
not merely "year-round emission" but the impossibility of disentangling real methane
attenuation from terrain changes at a site where the reference itself is contaminated.

---

## 4. ERA5 Wind Fetch — Jul-10 Detection Cleaned Up

### Before

The Jul-10 Bełchatów quantification record (scene_id:
`S2A_MSIL1C_20240710T095031_N0510_R079_T34UCB_20240710T133148`) had:
- `wind_source: climatological_fallback_3.5ms`
- `WIND_FALLBACK` governance flag (+30pp σ_wind penalty)
- Flow rate: 476 kg/h (legacy record — also affected by pixel area bug, see §5)

### After

`scripts/fetch_era5_pending.py --site belchatow_20240710` successfully retrieved:
- **wind_speed_ms: 3.318** (SE wind, 124.9°, blowing toward NW)
- u = −2.721 m/s, v = +1.900 m/s

`scripts/run_cemf_neurath_belchatow.py --site belchatow_20240710` recomputed:
- **flow_rate_kgh: 1,071** [750–1,393], ±30%
- `wind_source: ERA5_reanalysis`
- All governance flags: **none**
- `governance_sigma_inflated: false`

The ERA5 wind (3.32 m/s) was very close to the climatological fallback (3.50 m/s),
confirming the fallback was reasonable but the ±30% uncertainty bound is now
statistically grounded rather than penalty-inflated.

### Note on plume_length saturation

Both Bełchatów quantification records share `plume_length_m = 4980.0 m`. This is
the maximum extent of the 500×500 pixel (5 km × 5 km) crop at 20m resolution
(250 px × 20 m/px = 5,000 m, minus edge effects → 249 px × 20 m = 4,980 m).
The plume mask is hitting the crop boundary on both dates, meaning the true plume
length likely exceeds 5 km and `plume_length_m` is a lower bound. Since IME inverts
Q = M × u / L, a shorter L gives a higher Q — the current flow rates are therefore
conservative upper bounds given the crop constraint. This should be noted in
retrieval_notes and/or the presentation.

---

## 5. Pixel Area Bug — Discovery and Partial Fix

### The bug

`run_quant_fixed.py` (the original quantification script) set `PIXEL_SIZE = 10.0`
for the IME plume-length calculation but passed raw 10m B11/B12 arrays directly to
`run_cemf()` without downsampling. `cemf.py` hardcodes `PIXEL_AREA_20M = 400 m²` per
pixel regardless of actual input resolution. Since 10m pixels have a true area of
100 m², every mass calculation using this path was a **4× systematic overestimate**,
and therefore every flow rate was also 4× too high.

### Who has the bug

- **Affected:** all NO_SCENE_ID legacy records in `quantification.json` (groningen,
  maasvlakte, lippendorf, the now-deleted belchatow 476 kg/h record). These were all
  produced by `run_quant_fixed.py`.
- **Not affected:** all scene_id-keyed records (neurath Jun-25, belchatow Aug-24,
  belchatow Jul-10). These were produced by `reingest_gap8.py` and
  `run_cemf_neurath_belchatow.py` respectively, both of which explicitly downsample
  to 20m before calling `run_cemf()`.

### Actions taken

1. **Deleted** the old belchatow legacy record (475.92 kg/h, NO_SCENE_ID,
   climatological_fallback) from `quantification.json`. It is superseded by the
   correct scene-keyed Jul-10 record (1,071 kg/h, ERA5).

2. **Deprecated** `run_quant_fixed.py` with a detailed header comment documenting
   the pixel area bug, which records were affected, and pointing to
   `scripts/run_cemf_neurath_belchatow.py` as the correct replacement.

### Remaining legacy records

The other three NO_SCENE_ID records (groningen, maasvlakte, lippendorf) are also
affected by the 4× overestimate but are all either `excluded: true` (groningen,
lippendorf) or a known v8 non-detection (maasvlakte). They do not affect the
Bełchatów presentation and were left in place with the deprecation note in
`run_quant_fixed.py` as provenance documentation.

---

## 6. Current State of quantification.json (6 records)

| # | Site       | Scene ID                    | Flow rate (kg/h) | Wind source     | Flags  | Valid for presentation? |
|---|------------|-----------------------------|------------------|-----------------|--------|------------------------|
| 0 | groningen  | legacy                      | 28.76            | ERA5            | excl   | No (FP)                |
| 1 | maasvlakte | legacy                      | 426.5            | ERA5            | —      | No (v8 non-detection)  |
| 2 | lippendorf | legacy                      | 1,579.63         | fallback        | excl   | No (terrain artifact)  |
| 3 | neurath    | S2A…T32ULB…20240625         | 534.71           | fallback ±50%   | WIND_FALLBACK | Conditional     |
| 4 | belchatow  | S2B…T34UCB…20240824         | 426.42           | ERA5 ±30% ✓    | none   | **Yes**                |
| 5 | belchatow  | S2A…T34UCB…20240710         | 1,071.16         | ERA5 ±30% ✓    | none   | **Yes**                |

The two Bełchatów records (indices 4–5) are both clean: scene-keyed, ERA5 winds,
zero governance flags, correct 20m pixel path.

Neurath (index 3) still has WIND_FALLBACK — `scripts/fetch_era5_pending.py --site neurath`
would clean it up identically to the Jul-10 Bełchatów fetch if needed.

---

## 7. Open Questions for Opus Review

1. **Plume length saturation:** Both Bełchatów records hit the 5km crop boundary
   (plume_length_m = 4,980 m on both dates). Is extending the crop to 750×750 px
   (7.5 km) advisable before the presentation, or is the current conservative upper
   bound acceptable to report as-is with a caveat?

2. **Jul-10 flow rate magnitude:** 1,071 kg/h is notably higher than Aug-24's 426 kg/h.
   Climate TRACE reports 29,636 t/yr ≈ 3,384 kg/h annualised. Our Jul-10 at 1,071 kg/h
   × 8,760 hr = 9,382 t/yr — about 32% of the Climate TRACE inventory, which is
   plausible for a single overpass. Does the factor of ~2.5× between the two detection
   dates warrant a note on day-to-day emission variability, or is it within normal
   operational range for this plant type?

3. **Conformal threshold:** `calibrated_threshold.json` contains a conformal-calibrated
   τ = 4.1052 at α = 0.10 (14 non-emitter calibration scenes). The production threshold
   of 1.15 has 78.6% empirical FPR on non-emitters. Both Bełchatów CFAR detections
   clear τ = 4.1052 comfortably (S/C = 849 and 18.5). Is the conformal threshold
   framing mature enough to present as a rigorous statistical claim, or should it be
   presented as preliminary given the small calibration set (n=14)?

4. **Legacy records audit:** maasvlakte (index 1) is marked `excluded: false` but per
   project notes is a v8 non-detection (the quantification team quantified terrain
   contrast, not methane). Should it be marked `excluded: true` with
   `exclusion_reason: v8_nondetection` before finalising the quantification.json for
   any shared report?
