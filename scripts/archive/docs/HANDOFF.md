# CH4Net European Retraining -- Project Handoff

**Date**: 2026-04-08 (last updated 2026-04-09 ~01:10 local)
**Project**: ECB/EIB Methane Monitoring -- European Fine-tuning of CH4Net v2
**Location**: `~/Downloads/methane-api/`

---

## 0. Current Status (pick up from here)

| Task | Status |
|------|--------|
| Tile downloads (41/41) | ✅ Complete |
| Crop extraction | ✅ Complete — 25 crops (12 neg + 13 pos) |
| Training v1 (60 epochs, european_model.pth) | ✅ Complete — val_loss=0.7926, val_acc=80% |
| Bitemporal eval — Weisweiler | ✅ Done (S/C baseline=1.000, bitemporal=1.000) |
| Bitemporal eval — Boxberg | ✅ Done (S/C baseline=0.358, bitemporal=0.035) |
| Bitemporal eval — Rybnik | 🔄 In progress — baseline DONE (S/C=3.708), bitemporal running |
| Bitemporal eval — Groningen | ⏳ Queued (after Rybnik finishes, ~4h) |
| Training v2 (larger dataset) | ⏳ Run after bitemporal finishes |

**Immediate next command** (when bitemporal finishes):
```bash
cd ~/Downloads/methane-api && conda activate methane && python approach_c_retrain.py
```

---

## 1. Project Overview

This project fine-tunes CH4Net v2 (a U-Net for methane plume segmentation from Sentinel-2 imagery) to detect European methane emissions from coal mines, gas compressors, gas storage facilities, and gas terminals. The original model was trained on the global MARS-S2L dataset (Vaughan et al., 2024, AMT). Our work validates the model against TROPOMI XCH4 satellite data and retrains on European-specific examples.

### Key Findings So Far

- **41 TROPOMI-confirmed positive detections** across 14 European sites (coal mines dominate: Silesian cluster, Romanian mines, German lignite)
- **Rybnik reassessment**: Previously classified as "terrain artifact" due to ring-shaped probability profile. TROPOMI confirmed 5 positive dates (max +19.7 ppb). The ring increase is caused by adjacent Silesian mines (Jastrzebie, Knurow, Pniowek, Zofiowka) all emitting simultaneously.
- **Groningen explained**: CH4Net S/C=4.191 is mislocalization -- detecting gas infrastructure at nl_grijpskerk compressor station on the same tile, not the depleted Groningen field itself. TROPOMI confirmed non-detection.
- **European fine-tuned model trained**: 60 epochs, best val_loss=0.7926, val_acc=80%. Small dataset (19 train / 5 val) limits performance.

---

## 2. Environment Setup

```bash
conda activate methane
```

All scripts require the `methane` conda environment. Running in `(base)` will fail with `ModuleNotFoundError: No module named 'numpy'`. Key packages: PyTorch, numpy, rasterio, requests.

### Copernicus API Credentials

For tile downloads, set environment variables:
```bash
export COPERNICUS_USER='your_username'
export COPERNICUS_PASS='your_password'
```

These MUST be set before running download scripts in nohup/background mode (scripts cannot prompt for stdin in background).

---

## 3. Critical Technical Details

### 3.1 Model Architecture -- div_factor=1, NOT 8

**CRITICAL**: The saved weights (`weights/best_model.pth`) use `div_factor=1` (~14M params, full-size UNet with 64/128/256/512/512 channels), NOT `div_factor=8` (~214K params) as the paper describes. The `out.weight` shape `[1, 128, 1, 1]` confirms div_factor=1.

Both `ch4net_model.py` and `approach_c_retrain.py` now auto-detect div_factor from the checkpoint:

```python
if "out.weight" in state_dict:
    in_ch = state_dict["out.weight"].shape[1]
    div_factor = max(1, 128 // in_ch)
```

### 3.2 State Dict Key Remapping

Saved weights use flat Sequential indexing. Current model code uses `.net`-wrapped attributes. Regex remapping is applied unconditionally (idempotent):

```python
k = re.sub(r'^inc\.(\d)',          r'inc.net.\1',       k)
k = re.sub(r'^(down\d)\.1\.(\d)', r'\1.net.1.net.\2',  k)
k = re.sub(r'^(up\d\.conv)\.(\d)', r'\1.net.\2',       k)
```

This is already implemented in both `ch4net_model.py` (CH4NetDetector) and `approach_c_retrain.py` (load_model).

### 3.3 Band Mapping

- B11 = index 10 (1610nm SWIR reference)
- B12 = index 11 (2190nm methane absorption)
- All 12 Sentinel-2 bands used as model input
- L1C data required (L2A atmospheric correction destroys methane signal)

### 3.4 S/C Ratio

Signal-to-Control ratio: mean CH4Net probability over a plant-centred crop divided by mean probability over a 20km-offset control crop. S/C > 1.15 suggests real emission; S/C < 1.15 suggests terrain artifact or non-detection.

### 3.5 MGRS Tile Gotchas

- **T34UCB** has NO L1C products (only L2A) -- fallback to T34UCA
- **T31UGS** -- fallback to T31UFT or T31UGT
- **T31UCB** -- fallback to T31UCU or T30UYE
- Some sites (de_sayda, ro_medias) have pixel coords outside their fallback tiles -- no crops extracted for those

---

## 4. File Structure

### 4.1 Key Scripts

| Script | Purpose | Status |
|--------|---------|--------|
| `approach_c_retrain.py` | Fine-tune CH4Net on European crops | **DONE** -- 60 epochs, best val_loss=0.7926 |
| `apply_bitemporal_diff.py` | Bi-temporal evaluation on 4 test sites | **PARTIALLY DONE** -- Weisweiler+Boxberg done, Rybnik+Groningen pending |
| `extract_training_crops.py` | Extract 200x200x12 crops from full tiles | **DONE** -- 24 crops (12 pos + 12 neg) |
| `download_training_tiles.py` | Download S2 L1C tiles for positives | **DONE** -- 20/41 tiles downloaded (originally 35, but some recount) |
| `download_reference_tiles.py` | Download winter reference tiles for bitemporal | **DONE** -- all 4 sites cached |
| `run_tropomi_colocation.py` | TROPOMI XCH4 co-location analysis | **DONE** |
| `explore_mars_s2l.py` | Explore MARS-S2L dataset (needs HuggingFace access) | **BLOCKED** -- awaiting dataset approval |
| `src/detection/ch4net_model.py` | Core model definition + inference wrapper | **STABLE** |

### 4.2 Weight Files

| File | Description |
|------|-------------|
| `weights/best_model.pth` | Original CH4Net weights (div_factor=1, ~14M params, flat key format) |
| `weights/european_model.pth` | European fine-tuned weights (best val_loss=0.7926, val_acc=80%) |
| `weights/ch4net_div8_retrained.pth` | Earlier experiment (div_factor=8, from prior session) |

### 4.3 Data Directories

```
data/
  crops/
    positive/       # 12 crops (24 files: .npy + _label.json each)
    negative/       # 12 crops (24 files)
    manifest.json
    dataset_stats.txt
  npy_cache/
    training/       # 20 full S2 L1C tiles (10980x10980x12, ~1.4GB each)
    reference/      # Winter reference tiles for bitemporal diff
    <site_name>/    # Tiles per survey site
```

### 4.4 Results Directories

```
results_analysis/
  tropomi_positives.json    # 41 confirmed TROPOMI positive detections
  retrain_history.json      # Training loss/accuracy curves (60 epochs)
  retrain.log               # Training console log
  bitemporal_comparison.json  # NOT YET CREATED (apply_bitemporal_diff.py incomplete)

results_bitemporal/
  weisweiler/    # baseline + bitemporal GeoTIFFs (DONE)
  boxberg/       # baseline + bitemporal GeoTIFFs (DONE)
  rybnik/        # empty (IN PROGRESS when session ended)
  groningen/     # NOT YET CREATED

results_expanded/   # Original expanded survey results
results_validation/ # Validation survey results
```

---

## 5. Training Details

### 5.1 Dataset

- **24 crops total**: 12 positive (TROPOMI-confirmed), 12 negative (S/C < 1.15)
- **Split**: 19 train (10 pos + 9 neg), 5 val (2 pos + 3 neg)
- **Crop size**: 200x200x12 uint8 .npy files (random 160x160 subcrop during training)
- **Positive mask**: Gaussian disk (sigma=30px) centred on site pixel
- **Negative mask**: all-zero
- **Loss**: BCEWithLogitsLoss with pos_weight=5.0
- **Augmentation**: horizontal flip, vertical flip, transpose (train only)

### 5.2 Training Configuration

```
LR=1e-4, Adam optimizer
StepLR decay: 0.95 every 10 epochs
Early stopping patience: 15 epochs
Batch size: 4
Starting weights: weights/best_model.pth
```

### 5.3 Results

- Best val_loss=0.7926 at epoch 45
- Val accuracy: 80% (from epoch 36 onward)
- Train accuracy: ~81%
- Early stopping triggered at epoch 60
- Model is learning the European emission signature but limited by small dataset size

### 5.4 Known Edge Cases

- **de_bad_lauchstaedt**: Produces 200x141 crops (site near tile boundary) -- silently skipped by training
- **de_sayda, ro_medias**: Pixel coords outside fallback tiles -- no crops extracted
- **Minor warning**: `torch.tensor(labels)` in approach_c_retrain.py should be `labels.clone().detach()` to suppress a PyTorch UserWarning

---

## 6. Bi-Temporal Evaluation Status

`apply_bitemporal_diff.py` computes CH4Net predictions on original tiles vs bi-temporal difference arrays (delta_B12 = target_B12 - reference_B12, shifted to [0,255] with 128=neutral).

### Completed

| Site | Baseline S/C | Bitemporal S/C | Notes |
|------|-------------|----------------|-------|
| Weisweiler | 1.000 | 1.000 | Unexpected -- prior survey had S/C ~2.0. May indicate S/C calculation differs from original survey scripts |
| Boxberg | 0.358 | 0.035 | Both non-detection. Boxberg was always weak in original survey too |

### Pending (was running when session ended)

| Site | Status |
|------|--------|
| Rybnik | Baseline pass in progress (~3000/11881 tiles processed). Then bitemporal pass needed. ~2-4 hours remaining on CPU |
| Groningen | Not yet started. Both passes needed. ~2-4 hours on CPU |

**To check status**: Look for output in the terminal running `apply_bitemporal_diff.py`. When complete, results appear in `results_bitemporal/` and `results_analysis/bitemporal_comparison.json`.

---

## 7. Pending Tasks (Priority Order)

### 7.1 Finish apply_bitemporal_diff.py (Groningen remaining)

**Status as of ~21:10 local**: Rybnik baseline DONE (S/C=3.708 — strong positive signal, confirmed real emitter). Rybnik bitemporal was running. Groningen still queued.

**KEY FINDING**: Rybnik S/C=3.708 is the strongest baseline signal of all 4 test sites. The script's label "Terrain artefact (ring profile increases outward)" is **wrong** — this is a confirmed emitter. Update this note in apply_bitemporal_diff.py when you get a chance.

If apply_bitemporal_diff.py is no longer running when you pick this up, restart it — it will skip all completed sites:
```bash
cd ~/Downloads/methane-api && conda activate methane && caffeinate -i python apply_bitemporal_diff.py
```

Estimated ~4h remaining for Groningen (both passes). Use `caffeinate -i` to prevent sleep.

### 7.2 Training Tile Downloads — ✅ COMPLETE (2026-04-09 ~01:03 UTC)

**41/41 tiles ready. 0 errors.**

Final run completed: 15 new downloads + 26 cached/shared. All source types covered:
- coal_mine: 9/9, coal_mine_plant: 5/5, gas_compressor: 18/18
- gas_processing: 4/4, gas_storage: 1/1, gas_terminal: 4/4

All countries: DE 11, FR 1, NL 5, PL 14, RO 6, UK 4

Manifest saved to `results_analysis/training_manifest.json` and `results_analysis/training_summary.txt`.

**Root cause of all prior download failures**: Script was run either (a) from the wrong directory (`~` instead of `~/Downloads/methane-api/`), or (b) without `conda activate methane`. The fix is always:
```bash
cd ~/Downloads/methane-api
export COPERNICUS_USER='...'
export COPERNICUS_PASS='...'
conda activate methane && python download_training_tiles.py
```

### 7.3 Re-extract Crops — ✅ DONE (2026-04-09 ~01:05 UTC)

Result: **25 crops (12 neg + 13 pos)**, up from 24. Only 1 new positive added (ro_totea 2023-02-21 +18.6ppb). The gain was small because most newly downloaded tiles are shared (same MGRS tile used across multiple sites).

**Permanently out-of-bounds sites** — no fix available without different tile downloads:
- **ro_medias**: pixel (-1624, 6027) — site falls outside left edge of tile T34TGR
- **de_sayda**: pixel (-2210, 8710) — site falls outside left edge of fallback tile
- **fr_lacq**: pixel (-1130, 9182) — site falls outside left edge of fallback tile T30TXN

These sites are genuinely in a different MGRS tile from what was downloaded. Those exact tiles had no L1C products available in the search window.

### 7.4 Training v2 — Run after bitemporal completes

Dataset grew by 1 crop (24→25). Don't expect dramatic improvement. Consider warm-starting from the existing model:
```bash
cd ~/Downloads/methane-api && conda activate methane && python approach_c_retrain.py --weights-in weights/european_model.pth
```

### 7.4 Evaluate european_model.pth

Run the fine-tuned model on the 4 test sites. This could be done by modifying `apply_bitemporal_diff.py` to accept `--weights` flag, or by creating a new evaluation script that compares `best_model.pth` vs `european_model.pth` predictions on the same tiles.

### 7.5 Second Training Run

With more data from retried downloads, train again starting from `european_model.pth`:
```bash
conda activate methane && python approach_c_retrain.py --weights-in weights/european_model.pth --epochs 100
```

### 7.6 MARS-S2L Dataset Access

Waiting for HuggingFace approval (discussion post already submitted). Contact corresponding author Anna Vaughan (av555@cam.ac.uk) if needed. Not blocking for European retraining, but would provide thousands of additional global training samples.

### 7.7 Report Update

Incorporate into the project report:
- TROPOMI mining findings
- Revised Rybnik assessment (real emitter, not terrain artifact)
- Training results and convergence curves
- Bi-temporal evaluation results (once complete)

---

## 8. Conventions

- **Shell quoting**: Use single quotes only. Double-quoted strings cause `dquote>` prompt issues.
- **Working directory**: ALWAYS `cd ~/Downloads/methane-api` first. Scripts use relative paths and fail silently or with misleading errors from the wrong directory.
- **Conda**: Always prefix commands with `conda activate methane &&`. Running in `(base)` causes `ModuleNotFoundError` or auth failures.
- **Device**: Training uses MPS (Apple Silicon GPU). Full-tile inference in apply_bitemporal_diff.py uses CPU (tiles too large for MPS memory).
- **File format**: All imagery stored as uint8 .npy files, 12 bands, values 0-255. Normalized to [0,1] float32 at model input time.
- **Sleep prevention**: Use `caffeinate -i python script.py` for any long-running inference job (bitemporal eval takes 6-8h on CPU).

---

## 9. Quick Start Commands

```bash
# Check training results
cat results_analysis/retrain_history.json | python -m json.tool

# Check bitemporal evaluation status
ls results_bitemporal/*/
cat results_analysis/bitemporal_comparison.json 2>/dev/null

# Re-run training (from European model)
conda activate methane && python approach_c_retrain.py --weights-in weights/european_model.pth

# Run inference on a specific tile
conda activate methane && python -c "
from src.detection.ch4net_model import CH4NetDetector
import numpy as np
det = CH4NetDetector('weights/european_model.pth')
tile = np.load('data/npy_cache/training/TILE_NAME.npy')
crop = tile[4890:5050, 4890:5050, :]  # 160x160 centre crop
result = det.detect(crop)
print(f'Plume: {result.has_plume}, Confidence: {result.confidence:.3f}')
"

# Check TROPOMI positives
conda activate methane && python -c "
import json
with open('results_analysis/tropomi_positives.json') as f:
    data = json.load(f)
for d in data[:5]:
    print(f'{d[\"site\"]:30s} {d[\"date\"]}  enh={d[\"enhancement_ppb\"]:+.1f} ppb')
"
```
