# Live Pipeline Quick Start
## Connecting Copernicus API → CH4Net Detection

This is the meeting deliverable: an operationalized pipeline that takes
a geographic region + date range and produces georeferenced methane
detection maps.

### What Changed From the Notebooks

| Before (notebooks) | After (live pipeline) |
|---|---|
| Copernicus notebook: manual download, hardcoded credentials, interactive `input()` prompts | Automated: API queries, server-side cloud filtering, batch download |
| CH4Net notebook: loads `.npy` from Hugging Face, trains on GPU, evaluates on val set | Production inference: loads your trained weights, processes any Sentinel-2 scene |
| Gap: no way to feed live Copernicus data into CH4Net | Bridge: `preprocessing.py` converts SAFE `.jp2` → resampled `.npy` with geo-metadata sidecar |
| Output: matplotlib plots in notebook | Output: georeferenced GeoTIFF + JSON event log with lat/lon coordinates |

### New Files

```
src/ingestion/preprocessing.py    ← The bridge (SAFE → .npy + geo JSON)
scripts/live_pipeline.py          ← The pipeline (Copernicus → CH4Net → GeoTIFF)
```

### Setup (one time)

```bash
# 1. Activate your environment
conda activate methane

# 2. Install rasterio (reads satellite band files)
conda install rasterio -c conda-forge

# 3. Set your credentials
cp config/.env.example config/.env
# Edit config/.env with your Copernicus email + password

# 4. Copy your trained CH4Net weights from Google Drive
mkdir -p weights
# Copy best_model.pth into weights/
```

### Run the Pipeline

```bash
# Monitor the Turkmenistan super-emitter sites from the paper
python -m scripts.live_pipeline \
  --region "POLYGON((53.5 39.2, 54.3 39.2, 54.3 39.6, 53.5 39.6, 53.5 39.2))" \
  --start 2021-01-01 \
  --end 2021-01-31 \
  --weights weights/best_model.pth \
  --max-products 3

# The pipeline will:
#   1. Query Copernicus → find L1C tiles with <30% cloud cover
#   2. Download the SAFE archives
#   3. Extract 12 bands, resample to 10m, save as .npy
#   4. Run CH4Net inference (100x100 patch tiling)
#   5. Save detection GeoTIFF in results/
#   6. Log events with lat/lon to results/events_*.json
```

### If You Don't Have Weights Yet

The pipeline still works in download-only mode. It will:
- Query and download Sentinel-2 products
- Convert SAFE → .npy (so you can inspect the data)
- Skip inference (and tell you it's skipping)

This is useful for verifying the Copernicus → preprocessing chain works
before plugging in the model.

### Output Files

```
results/
  detection_T40SBH_2021-01-15.tif   ← Georeferenced probability map
  events_2021-01-01_to_2021-01-31.json  ← All detected plume events
data/
  downloads/S2A_MSIL1C_*.zip         ← Cached SAFE archives
  npy_cache/S2A_MSIL1C_*.npy         ← Converted arrays (reusable)
  npy_cache/S2A_MSIL1C_*_geo.json    ← Geo-metadata sidecars
```

### Key Design Decisions

**Why L1C only (not L2A)?**
L2A applies atmospheric correction that can erase methane absorption
signals. CH4Net was trained on L1C top-of-atmosphere reflectance.
The pipeline filters to L1C by default.

**Why NOT use the EMRDM cloud removal from the previous group?**
Methane plumes exist IN the atmosphere. Cloud removal algorithms
reconstruct the surface UNDER clouds, which would destroy the gas
signal. The pipeline uses raw, unprocessed imagery and relies on
cloud-cover filtering instead.

**Why save geo-metadata as a separate JSON sidecar?**
CH4Net needs raw numpy arrays (no geo info). But after inference,
we need to know WHERE on Earth the detection occurred. The sidecar
preserves CRS + affine transform so we can re-project the output
as a GeoTIFF.

**Why tile into 100x100 patches?**
CH4Net was trained on 100x100 pixel crops. A full Sentinel-2 tile
is 10,980 x 10,980 pixels. We tile it into ~12,000 patches, run
inference on each, then stitch the results back together.
