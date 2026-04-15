"""
explore_mars_s2l.py
===================
Download and explore the UNEP-IMEO MARS-S2L dataset from HuggingFace.
Checks:
  1. Dataset structure and metadata fields
  2. How many European examples exist (by country / lat-lon)
  3. Band coverage — does it include all 12 Sentinel-2 bands or only SWIR?
  4. Annotation format — binary mask, enhancement map, or emission rate?
  5. Patch size compatibility with our training pipeline
  6. Writes a filtered CSV of European examples ready for download

Usage:
  conda activate methane
  pip install datasets huggingface_hub pandas tqdm
  python explore_mars_s2l.py

Outputs:
  results_analysis/mars_s2l_summary.json     — full dataset stats
  results_analysis/mars_s2l_european.csv     — European examples only
  results_analysis/mars_s2l_compatibility.txt — pipeline compatibility report
"""

import os, json, sys
from pathlib import Path

# ── Check dependencies ────────────────────────────────────────────────────────
missing = []
for pkg in ["datasets", "huggingface_hub", "pandas", "numpy"]:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)

if missing:
    print(f"Missing packages: {', '.join(missing)}")
    print(f"Install with:  pip install {' '.join(missing)}")
    sys.exit(1)

import numpy as np
import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download

OUT_DIR = Path("results_analysis")
OUT_DIR.mkdir(exist_ok=True)

# ── European bounding box ─────────────────────────────────────────────────────
# Covers continental Europe + UK + Iceland
EUROPE_LAT = (34.0, 72.0)
EUROPE_LON = (-25.0, 45.0)

# ── European countries of interest ───────────────────────────────────────────
EUROPEAN_COUNTRIES = {
    "NL", "DE", "PL", "FR", "GB", "UK", "BE", "LU", "DK", "NO", "SE",
    "FI", "EE", "LV", "LT", "CZ", "SK", "AT", "HU", "RO", "BG", "HR",
    "SI", "IT", "ES", "PT", "GR", "AL", "RS", "BA", "ME", "MK", "UA",
    "BY", "MD", "CH", "IE", "IS", "Netherlands", "Germany", "Poland",
    "France", "United Kingdom", "Belgium", "Romania", "Czech Republic",
}

# ── Our pipeline's expected format ────────────────────────────────────────────
# From run_analysis.py and the training pipeline
PIPELINE_BANDS    = 12        # B01–B09, B8A, B11, B12 (B10 excluded)
PIPELINE_DTYPE    = "uint8"   # 0–255 normalized
PIPELINE_RES_M    = 10        # resampled to 10m
PIPELINE_PATCH    = None      # variable — the full Sentinel-2 tile (10980×10980)
# Training uses crops extracted from these tiles, not fixed-size patches

print("=" * 65)
print("  MARS-S2L Dataset Exploration")
print("  Source: UNEP-IMEO/MARS-S2L (HuggingFace)")
print("=" * 65)

# ── Step 1: Load dataset metadata ─────────────────────────────────────────────
print("\n[1] Loading dataset info (metadata only, no images yet)...")
try:
    # Load just the metadata split first to avoid downloading all imagery
    ds = load_dataset(
        "UNEP-IMEO/MARS-S2L",
        split="train",
        streaming=True,    # stream to avoid downloading everything
        trust_remote_code=True,
    )
    print("    ✓ Connected to MARS-S2L")
except Exception as e:
    print(f"    ✗ Could not load dataset: {e}")
    print("    Try: huggingface-cli login")
    sys.exit(1)

# ── Step 2: Inspect first few examples ───────────────────────────────────────
print("\n[2] Inspecting dataset structure (first 10 examples)...")

examples = []
column_names = set()

for i, example in enumerate(ds):
    if i == 0:
        print(f"    Columns: {list(example.keys())}")
        for k, v in example.items():
            if hasattr(v, "shape"):
                print(f"      {k}: shape={v.shape}, dtype={getattr(v, 'dtype', '?')}")
            elif isinstance(v, (list, tuple)):
                print(f"      {k}: list, len={len(v)}")
            else:
                print(f"      {k}: {type(v).__name__} = {repr(v)[:80]}")
    column_names.update(example.keys())
    examples.append(example)
    if i >= 9:
        break

print(f"\n    Confirmed {len(column_names)} columns: {sorted(column_names)}")

# ── Step 3: Load more examples to assess European coverage ────────────────────
print("\n[3] Scanning full dataset for metadata (may take a few minutes)...")

records = []
for i, ex in enumerate(ds):
    rec = {k: v for k, v in ex.items()
           if not hasattr(v, "shape") and not isinstance(v, (bytes, bytearray))}
    # Extract image shape if present
    for k, v in ex.items():
        if hasattr(v, "shape"):
            rec[f"{k}_shape"] = str(v.shape)
            rec[f"{k}_dtype"] = str(getattr(v, "dtype", "?"))
            rec[f"{k}_min"]   = float(v.min()) if v.size > 0 else None
            rec[f"{k}_max"]   = float(v.max()) if v.size > 0 else None
    records.append(rec)
    if i % 100 == 0:
        print(f"    Scanned {i} examples...", end="\r")

print(f"    Total examples: {len(records)}")
df = pd.DataFrame(records)

# ── Step 4: European filter ────────────────────────────────────────────────────
print("\n[4] Filtering for European examples...")

# Try lat/lon columns first
if "lat" in df.columns and "lon" in df.columns:
    european_mask = (
        df["lat"].between(*EUROPE_LAT) &
        df["lon"].between(*EUROPE_LON)
    )
    df_eu = df[european_mask]
    print(f"    By lat/lon: {len(df_eu)} / {len(df)} examples in European bounding box")

elif "latitude" in df.columns and "longitude" in df.columns:
    european_mask = (
        df["latitude"].between(*EUROPE_LAT) &
        df["longitude"].between(*EUROPE_LON)
    )
    df_eu = df[european_mask]
    print(f"    By latitude/longitude: {len(df_eu)} / {len(df)} examples in European bounding box")

elif "country" in df.columns:
    european_mask = df["country"].isin(EUROPEAN_COUNTRIES)
    df_eu = df[european_mask]
    print(f"    By country field: {len(df_eu)} / {len(df)} European examples")

else:
    print("    WARNING: No lat/lon or country column found — cannot filter by geography")
    print(f"    Available columns: {list(df.columns)}")
    df_eu = pd.DataFrame()

# ── Step 5: Source type breakdown ─────────────────────────────────────────────
print("\n[5] Source type breakdown (full dataset):")
for col in ["source_type", "facility_type", "sector", "type", "category"]:
    if col in df.columns:
        print(f"    {col}:")
        for val, cnt in df[col].value_counts().items():
            print(f"      {val:<30} {cnt:>5} examples")
        break

print("\n    Country distribution (top 20):")
for col in ["country", "Country", "country_code"]:
    if col in df.columns:
        for val, cnt in df[col].value_counts().head(20).items():
            eu_marker = " ← EU" if str(val) in EUROPEAN_COUNTRIES else ""
            print(f"      {str(val):<30} {cnt:>5}{eu_marker}")
        break

# ── Step 6: Image / annotation format ─────────────────────────────────────────
print("\n[6] Image and annotation format:")
shape_cols = [c for c in df.columns if "_shape" in c]
for col in shape_cols:
    unique_shapes = df[col].value_counts().head(5)
    print(f"    {col}:")
    for shape, cnt in unique_shapes.items():
        print(f"      {shape:<30} × {cnt} examples")

# Check band count from shape
image_cols = [c for c in df.columns if c.replace("_shape", "") in
              {"image", "img", "s2", "sentinel2", "bands", "input"}]
if image_cols:
    print(f"\n    Primary image column: {image_cols[0]}")

dtype_cols = [c for c in df.columns if "_dtype" in c]
for col in dtype_cols:
    print(f"    {col}: {df[col].value_counts().to_dict()}")

# ── Step 7: Emission rate / annotation stats ───────────────────────────────────
print("\n[7] Emission rate / quantification fields:")
for col in ["emission_rate", "flux_kgh", "flux", "rate_kgh", "emission",
            "plume_mask", "mask", "label", "annotation"]:
    if col in df.columns:
        if df[col].dtype in [float, int]:
            print(f"    {col}: min={df[col].min():.1f}  "
                  f"max={df[col].max():.1f}  "
                  f"median={df[col].median():.1f}  "
                  f"n_nonzero={( df[col] > 0).sum()}")
        else:
            print(f"    {col}: {df[col].value_counts().head(5).to_dict()}")

# ── Step 8: Pipeline compatibility assessment ─────────────────────────────────
print("\n[8] Pipeline compatibility assessment:")

compat_lines = []

# Band count
band_col = next((c for c in shape_cols if "image" in c.lower() or "band" in c.lower()), None)
if band_col:
    shapes = df[band_col].value_counts()
    first_shape = eval(shapes.index[0]) if shapes.index[0].startswith("(") else None
    if first_shape:
        n_bands = first_shape[-1] if len(first_shape) == 3 else 1
        if n_bands == PIPELINE_BANDS:
            line = f"  ✓ Band count: {n_bands} (matches pipeline's {PIPELINE_BANDS})"
        elif n_bands == 2:
            line = f"  ~ Band count: {n_bands} (SWIR only — pipeline needs all {PIPELINE_BANDS})"
        else:
            line = f"  ~ Band count: {n_bands} (pipeline uses {PIPELINE_BANDS})"
        print("    " + line.strip())
        compat_lines.append(line)

# Annotation type
has_mask = any(c in df.columns for c in ["mask", "plume_mask", "label", "annotation"])
has_rate = any(c in df.columns for c in ["emission_rate", "flux", "rate_kgh"])
if has_mask:
    line = "  ✓ Binary mask available — compatible with current U-Net training"
    print("    " + line.strip())
    compat_lines.append(line)
if has_rate:
    line = "  ✓ Emission rate available — enables regression head training"
    print("    " + line.strip())
    compat_lines.append(line)
if not has_mask and not has_rate:
    line = "  ? No mask or emission rate column found — annotation format unclear"
    print("    " + line.strip())
    compat_lines.append(line)

# Key gap
if len(df_eu) == 0 or len(df_eu) < 20:
    line = f"  ! European examples: {len(df_eu)} — may need TROPOMI mining to supplement"
    print("    " + line.strip())
    compat_lines.append(line)
else:
    line = f"  ✓ European examples: {len(df_eu)} — sufficient for fine-tuning"
    print("    " + line.strip())
    compat_lines.append(line)

# ── Step 9: Save outputs ───────────────────────────────────────────────────────
print("\n[9] Saving outputs...")

# Summary JSON
summary = {
    "total_examples": len(df),
    "european_examples": len(df_eu),
    "columns": list(df.columns),
    "image_shapes": {c: df[c].value_counts().to_dict()
                     for c in shape_cols},
    "source_type_dist": df["source_type"].value_counts().to_dict()
                        if "source_type" in df.columns else {},
    "country_dist": df["country"].value_counts().to_dict()
                    if "country" in df.columns else {},
    "has_binary_mask": has_mask,
    "has_emission_rate": has_rate,
}
with open(OUT_DIR / "mars_s2l_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"    → results_analysis/mars_s2l_summary.json")

# European CSV
if len(df_eu) > 0:
    df_eu.to_csv(OUT_DIR / "mars_s2l_european.csv", index=False)
    print(f"    → results_analysis/mars_s2l_european.csv  ({len(df_eu)} rows)")

# Compatibility report
with open(OUT_DIR / "mars_s2l_compatibility.txt", "w") as f:
    f.write("MARS-S2L × CH4Net Pipeline Compatibility Report\n")
    f.write("=" * 55 + "\n\n")
    f.write(f"Dataset: UNEP-IMEO/MARS-S2L\n")
    f.write(f"Total examples: {len(df)}\n")
    f.write(f"European examples: {len(df_eu)}\n\n")
    f.write("Compatibility:\n")
    for line in compat_lines:
        f.write(line + "\n")
    f.write("\nAction items:\n")
    if len(df_eu) < 50:
        f.write("  - Supplement with TROPOMI mining (run tropomi_mine_europe.py)\n")
    if not has_mask:
        f.write("  - Need to derive binary masks from enhancement maps\n")
    if not has_rate:
        f.write("  - No emission rates — regression training requires TROPOMI labels\n")
    f.write("  - Verify B11/B12 are in correct units (TOA reflectance 0–1)\n")
    f.write("  - If only SWIR bands: need to pair with full S2 L1C download\n")
print(f"    → results_analysis/mars_s2l_compatibility.txt")

print("\n" + "=" * 65)
print("  Exploration complete.")
print(f"  European examples found: {len(df_eu)}")
if len(df_eu) > 0 and "date" in df_eu.columns:
    print(f"  Date range (EU): {df_eu['date'].min()} → {df_eu['date'].max()}")
print("=" * 65)
