"""
preprocessing.py — The Bridge Between Copernicus and CH4Net

This module solves the core compatibility problem documented in the
integration report: Copernicus delivers .zip SAFE archives containing
.jp2 band files, but CH4Net expects .npy arrays of shape (H, W, 12).

Pipeline:
  .zip SAFE archive
    → unzip → find .jp2 band files in GRANULE/*/IMG_DATA/
    → read each band with rasterio
    → resample 20m/60m bands to 10m resolution
    → stack into (H, W, 12) numpy array
    → save geo-metadata as sidecar JSON (for re-projection after inference)
    → tile into 100x100 patches for CH4Net
    → after inference, stitch patches back and re-project to GeoTIFF

CRITICAL DESIGN DECISION (from integration report):
  CH4Net MUST run on raw L1C data, NOT L2A or EMRDM-denoised data.
  Atmospheric correction in L2A can erase methane absorption signals.
  The EMRDM cloud removal model used by the previous group for flood
  analysis would destroy atmospheric methane plumes entirely.
  This module explicitly skips any denoising/atmospheric correction.

Band mapping:
  CH4Net .npy files contain 12 bands (B10/cirrus excluded):
  Index:  0    1    2    3    4    5    6    7    8    9   10   11
  Band:  B01  B02  B03  B04  B05  B06  B07  B08  B8A  B09  B11  B12

  B11 (index 10) = 1610nm SWIR reference (methane-transparent)
  B12 (index 11) = 2190nm SWIR absorption (methane-sensitive)
"""

import os
import re
import sys
import glob
import json
import zipfile
import logging
import tempfile
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Try to import rasterio — this is required for the geo bridge
try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import calculate_default_transform, reproject
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    logger.warning(
        "rasterio not installed. Install with: conda install rasterio\n"
        "Without rasterio, only pre-converted .npy files can be used."
    )


# ─────────────────────────────────────────────────────────────
# Sentinel-2 SAFE Archive Structure
# ─────────────────────────────────────────────────────────────

# The 12 bands CH4Net expects, in order, with their native resolutions
BAND_CONFIG = [
    {"name": "B01", "resolution": 60},
    {"name": "B02", "resolution": 10},
    {"name": "B03", "resolution": 10},
    {"name": "B04", "resolution": 10},
    {"name": "B05", "resolution": 20},
    {"name": "B06", "resolution": 20},
    {"name": "B07", "resolution": 20},
    {"name": "B08", "resolution": 10},
    {"name": "B8A", "resolution": 20},
    {"name": "B09", "resolution": 60},
    # B10 (cirrus) excluded — not in CH4Net training data
    {"name": "B11", "resolution": 20},  # ← SWIR reference
    {"name": "B12", "resolution": 20},  # ← Methane absorption
]

BAND_NAMES = [b["name"] for b in BAND_CONFIG]


@dataclass
class GeoMetadata:
    """
    Geospatial metadata stripped during .npy conversion.

    Saved as a JSON sidecar file alongside each .npy array.
    After CH4Net inference, this metadata is used to re-project
    the detection mask back into a georeferenced GeoTIFF that
    can be overlaid on maps or fed into economic damage models.
    """
    crs: str                    # e.g., "EPSG:32640"
    transform: list[float]      # 6-element affine transform
    width: int
    height: int
    tile_id: str
    acquisition_date: str
    satellite: str
    source_product: str         # original SAFE product name

    def save(self, path: str):
        """Save metadata as JSON sidecar file."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "GeoMetadata":
        """Load metadata from JSON sidecar file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)


# ─────────────────────────────────────────────────────────────
# SAFE Archive Extraction
# ─────────────────────────────────────────────────────────────

def unzip_safe_archive(zip_path: str, extract_dir: Optional[str] = None) -> str:
    """
    Unzip a Copernicus SAFE archive and return the .SAFE directory path.

    SAFE structure:
      S2A_MSIL1C_20210315T*.SAFE/
        GRANULE/
          L1C_T40SBH_*/
            IMG_DATA/
              T40SBH_20210315T*_B01.jp2
              T40SBH_20210315T*_B02.jp2
              ... (one .jp2 per band)
    """
    if extract_dir is None:
        extract_dir = os.path.dirname(zip_path)

    logger.info("Unzipping %s ...", os.path.basename(zip_path))
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    # Find the .SAFE directory
    safe_dirs = glob.glob(os.path.join(extract_dir, "*.SAFE"))
    if not safe_dirs:
        # Sometimes nested inside another folder
        safe_dirs = glob.glob(os.path.join(extract_dir, "*", "*.SAFE"))

    if not safe_dirs:
        raise FileNotFoundError(f"No .SAFE directory found after unzipping {zip_path}")

    safe_dir = safe_dirs[0]
    logger.info("Extracted: %s", os.path.basename(safe_dir))
    return safe_dir


def find_band_files(safe_dir: str) -> dict[str, str]:
    """
    Locate the .jp2 file for each of the 12 bands inside a SAFE archive.

    Returns dict mapping band name → file path, e.g.:
      {"B01": "/path/to/T40SBH_..._B01.jp2", "B02": "...", ...}

    Note: We look in GRANULE/*/IMG_DATA/ for L1C products.
    L2A products have a different structure (IMG_DATA/R10m/, R20m/, R60m/)
    but we prefer L1C for methane detection (raw TOA reflectance).
    """
    # Search for .jp2 files in the SAFE directory
    jp2_files = glob.glob(os.path.join(safe_dir, "GRANULE", "*", "IMG_DATA", "*.jp2"))

    if not jp2_files:
        # Try L2A structure
        jp2_files = glob.glob(
            os.path.join(safe_dir, "GRANULE", "*", "IMG_DATA", "R*m", "*.jp2")
        )

    if not jp2_files:
        raise FileNotFoundError(f"No .jp2 band files found in {safe_dir}")

    # Map band name → file path
    band_files = {}
    for fp in jp2_files:
        fname = os.path.basename(fp)
        # Extract band name from filename like T40SBH_20210315T064629_B02.jp2
        # or S2A_MSIL1C_..._B02.jp2
        for band_name in BAND_NAMES:
            # Match _B02. or _B8A. patterns
            if f"_{band_name}." in fname or f"_{band_name}_" in fname:
                band_files[band_name] = fp
                break

    found = set(band_files.keys())
    expected = set(BAND_NAMES)
    missing = expected - found

    if missing:
        logger.warning("Missing bands: %s", missing)
        # B09 and B01 sometimes missing in some products — not critical
        # B11 and B12 are MANDATORY for methane detection
        if "B11" in missing or "B12" in missing:
            raise FileNotFoundError(
                f"CRITICAL: Methane bands B11/B12 missing from {safe_dir}"
            )

    logger.info("Found %d/%d bands in SAFE archive", len(band_files), len(BAND_NAMES))
    return band_files


# ─────────────────────────────────────────────────────────────
# Band Reading and Resampling
# ─────────────────────────────────────────────────────────────

def read_and_resample_bands(
    band_files: dict[str, str],
    target_resolution: float = 10.0,
) -> tuple[np.ndarray, dict]:
    """
    Read all bands from .jp2 files and resample to a common 10m grid.

    This is the core transformation step. Sentinel-2 bands have different
    native resolutions:
      10m: B02, B03, B04, B08
      20m: B05, B06, B07, B8A, B11, B12
      60m: B01, B09

    CH4Net was trained on all bands resampled to 10m.
    We use bilinear resampling (matching the paper).

    Memory-efficient implementation: bands are read, normalized, and written
    to the output array one at a time. Peak memory is ~one float32 band
    (~250MB) rather than all 12 bands simultaneously (~8GB).

    Returns:
      - stacked array of shape (H, W, 12) in uint8 [0, 255] range
        (already normalized — skips the separate normalize_to_ch4net_range call)
      - metadata dict with CRS, transform, dimensions from the 10m reference
    """
    import gc

    if not HAS_RASTERIO:
        raise RuntimeError("rasterio required for band reading. Install with: conda install rasterio")

    # Use a 10m band as the reference grid
    ref_band = "B02"  # Always 10m native
    if ref_band not in band_files:
        # Fallback to any 10m band
        for b in ["B03", "B04", "B08"]:
            if b in band_files:
                ref_band = b
                break

    with rasterio.open(band_files[ref_band]) as ref_src:
        ref_height = ref_src.height
        ref_width = ref_src.width
        ref_transform = ref_src.transform
        ref_crs = ref_src.crs
        ref_profile = ref_src.profile.copy()

    logger.info(
        "Reference grid: %dx%d pixels at %.0fm resolution (CRS: %s)",
        ref_width, ref_height, target_resolution, ref_crs,
    )

    # Pre-allocate the final uint8 output — one allocation, no accumulation
    # Shape: (H, W, 12) uint8 ≈ ~120MB for a full 10980×10980 scene
    n_bands = len(BAND_CONFIG)
    out = np.zeros((ref_height, ref_width, n_bands), dtype=np.uint8)

    # Temporary float32 buffer reused for each band (avoids re-allocation per band)
    band_buf = np.empty((ref_height, ref_width), dtype=np.float32)

    # Read, resample, normalize, and write each band directly into out[:, :, i]
    for i, band_info in enumerate(BAND_CONFIG):
        band_name = band_info["name"]

        if band_name not in band_files:
            # Missing band — fill with zeros (only B01/B09 might be missing)
            logger.warning("Band %s missing, filling with zeros", band_name)
            out[:, :, i] = 0
            continue

        with rasterio.open(band_files[band_name]) as src:
            if src.height == ref_height and src.width == ref_width:
                # Same resolution — read directly into buffer
                src.read(1, out=band_buf)
            else:
                # Different resolution — resample to 10m grid
                reproject(
                    source=rasterio.band(src, 1),
                    destination=band_buf,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    dst_resolution=(target_resolution, target_resolution),
                    resampling=Resampling.bilinear,
                )
                logger.debug(
                    "Resampled %s from %dx%d to %dx%d",
                    band_name, src.height, src.width, ref_height, ref_width,
                )

        # Normalize in-place: L1C reflectance [0, 10000] → uint8 [0, 255]
        # Equivalent to normalize_to_ch4net_range but without an extra copy
        np.clip(band_buf * (255.0 / 10000.0), 0, 255, out=band_buf)
        out[:, :, i] = band_buf  # uint8 cast happens here

        logger.debug("Band %s (%d/%d) written", band_name, i + 1, n_bands)

        # Explicitly free any rasterio-internal buffers and prompt GC
        gc.collect()

    del band_buf
    gc.collect()

    metadata = {
        "crs": str(ref_crs),
        "transform": list(ref_transform)[:6],
        "width": ref_width,
        "height": ref_height,
    }

    logger.info("Stacked array shape: %s, dtype: %s", out.shape, out.dtype)
    return out, metadata


# ─────────────────────────────────────────────────────────────
# Normalization (matching CH4Net training data)
# ─────────────────────────────────────────────────────────────

def normalize_to_ch4net_range(array: np.ndarray) -> np.ndarray:
    """
    Normalize Sentinel-2 reflectance values to [0, 255] range.

    CH4Net training data (.npy files from Hugging Face) are stored as
    uint8-like values in [0, 255]. The CH4NetDataset class then divides
    by 255.0 to get [0, 1] floats.

    Raw Sentinel-2 L1C reflectance values are typically in [0, 10000]
    (scaled by 10000 from the actual 0-1 reflectance).

    We scale: pixel_value / 10000 * 255 → clipped to [0, 255]
    """
    # Sentinel-2 L1C: reflectance * 10000, so divide to get [0, 1], then * 255
    normalized = (array / 10000.0) * 255.0
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    return normalized


# ─────────────────────────────────────────────────────────────
# Tiling: Split large scene into 100x100 patches for CH4Net
# ─────────────────────────────────────────────────────────────

@dataclass
class Tile:
    """A 100x100 patch extracted from a larger scene."""
    data: np.ndarray          # (100, 100, 12)
    row_start: int            # position in full scene
    col_start: int
    row_end: int
    col_end: int


def tile_scene(
    scene: np.ndarray,
    tile_size: int = 100,
    overlap: int = 0,
) -> list[Tile]:
    """
    Split a full Sentinel-2 scene into 100x100 patches for CH4Net.

    A full Sentinel-2 tile is 10980x10980 pixels at 10m resolution.
    CH4Net expects 100x100 input patches.

    Args:
        scene: array of shape (H, W, 12)
        tile_size: patch size (100 for CH4Net)
        overlap: pixel overlap between adjacent patches (0 for now,
                 could add overlap for better edge detection later)

    Returns:
        List of Tile objects, each containing a (100, 100, 12) array
        and its position in the original scene.
    """
    H, W, C = scene.shape
    step = tile_size - overlap
    tiles = []

    for row in range(0, H - tile_size + 1, step):
        for col in range(0, W - tile_size + 1, step):
            patch = scene[row:row + tile_size, col:col + tile_size, :]
            tiles.append(Tile(
                data=patch,
                row_start=row,
                col_start=col,
                row_end=row + tile_size,
                col_end=col + tile_size,
            ))

    logger.info(
        "Tiled %dx%d scene into %d patches of %dx%d",
        H, W, len(tiles), tile_size, tile_size,
    )
    return tiles


def stitch_predictions(
    tiles: list[Tile],
    predictions: list[np.ndarray],
    scene_height: int,
    scene_width: int,
) -> np.ndarray:
    """
    Stitch tile-level predictions back into a full-scene probability map.

    Args:
        tiles: list of Tile objects (positions in original scene)
        predictions: list of (100, 100) probability maps from CH4Net
        scene_height, scene_width: dimensions of the original scene

    Returns:
        Full-scene probability map of shape (H, W)
    """
    full_map = np.zeros((scene_height, scene_width), dtype=np.float32)
    count_map = np.zeros((scene_height, scene_width), dtype=np.float32)

    for tile, pred in zip(tiles, predictions):
        full_map[tile.row_start:tile.row_end, tile.col_start:tile.col_end] += pred
        count_map[tile.row_start:tile.row_end, tile.col_start:tile.col_end] += 1

    # Average overlapping regions
    count_map[count_map == 0] = 1  # avoid division by zero
    full_map /= count_map

    return full_map


# ─────────────────────────────────────────────────────────────
# GeoTIFF Re-projection (reverse bridge)
# ─────────────────────────────────────────────────────────────

def save_prediction_geotiff(
    prediction: np.ndarray,
    geo_metadata: GeoMetadata,
    output_path: str,
):
    """
    Write a CH4Net prediction mask back as a georeferenced GeoTIFF.

    This is the reverse bridge — takes the raw numpy prediction and
    re-attaches the CRS and affine transform that were stripped during
    the .npy conversion. The resulting GeoTIFF can be opened in QGIS,
    overlaid on maps, or fed into the DamageScanner/economic models.
    """
    if not HAS_RASTERIO:
        raise RuntimeError("rasterio required for GeoTIFF output")

    from rasterio.transform import Affine

    transform = Affine(*geo_metadata.transform[:6])

    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": prediction.shape[1],
        "height": prediction.shape[0],
        "count": 1,
        "crs": geo_metadata.crs,
        "transform": transform,
        "compress": "lzw",
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(prediction.astype(np.float32), 1)

    logger.info("Saved georeferenced prediction to %s", output_path)


# ─────────────────────────────────────────────────────────────
# Complete: SAFE archive → CH4Net-ready .npy + metadata
# ─────────────────────────────────────────────────────────────

def safe_to_npy(
    zip_path: str,
    output_dir: str,
    tile_id: str = "unknown",
    acquisition_date: str = "unknown",
    satellite: str = "unknown",
    extract_dir: Optional[str] = None,
) -> tuple[str, str]:
    """
    Complete conversion: .zip SAFE archive → .npy array + geo sidecar JSON.

    This is the main function you call to bridge Copernicus → CH4Net.

    Args:
        zip_path: path to the downloaded .zip SAFE archive
        output_dir: where to save the .npy and .json files
        tile_id: Sentinel-2 tile ID (e.g., T40SBH)
        acquisition_date: ISO date string
        satellite: S2A or S2B

    Returns:
        Tuple of (npy_path, json_metadata_path)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Unzip
    if extract_dir is None:
        extract_dir = tempfile.mkdtemp(prefix="s2_extract_")
    safe_dir = unzip_safe_archive(zip_path, extract_dir)

    # Step 2: Find band files
    band_files = find_band_files(safe_dir)

    # Step 3: Read, resample to 10m grid, and normalize to [0, 255] uint8
    # (read_and_resample_bands now returns uint8 directly — no separate normalize step)
    normalized, metadata = read_and_resample_bands(band_files)

    # Step 5: Save .npy
    product_name = os.path.basename(safe_dir).replace(".SAFE", "")
    npy_path = os.path.join(output_dir, f"{product_name}.npy")
    np.save(npy_path, normalized)
    logger.info("Saved .npy array: %s (shape: %s)", npy_path, normalized.shape)

    # Step 6: Save geo metadata sidecar
    geo_meta = GeoMetadata(
        crs=metadata["crs"],
        transform=metadata["transform"],
        width=metadata["width"],
        height=metadata["height"],
        tile_id=tile_id,
        acquisition_date=acquisition_date,
        satellite=satellite,
        source_product=product_name,
    )
    meta_path = os.path.join(output_dir, f"{product_name}_geo.json")
    geo_meta.save(meta_path)
    logger.info("Saved geo metadata: %s", meta_path)

    return npy_path, meta_path
