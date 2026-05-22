"""
live_pipeline.py — The Deliverable: Copernicus → CH4Net Live Detection

This script is the operationalized pipeline that connects:
  1. Copernicus API (previous group's work, refactored)
  2. SAFE → NumPy preprocessing (the bridge we built)
  3. CH4Net inference (your trained model)
  4. Georeferenced output (detection maps with real-world coordinates)

It transitions the project from a static historical experiment
into a live monitoring tool.

Usage:
  # Monitor Turkmenistan super-emitters (same sites as the paper)
  python -m scripts.live_pipeline \
    --region "POLYGON((53.5 39.2, 54.3 39.2, 54.3 39.6, 53.5 39.6, 53.5 39.2))" \
    --start 2021-01-01 \
    --end 2021-03-31 \
    --weights weights/best_model.pth \
    --output results/

  # Or monitor a European gas facility
  python -m scripts.live_pipeline \
    --region "POLYGON((5.0 51.4, 5.5 51.4, 5.5 51.7, 5.0 51.7, 5.0 51.4))" \
    --start 2025-01-01 \
    --end 2025-01-31 \
    --weights weights/best_model.pth

What happens step by step:
  1. Queries Copernicus OData API for L1C tiles in the region/date range
  2. Filters out cloudy scenes (>30% cloud cover, server-side)
  3. Downloads each SAFE .zip archive
  4. Extracts 12 spectral bands from .jp2 files
  5. Resamples all bands to 10m common grid
  6. Saves geo-metadata as JSON sidecar (CRS + affine transform)
  7. Normalizes to [0, 255] range matching CH4Net training data
  8. Tiles the full scene into 100x100 patches
  9. Runs CH4Net U-Net inference on each patch
  10. Stitches predictions into full-scene probability map
  11. Applies calibrated threshold (0.18) to create binary mask
  12. Re-projects detection mask as georeferenced GeoTIFF
  13. Logs all detected plumes with coordinates and confidence
"""

import os
import sys
import json
import time
import logging
import argparse
import uuid
from datetime import datetime
from typing import Optional

import numpy as np

# Add project root to path so imports work
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.ingestion.copernicus_client import CopernicusClient, SentinelProduct
from src.ingestion.preprocessing import (
    safe_to_npy,
    tile_scene,
    stitch_predictions,
    normalize_to_ch4net_range,
    GeoMetadata,
    save_prediction_geotiff,
    HAS_RASTERIO,
)
from src.detection.ch4net_model import CH4NetDetector, Unet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("live_pipeline")


# ─────────────────────────────────────────────────────────────
# The Live Pipeline Class
# ─────────────────────────────────────────────────────────────

class LivePipeline:
    """
    End-to-end pipeline: Copernicus API → CH4Net Detection → Georeferenced Output.

    This is the operationalized version of what was previously two separate
    Jupyter notebooks (Copernicus_API_Automation + CH4Net_Paper_Implementation).

    Initialize once with credentials and model weights, then call run()
    with any region and date range to get detections.
    """

    def __init__(
        self,
        copernicus_user: str,
        copernicus_password: str,
        weights_path: Optional[str] = None,
        threshold: float = 0.18,
        min_plume_pixels: int = 115,
        max_cloud_cover: float = 30.0,
        download_dir: str = "data/downloads",
        npy_dir: str = "data/npy_cache",
        results_dir: str = "results",
    ):
        """
        Args:
            copernicus_user: Copernicus Data Space email
            copernicus_password: Copernicus Data Space password
            weights_path: Path to trained CH4Net best_model.pth
            threshold: Detection confidence threshold (0.18 = optimized F1)
            min_plume_pixels: Minimum contiguous pixels to count as plume
            max_cloud_cover: Discard scenes above this cloud % (server-side)
            download_dir: Where to store downloaded .zip archives
            npy_dir: Where to cache converted .npy arrays
            results_dir: Where to save detection GeoTIFFs and event logs
        """
        # Initialize Copernicus client (token-cached, secure credentials)
        self.client = CopernicusClient(copernicus_user, copernicus_password)
        self.max_cloud_cover = max_cloud_cover
        self.download_dir = download_dir
        self.npy_dir = npy_dir
        self.results_dir = results_dir
        self.threshold = threshold
        self.min_plume_pixels = min_plume_pixels

        os.makedirs(download_dir, exist_ok=True)
        os.makedirs(npy_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        # Initialize CH4Net detector if weights provided
        self.detector = None
        if weights_path and os.path.exists(weights_path):
            logger.info("Loading CH4Net model from %s", weights_path)
            self.detector = CH4NetDetector(
                weights_path=weights_path,
                threshold=threshold,
                min_plume_pixels=min_plume_pixels,
            )
            logger.info("CH4Net model loaded successfully")
        else:
            logger.warning(
                "No model weights at '%s' — pipeline will run in "
                "download-only mode (no inference). Copy your "
                "best_model.pth to weights/ to enable detection.",
                weights_path,
            )

    def run(
        self,
        wkt_polygon: str,
        start_date: str,
        end_date: str,
        max_products: int = 50,
        l1c_only: bool = True,
    ) -> list[dict]:
        """
        Execute the full pipeline for a region and time window.

        Args:
            wkt_polygon: WKT polygon defining the region of interest
                e.g., "POLYGON((53.5 39.2, 54.3 39.2, 54.3 39.6, 53.5 39.6, 53.5 39.2))"
            start_date: "2021-01-01" format
            end_date: "2021-03-31" format
            max_products: Max tiles to process (for testing, limit this)
            l1c_only: If True, only process L1C products (recommended for
                      methane detection — L2A atmospheric correction can
                      destroy methane signals)

        Returns:
            List of detection event dicts, each containing:
              - event_uuid, timestamp, lat/lon, confidence, plume_area
              - paths to the georeferenced prediction GeoTIFF
        """
        logger.info("=" * 70)
        logger.info("LIVE PIPELINE — Methane Detection Run")
        logger.info("  Region: %s", wkt_polygon[:60] + "...")
        logger.info("  Period: %s to %s", start_date, end_date)
        logger.info("  Cloud filter: ≤%.0f%%", self.max_cloud_cover)
        logger.info("  Model: %s", "loaded" if self.detector else "NOT LOADED")
        logger.info("=" * 70)

        all_events = []

        # ── Step 1: Query Copernicus API ──────────────────────────
        logger.info("[Step 1] Querying Copernicus for Sentinel-2 products...")
        start_iso = f"{start_date}T00:00:00.000Z"
        end_iso = f"{end_date}T23:59:59.999Z"

        products = self.client.search_products(
            wkt_polygon=wkt_polygon,
            start_date=start_iso,
            end_date=end_iso,
            collection="SENTINEL-2",
            max_cloud_cover=self.max_cloud_cover,
        )

        if not products:
            logger.warning("No products found. Check region/dates/cloud cover.")
            return []

        # Filter to L1C only if requested (critical for methane detection)
        if l1c_only:
            before = len(products)
            products = [p for p in products if "L1C" in p.processing_level.upper()]
            logger.info(
                "Filtered to L1C only: %d → %d products", before, len(products)
            )

        # Sort by cloud cover (best first) and limit
        products.sort(key=lambda p: p.cloud_cover or 100)
        products = products[:max_products]

        logger.info("[Step 1] Found %d products to process", len(products))
        for i, p in enumerate(products):
            logger.info(
                "  [%d] %s | %s | cloud: %.1f%% | %s",
                i + 1, p.tile_id, p.acquisition_date[:10],
                p.cloud_cover or -1, p.cloud_quality,
            )

        # ── Step 2-12: Process each product ───────────────────────
        for idx, product in enumerate(products):
            logger.info(
                "\n[Product %d/%d] %s (%s)",
                idx + 1, len(products), product.name, product.tile_id,
            )

            try:
                events = self._process_single_product(product)
                all_events.extend(events)
            except Exception as e:
                logger.error("Failed to process %s: %s", product.name, e)
                continue

            # Rate-limit API calls
            time.sleep(2)

        # ── Summary ───────────────────────────────────────────────
        logger.info("\n" + "=" * 70)
        logger.info("PIPELINE COMPLETE")
        logger.info("  Products processed: %d", len(products))
        logger.info("  Plume events detected: %d", len(all_events))
        logger.info("=" * 70)

        # Save event log
        if all_events:
            log_path = os.path.join(
                self.results_dir,
                f"events_{start_date}_to_{end_date}.json",
            )
            with open(log_path, "w") as f:
                json.dump(all_events, f, indent=2, default=str)
            logger.info("Event log saved to %s", log_path)

        return all_events

    def _process_single_product(self, product: SentinelProduct) -> list[dict]:
        """
        Process a single Sentinel-2 product through the full chain:
        Download → Extract → Preprocess → Detect → Geolocate
        """
        events = []

        # ── Step 2: Download ──────────────────────────────────────
        logger.info("  [Download] Fetching from Copernicus...")
        zip_path = self.client.download_product(product, self.download_dir)
        if not zip_path:
            logger.error("  Download failed, skipping")
            return []

        # ── Step 3-6: Convert SAFE → .npy + geo metadata ─────────
        if HAS_RASTERIO:
            logger.info("  [Preprocess] Extracting bands and resampling to 10m...")
            npy_path, meta_path = safe_to_npy(
                zip_path=zip_path,
                output_dir=self.npy_dir,
                tile_id=product.tile_id,
                acquisition_date=product.acquisition_date,
                satellite=product.satellite,
            )
            # Load the converted data
            scene = np.load(npy_path)
            geo_meta = GeoMetadata.load(meta_path)
        else:
            logger.warning(
                "  [Preprocess] rasterio not installed — cannot extract bands.\n"
                "  Install with: conda install rasterio\n"
                "  Skipping inference for this product."
            )
            return []

        # ── Step 7-8: Tile and run inference ──────────────────────
        if self.detector is None:
            logger.info("  [Detect] Model not loaded — skipping inference")
            logger.info("  Downloaded and preprocessed: %s", npy_path)
            return []

        logger.info("  [Detect] Tiling scene (%dx%d) into 100x100 patches...", 
                     scene.shape[0], scene.shape[1])
        tiles = tile_scene(scene, tile_size=100)

        logger.info("  [Detect] Running CH4Net on %d patches (BATCHED)...", len(tiles))
        # Extract raw arrays from Tile objects
        tile_arrays = [tile.data for tile in tiles]
        # Run inference in batches of 64
        predictions = self.detector.detect_batch(tile_arrays, batch_size=64)

        # ── Step 9-10: Stitch and threshold ───────────────────────
        logger.info("  [Stitch] Assembling full-scene prediction map...")
        full_prob_map = stitch_predictions(
            tiles, predictions, scene.shape[0], scene.shape[1]
        )
        binary_mask = (full_prob_map >= self.threshold).astype(np.uint8)
        total_plume_pixels = int(binary_mask.sum())

        logger.info(
            "  [Result] Total plume pixels: %d (threshold: %.2f)",
            total_plume_pixels, self.threshold,
        )

        # ── Step 11: Save georeferenced GeoTIFF ───────────────────
        geotiff_name = f"detection_{product.tile_id}_{product.acquisition_date[:10]}.tif"
        geotiff_path = os.path.join(self.results_dir, geotiff_name)
        save_prediction_geotiff(full_prob_map, geo_meta, geotiff_path)

        
        # ── Step 12: Extract per-blob events using connected components ──
        from rasterio.transform import Affine
        from pyproj import Transformer
        from scipy import ndimage as ndi

        transform = Affine(*geo_meta.transform[:6])
        transformer = Transformer.from_crs(geo_meta.crs, "EPSG:4326", always_xy=True)

        labeled, n_blobs = ndi.label(binary_mask)
        blob_sizes = np.bincount(labeled.ravel())[1:]
        significant = [(i + 1, int(sz)) for i, sz in enumerate(blob_sizes)
                       if sz >= self.min_plume_pixels]

        if significant:
            logger.info("  Found %d significant blobs (>= %d px)",
                        len(significant), self.min_plume_pixels)
            for blob_label, blob_size in significant:
                blob_mask = labeled == blob_label
                rows, cols = np.where(blob_mask)
                centroid_row = int(rows.mean())
                centroid_col = int(cols.mean())
                max_conf = float(full_prob_map[blob_mask].max())
                mean_conf = float(full_prob_map[blob_mask].mean())

                x_utm, y_utm = transform * (centroid_col, centroid_row)
                lon, lat = transformer.transform(x_utm, y_utm)

                event = {
                    "event_uuid": f"evt-{uuid.uuid4().hex[:12]}",
                    "timestamp_utc": product.acquisition_date,
                    "latitude": round(lat, 6),
                    "longitude": round(lon, 6),
                    "model_confidence": round(max_conf, 4),
                    "mean_confidence": round(mean_conf, 4),
                    "plume_area_pixels": blob_size,
                    "tile_id": product.tile_id,
                    "satellite": product.satellite,
                    "cloud_cover_pct": product.cloud_cover,
                    "geotiff_path": geotiff_path,
                    "source_product": product.name,
                }
                events.append(event)
                logger.info(
                    "  *** PLUME DETECTED ***  lat=%.4f, lon=%.4f, "
                    "max_conf=%.2f, mean_conf=%.2f, pixels=%d",
                    lat, lon, max_conf, mean_conf, blob_size,
                )
        else:
            logger.info("  No significant blobs (>= %d px) detected",
                        self.min_plume_pixels)
        return events


# ─────────────────────────────────────────────────────────────
# Command-Line Interface
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Methane Detection Live Pipeline: Copernicus → CH4Net",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Turkmenistan super-emitters (paper validation sites)
  python -m scripts.live_pipeline \\
    --region "POLYGON((53.5 39.2, 54.3 39.2, 54.3 39.6, 53.5 39.6, 53.5 39.2))" \\
    --start 2021-01-01 --end 2021-03-31

  # European gas facility
  python -m scripts.live_pipeline \\
    --region "POLYGON((5.0 51.4, 5.5 51.4, 5.5 51.7, 5.0 51.7, 5.0 51.4))" \\
    --start 2025-01-01 --end 2025-01-31
        """,
    )

    parser.add_argument(
        "--region", required=True,
        help="WKT POLYGON defining the monitoring region",
    )
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--weights", default="weights/best_model.pth",
        help="Path to CH4Net model weights (default: weights/best_model.pth)",
    )
    parser.add_argument(
        "--max-products", type=int, default=10,
        help="Max number of products to process (default: 10)",
    )
    parser.add_argument(
        "--max-cloud", type=float, default=30.0,
        help="Max cloud cover %% (default: 30)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.18,
        help="Detection confidence threshold (default: 0.18)",
    )
    parser.add_argument(
        "--output", default="results",
        help="Output directory for detection GeoTIFFs (default: results/)",
    )

    args = parser.parse_args()

    # Load credentials from environment
    copernicus_user = os.environ.get("COPERNICUS_USER")
    copernicus_password = os.environ.get("COPERNICUS_PASSWORD")

    if not copernicus_user or not copernicus_password:
        # Try loading from .env file
        env_path = os.path.join(PROJECT_ROOT, "config", ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, val = line.split("=", 1)
                        os.environ[key.strip()] = val.strip()
            copernicus_user = os.environ.get("COPERNICUS_USER")
            copernicus_password = os.environ.get("COPERNICUS_PASSWORD")

    if not copernicus_user or not copernicus_password:
        print("ERROR: Set COPERNICUS_USER and COPERNICUS_PASSWORD in config/.env")
        print("  cp config/.env.example config/.env")
        print("  # Then edit config/.env with your credentials")
        sys.exit(1)

    # Run the pipeline
    pipeline = LivePipeline(
        copernicus_user=copernicus_user,
        copernicus_password=copernicus_password,
        weights_path=args.weights,
        threshold=args.threshold,
        max_cloud_cover=args.max_cloud,
        results_dir=args.output,
    )

    events = pipeline.run(
        wkt_polygon=args.region,
        start_date=args.start,
        end_date=args.end,
        max_products=args.max_products,
    )

    # Print summary
    print(f"\n{'='*60}")
    print(f"Detection complete: {len(events)} plume events found")
    if events:
        print(f"\nEvents:")
        for e in events:
            print(f"  {e['timestamp_utc'][:10]} | "
                  f"({e['latitude']:.4f}, {e['longitude']:.4f}) | "
                  f"conf: {e['model_confidence']:.2f} | "
                  f"pixels: {e['plume_area_pixels']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
