"""
fix_weisweiler_reference.py
============================
The existing T31UGS winter reference tile (T31UGS_ref_20231206.npy) was
downloaded from orbit R108, which is zero-padded at lon=6.322.

This script:
  1. Removes the bad reference tile
  2. Searches Copernicus for T31UGS winter acquisitions from orbit R008
     (the same orbit as the September 2024 target tile)
  3. Downloads the best clear-sky R008 acquisition from Nov 2023 – Feb 2024
  4. Converts it to .npy and saves as a replacement reference tile

After this runs, re-execute:
    python apply_bitemporal_diff.py --weights weights/european_model.pth \\
        --output-dir results_bitemporal_eu3_fixed

Usage:
    export COPERNICUS_USER=your@email.com
    export COPERNICUS_PASS='yourpassword'
    conda activate methane
    python fix_weisweiler_reference.py
"""

import os
import sys
import json
import getpass
import logging
import tempfile
import shutil
from pathlib import Path

import numpy as np

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))
from src.ingestion.copernicus_client import CopernicusClient
from src.ingestion.preprocessing import safe_to_npy

# ── Config ─────────────────────────────────────────────────────────────────────
TILE_ID        = "T31UGS"
ORBIT          = "R008"         # must match target tile orbit
WEISWEILER_LAT = 50.837
WEISWEILER_LON = 6.322
NPY_CACHE      = Path("data/npy_cache")
DOWNLOAD_DIR   = Path("data/downloads/reference")
REF_DATE_START = "2023-11-01T00:00:00.000Z"
REF_DATE_END   = "2024-02-28T23:59:59.000Z"
MAX_CLOUD      = 30.0

B11_IDX = 10
B12_IDX = 11

DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ── Build WKT around Weisweiler ────────────────────────────────────────────────
def site_bbox_wkt(lat, lon, margin=0.3):
    return (
        f"POLYGON(({lon-margin} {lat-margin},{lon+margin} {lat-margin},"
        f"{lon+margin} {lat+margin},{lon-margin} {lat+margin},{lon-margin} {lat-margin}))"
    )


# ── Verify pixel after download ────────────────────────────────────────────────
def verify_pixel(npy_path: Path, geo_path: Path, lat: float, lon: float) -> bool:
    """Return True if the Weisweiler pixel is non-zero in the downloaded tile."""
    try:
        import rasterio.warp
        import rasterio.transform
        from rasterio.transform import Affine

        with open(geo_path) as f:
            geo = json.load(f)

        a, b, c, d, e, ff = geo["transform"]
        aff = Affine(a, b, c, d, e, ff)
        xs, ys = rasterio.warp.transform("EPSG:4326", geo["crs"], [lon], [lat])
        row, col = rasterio.transform.rowcol(aff, xs[0], ys[0])
        row, col = int(row), int(col)

        arr = np.load(npy_path, mmap_mode="r")
        H, W, _ = arr.shape

        if not (0 <= row < H and 0 <= col < W):
            log.error("  Pixel (%d, %d) out of bounds (%d × %d)", row, col, H, W)
            return False

        b12 = int(arr[row, col, B12_IDX])
        b11 = int(arr[row, col, B11_IDX])
        log.info("  Pixel (%d, %d):  B11=%d  B12=%d  B12/B11=%.3f",
                 row, col, b11, b12, b12 / max(b11, 1))

        if b12 == 0 and b11 == 0:
            log.warning("  ⚠️  Pixel is zero — this orbit may not cover Weisweiler")
            return False

        log.info("  ✅  Pixel is valid (non-zero)")
        return True

    except Exception as e:
        log.warning("  Pixel verification failed: %s", e)
        return False


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 65)
    log.info("  Fix Weisweiler Reference Tile — force orbit %s", ORBIT)
    log.info("=" * 65)

    # ── Step 1: Remove bad reference tile ─────────────────────────────────────
    bad_tiles = list(NPY_CACHE.glob(f"{TILE_ID}_ref_*.npy"))
    if bad_tiles:
        for bt in bad_tiles:
            log.info("Removing bad reference tile: %s", bt.name)
            bt.unlink()
            geo = bt.with_name(bt.stem + "_geo.json")
            if geo.exists():
                geo.unlink()
                log.info("Removed sidecar: %s", geo.name)
    else:
        log.info("No existing reference tiles to remove.")

    # ── Step 2: Credentials ────────────────────────────────────────────────────
    user = os.environ.get("COPERNICUS_USER") or input("Copernicus username: ").strip()
    pw   = os.environ.get("COPERNICUS_PASS") or getpass.getpass("Copernicus password: ")

    client = CopernicusClient(user, pw)

    # ── Step 3: Search for T31UGS winter products ──────────────────────────────
    wkt = site_bbox_wkt(WEISWEILER_LAT, WEISWEILER_LON)
    log.info("Searching %s – %s for tile %s ...", REF_DATE_START[:10], REF_DATE_END[:10], TILE_ID)

    products = client.search_products(
        wkt_polygon=wkt,
        start_date=REF_DATE_START,
        end_date=REF_DATE_END,
        collection="SENTINEL-2",
        max_cloud_cover=MAX_CLOUD,
        max_results=100,
    )

    tile_products = [p for p in products if p.tile_id == TILE_ID]
    log.info("Found %d products for tile %s", len(tile_products), TILE_ID)

    if not tile_products:
        log.warning("No products with cloud ≤ %.0f%%. Relaxing to 60%%...", MAX_CLOUD)
        products = client.search_products(
            wkt_polygon=wkt,
            start_date=REF_DATE_START,
            end_date=REF_DATE_END,
            collection="SENTINEL-2",
            max_cloud_cover=60.0,
            max_results=100,
        )
        tile_products = [p for p in products if p.tile_id == TILE_ID]
        log.info("Relaxed search: %d products for tile %s", len(tile_products), TILE_ID)

    # ── Step 4: Filter to orbit R008 AND L1C (not L2A) ────────────────────────
    r008_products = [
        p for p in tile_products
        if f"_{ORBIT}_" in p.name and "MSIL1C" in p.name
    ]
    log.info("  %d products are orbit %s L1C", len(r008_products), ORBIT)

    # Log all candidates so user can see what's available
    for p in sorted(r008_products, key=lambda x: (x.cloud_cover or 99, x.acquisition_date)):
        log.info("    %s  cloud=%.1f%%", p.name[:70], p.cloud_cover or -1)

    if not r008_products:
        log.error("No %s products found for %s in winter 2023/2024.", ORBIT, TILE_ID)
        log.error("Available orbits in results:")
        from collections import Counter
        orbits = Counter()
        for p in tile_products:
            parts = p.name.split("_")
            if len(parts) > 4:
                orbits[parts[4]] += 1
        for orbit, count in orbits.most_common():
            log.error("  %s: %d products", orbit, count)
        log.error("Check whether R008 has winter acquisitions for T31UGS.")
        sys.exit(1)

    # ── Step 5: Sort by cloud cover, then prefer Dec/Jan ──────────────────────
    def sort_key(p):
        cloud = p.cloud_cover if p.cloud_cover is not None else 99.0
        try:
            month = int(p.acquisition_date[5:7])
            month_score = {12: 0, 1: 1, 2: 2, 11: 3}.get(month, 5)
        except (ValueError, IndexError):
            month_score = 5
        return (cloud, month_score)

    r008_products.sort(key=sort_key)

    # ── Step 6: Try candidates until we get valid pixel coverage ──────────────
    downloaded_ref = None
    for rank, product in enumerate(r008_products[:5]):  # try top 5 candidates
        log.info("")
        log.info("Candidate %d/%d: %s", rank + 1, min(5, len(r008_products)), product.name[:70])
        log.info("  Date: %s   Cloud: %.1f%%",
                 product.acquisition_date[:10], product.cloud_cover or -1)

        # Download
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = client.download_product(product, tmpdir)
            if zip_path is None:
                log.warning("  Download failed — trying next candidate")
                continue

            # Convert SAFE → npy
            date_str = product.acquisition_date[:10].replace("-", "")
            ref_stem = f"{TILE_ID}_ref_{date_str}"
            out_npy  = NPY_CACHE / f"{ref_stem}.npy"
            out_geo  = NPY_CACHE / f"{ref_stem}_geo.json"

            log.info("  Converting SAFE archive → %s ...", out_npy.name)
            extract_tmp = tempfile.mkdtemp(prefix=f"ref_{TILE_ID}_")
            try:
                raw_npy, raw_geo = safe_to_npy(
                    zip_path=zip_path,
                    output_dir=str(NPY_CACHE),
                    tile_id=TILE_ID,
                    acquisition_date=product.acquisition_date[:10],
                    satellite=product.satellite,
                    extract_dir=extract_tmp,
                )
            except Exception as e:
                log.warning("  Conversion failed: %s — trying next candidate", e)
                shutil.rmtree(extract_tmp, ignore_errors=True)
                continue
            finally:
                shutil.rmtree(extract_tmp, ignore_errors=True)

            # Rename to reference tile convention: {TILE_ID}_ref_YYYYMMDD.npy
            actual_npy = Path(raw_npy)
            actual_geo = Path(raw_geo)
            if actual_npy != out_npy:
                actual_npy.rename(out_npy)
                actual_npy = out_npy
            if actual_geo != out_geo:
                actual_geo.rename(out_geo)
                actual_geo = out_geo

            # Verify the Weisweiler pixel is non-zero
            log.info("  Verifying Weisweiler pixel coverage ...")
            if verify_pixel(actual_npy, actual_geo, WEISWEILER_LAT, WEISWEILER_LON):
                downloaded_ref = actual_npy
                log.info("  ✅  Valid reference tile saved: %s", actual_npy.name)
                break
            else:
                log.warning("  Zero pixel — removing and trying next candidate")
                actual_npy.unlink(missing_ok=True)
                if actual_geo:
                    actual_geo.unlink(missing_ok=True)

    # ── Step 7: Summary ────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 65)
    if downloaded_ref:
        log.info("SUCCESS: New reference tile ready.")
        log.info("  %s", downloaded_ref.name)
        log.info("")
        log.info("Now re-run the bitemporal evaluation:")
        log.info("  python apply_bitemporal_diff.py \\")
        log.info("    --weights weights/european_model.pth \\")
        log.info("    --output-dir results_bitemporal_eu3_fixed")
        log.info("")
        log.info("And for the base model comparison:")
        log.info("  python apply_bitemporal_diff.py \\")
        log.info("    --weights weights/best_model.pth \\")
        log.info("    --output-dir results_bitemporal_base3_fixed")
    else:
        log.error("FAILED: No valid R008 reference tile found after trying 5 candidates.")
        log.error("")
        log.error("Options:")
        log.error("  1. Widen the search window:  edit REF_DATE_START/REF_DATE_END in this script")
        log.error("  2. Use a summer reference instead (avoid seasonal vegetation change)")
        log.error("  3. Skip BT differencing for Weisweiler — baseline S/C=4.004 already detects it")
        log.error("     Add 'skip_bitemporal': True to Weisweiler's entry in SITES (apply_bitemporal_diff.py)")
    log.info("=" * 65)


if __name__ == "__main__":
    main()
