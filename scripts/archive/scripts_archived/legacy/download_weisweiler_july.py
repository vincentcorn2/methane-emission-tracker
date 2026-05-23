"""
scripts/download_weisweiler_july.py
=====================================
Download Sentinel-2 L1C T31UGS scenes from July 2024 for Weisweiler.

Motivation:
  The Sep-18 CEMF estimate (2724 kg/h) was retracted because the binary plume
  mask is contaminated by terrain heterogeneity (CV=1.607, driven by a South
  directional control that is 43× larger than N/E/W).  The Rhine / Aachen
  agricultural patchwork has high spectral variance when crops are in various
  growth stages (June–August: mixed cereals, maize, beet).

  In late June / July the winter wheat and barley harvest completes and fields
  become spectrally uniform stubble — the primary driver of the terrain CV drops.
  If a July T31UGS scene has CV < 0.5, CEMF becomes viable.

  This script downloads up to MAX_SCENES July 2024 L1C scenes sorted by cloud
  cover.  After running, use apply_bitemporal_diff.py to check each date:

    python apply_bitemporal_diff.py --sites weisweiler \\
           --weights weights/european_model_v8.pth

  Look for: cv_ctrl < 0.5 AND sc_ratio > 1.15 → run CEMF on that date.

Output:
  data/npy_cache/<product_name>.npy
  data/npy_cache/<product_name>_geo.json
  results_analysis/weisweiler_july_manifest.json

Usage:
  conda activate methane
  export COPERNICUS_USER=your@email.com
  export COPERNICUS_PASS='yourpassword'

  python scripts/download_weisweiler_july.py           # download up to 3 scenes
  python scripts/download_weisweiler_july.py --dry-run # search only, no download
  python scripts/download_weisweiler_july.py --max-scenes 5
  python scripts/download_weisweiler_july.py --max-cloud 25  # relax cloud limit

Estimated time: ~40 min per tile (download + rasterio conversion)
Use 'caffeinate -i python ...' to prevent sleep.
"""

import os
import sys
import json
import getpass
import logging
import tempfile
import shutil
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

Path("results_analysis").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("results_analysis/weisweiler_july_download.log"),
    ],
)
log = logging.getLogger(__name__)

from src.ingestion.copernicus_client import CopernicusClient
from src.ingestion.preprocessing import safe_to_npy

# ── Constants ─────────────────────────────────────────────────────────────────
SITE_NAME   = "weisweiler"
SITE_LAT    = 50.837
SITE_LON    = 6.322
TILE_ID     = "T31UGS"

# July 2024 only — post-harvest window for minimum terrain CV
SEARCH_START = "2024-07-01T00:00:00.000Z"
SEARCH_END   = "2024-07-31T23:59:59.000Z"

DEFAULT_MAX_CLOUD  = 15.0   # % — relax to 25 if nothing found at this threshold
FALLBACK_MAX_CLOUD = 25.0
DEFAULT_MAX_SCENES = 3

NPY_CACHE    = Path("data/npy_cache")
DOWNLOAD_DIR = Path("data/downloads/weisweiler_july")
MANIFEST     = Path("results_analysis/weisweiler_july_manifest.json")


def site_bbox_wkt(margin: float = 0.3) -> str:
    lat, lon = SITE_LAT, SITE_LON
    return (
        f"POLYGON(({lon-margin} {lat-margin},{lon+margin} {lat-margin},"
        f"{lon+margin} {lat+margin},{lon-margin} {lat+margin},"
        f"{lon-margin} {lat-margin}))"
    )


def find_cached() -> list:
    """Return T31UGS L1C .npy files for July 2024 already in the cache."""
    matches = sorted(NPY_CACHE.glob(f"*{TILE_ID}*2024072*.npy"))
    l1c = [p for p in matches if "MSIL1C" in p.name]
    # Also check the first week of August (harvest sometimes runs into Aug)
    matches2 = sorted(NPY_CACHE.glob(f"*{TILE_ID}*20240[78]*.npy"))
    l1c += [p for p in matches2 if "MSIL1C" in p.name and p not in l1c]
    # Exclude the existing eval dates (Jun-08, Aug-31, Sep-18, Sep-20)
    known_dates = {"20240608", "20240831", "20240918", "20240920"}
    l1c = [p for p in l1c if not any(kd in p.name for kd in known_dates)]
    return l1c


def search_products(client: CopernicusClient, max_cloud: float) -> list:
    """Search Copernicus for T31UGS L1C products in July 2024."""
    wkt = site_bbox_wkt()
    try:
        products = client.search_products(
            wkt_polygon=wkt,
            start_date=SEARCH_START,
            end_date=SEARCH_END,
            collection="SENTINEL-2",
            max_cloud_cover=max_cloud,
        )
    except Exception as e:
        log.error("Search failed: %s", e)
        return []

    # Filter to T31UGS L1C only
    l1c = [p for p in products if p.tile_id == TILE_ID and "MSIL1C" in p.name]
    log.info("Found %d L1C products for %s in %s – %s (cloud ≤ %.0f%%)",
             len(l1c), TILE_ID, SEARCH_START[:10], SEARCH_END[:10], max_cloud)
    return l1c


def download_and_convert(product, client: CopernicusClient) -> tuple:
    """Download product, convert to .npy. Returns (npy_path, meta_path)."""
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    log.info("  Downloading %s ...", product.name[:70])
    zip_path = client.download_product(product, str(DOWNLOAD_DIR))
    if zip_path is None:
        raise RuntimeError(f"Download returned None for {product.name}")

    log.info("  Converting to .npy ...")
    extract_dir = tempfile.mkdtemp(prefix="s2_weisweiler_july_")
    try:
        npy_path, meta_path = safe_to_npy(
            zip_path=zip_path,
            output_dir=str(NPY_CACHE),
            tile_id=TILE_ID,
            acquisition_date=product.acquisition_date,
            satellite=product.satellite,
            extract_dir=extract_dir,
        )
        log.info("  Saved: %s", Path(npy_path).name)
        return npy_path, meta_path
    finally:
        shutil.rmtree(extract_dir, ignore_errors=True)
        log.info("  Zip retained: %s", zip_path)


def main():
    parser = argparse.ArgumentParser(
        description="Download Weisweiler T31UGS July 2024 scenes for lower-CV CEMF"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Search catalog only, no download")
    parser.add_argument("--max-scenes", type=int, default=DEFAULT_MAX_SCENES,
                        help=f"Maximum scenes to download (default: {DEFAULT_MAX_SCENES})")
    parser.add_argument("--max-cloud", type=float, default=DEFAULT_MAX_CLOUD,
                        help=f"Maximum cloud cover %% (default: {DEFAULT_MAX_CLOUD})")
    args = parser.parse_args()

    print("=" * 70)
    print("  Weisweiler July 2024 S2 Download — Low-CV CEMF Candidate Search")
    print(f"  Tile: {TILE_ID}   Site: {SITE_LAT}°N, {SITE_LON}°E")
    print(f"  Window: July 2024   Cloud limit: {args.max_cloud:.0f}%")
    print(f"  Max scenes: {args.max_scenes}")
    if args.dry_run:
        print("  MODE: DRY RUN (search only)")
    print("=" * 70)

    # ── Check cache first ─────────────────────────────────────────────────────
    cached = find_cached()
    if cached:
        log.info("Already cached (%d July/Aug scenes):", len(cached))
        for p in cached:
            log.info("  %s", p.name)
    else:
        log.info("No July 2024 T31UGS scenes cached yet")

    # ── Authenticate ─────────────────────────────────────────────────────────
    username = os.environ.get("COPERNICUS_USER", "").strip()
    password = os.environ.get("COPERNICUS_PASS", "").strip()
    if not username:
        username = input("\nCopernicus username (email): ").strip()
    if not password:
        password = getpass.getpass("Copernicus password: ")

    print("\n[1] Authenticating...")
    client = CopernicusClient(username, password)
    _ = client.token
    print("    OK\n")

    # ── Search ────────────────────────────────────────────────────────────────
    products = search_products(client, args.max_cloud)

    if not products and args.max_cloud < FALLBACK_MAX_CLOUD:
        log.warning("No products at %.0f%% cloud — retrying at %.0f%%...",
                    args.max_cloud, FALLBACK_MAX_CLOUD)
        products = search_products(client, FALLBACK_MAX_CLOUD)

    if not products:
        print("\nNo L1C products found in July 2024 for T31UGS. Try --max-cloud 30.")
        return

    # Sort by cloud cover (ascending), then date
    products.sort(key=lambda p: (p.cloud_cover or 99, p.acquisition_date or ""))

    # Filter out dates we already have in cache
    known_dates = {"20240608", "20240831", "20240918", "20240920"}
    new_products = [p for p in products if not any(kd in p.name for kd in known_dates)]

    print(f"\n[2] Available July 2024 L1C scenes for {TILE_ID}:")
    print(f"  {'Product':<70}  {'Cloud%':>7}  {'Date':>12}")
    print("  " + "-" * 95)
    for p in new_products:
        cloud_str = f"{p.cloud_cover:.1f}" if p.cloud_cover is not None else "—"
        acq = p.acquisition_date or "?"
        cached_flag = " [CACHED]" if any(p.name in str(cp) for cp in cached) else ""
        print(f"  {p.name[:70]:<70}  {cloud_str:>7}  {acq:>12}{cached_flag}")

    if not new_products:
        print("  (all found products are already in cache)")
        return

    if args.dry_run:
        print(f"\n[dry-run] Would download up to {args.max_scenes} scene(s)")
        print("\nNext steps after download:")
        print(f"  python apply_bitemporal_diff.py --sites {SITE_NAME} \\")
        print(f"         --weights weights/european_model_v8.pth")
        print("  Look for: cv_ctrl < 0.5 AND sc_ratio > 1.15")
        print("  If found: python scripts/run_cemf_weisweiler_bt.py --scan")
        return

    # ── Download ──────────────────────────────────────────────────────────────
    to_download = new_products[:args.max_scenes]
    results = {}

    print(f"\n[3] Downloading {len(to_download)} scene(s)...")
    for i, product in enumerate(to_download, 1):
        log.info("─" * 60)
        log.info("[%d/%d] %s  (cloud: %.1f%%)",
                 i, len(to_download), product.name[:65], product.cloud_cover or 0)

        # Skip if already in npy_cache
        existing = list(NPY_CACHE.glob(f"*{product.name[:44]}*.npy"))
        if existing:
            log.info("  Already cached: %s", existing[0].name)
            results[product.name] = {
                "status": "cached",
                "npy": str(existing[0]),
                "cloud_cover": product.cloud_cover,
                "acquisition_date": product.acquisition_date,
            }
            continue

        try:
            npy_path, meta_path = download_and_convert(product, client)
            results[product.name] = {
                "status": "ok",
                "npy": npy_path,
                "meta": meta_path,
                "cloud_cover": product.cloud_cover,
                "acquisition_date": product.acquisition_date,
            }
        except Exception as e:
            log.error("  Failed: %s", e)
            results[product.name] = {
                "status": "error",
                "error": str(e),
                "cloud_cover": product.cloud_cover,
                "acquisition_date": product.acquisition_date,
            }

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Results:")
    for name, r in results.items():
        status = r["status"]
        date   = r.get("acquisition_date", "?")
        cloud  = r.get("cloud_cover", 0) or 0
        if status == "ok":
            print(f"  ✓ {date}  cloud={cloud:.1f}%  → {Path(r['npy']).name}")
        elif status == "cached":
            print(f"  ✓ {date}  cloud={cloud:.1f}%  → {Path(r['npy']).name}  [cached]")
        else:
            print(f"  ✗ {date}  cloud={cloud:.1f}%  FAILED: {r.get('error', 'unknown')}")

    ok_count = sum(1 for r in results.values() if r["status"] in ("ok", "cached"))

    if ok_count > 0:
        print(f"\n[4] Next steps:")
        print(f"  1. Run bitemporal differencing on new dates:")
        print(f"     python apply_bitemporal_diff.py --sites {SITE_NAME} \\")
        print(f"            --weights weights/european_model_v8.pth")
        print(f"  2. Check results_analysis/multidate_validation.json for:")
        print(f"       cv_ctrl < 0.5   ← low terrain heterogeneity")
        print(f"       sc_ratio > 1.15 ← CFAR detection triggered")
        print(f"  3. If both conditions met on any July date:")
        print(f"     python scripts/run_cemf_weisweiler_bt.py \\")
        print(f"            --scene <scene_id> --scan")
        print(f"  Why July: post-harvest wheat/barley stubble is spectrally uniform,")
        print(f"  suppressing the Rhine agricultural patchwork that causes CV=1.607")
        print(f"  in September (mixed maize, beet, late cereals).")

    with open(MANIFEST, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Manifest saved → %s", MANIFEST)


if __name__ == "__main__":
    main()
