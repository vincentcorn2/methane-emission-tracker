"""
download_reference_tiles.py
============================
Step 2 of the bi-temporal retraining pipeline:

  For each key analysis site, download a December 2023 / January 2024
  Sentinel-2 L1C tile (same granule as our summer 2024 acquisitions),
  convert it to .npy format, and save it as a reference tile.

  The reference tile captures the "permanent" spectral signature of the
  terrain — geological features, urban surfaces, vegetation structure —
  in a season when the vegetation is dormant (no chlorophyll interference
  with SWIR bands) and methane emissions are known to be lower at most
  coal sites (reduced operational load in winter).

  Then, compute difference images:
    delta_B12 = target_B12 - reference_B12   (methane absorption channel)
    delta_B11 = target_B11 - reference_B11   (atmospheric reference channel)

  Running CH4Net on these difference images — even without retraining —
  should substantially suppress terrain false positives (Groningen, Rybnik)
  while preserving real emission signals at active sites (Weisweiler).

Usage:
    export COPERNICUS_USER=your@email.com
    export COPERNICUS_PASS='yourpassword'
    conda activate methane
    python download_reference_tiles.py

  Or run interactively — it will prompt for credentials if env vars absent.

Outputs (written to data/npy_cache/):
    {tile_id}_ref_YYYYMMDD.npy        — full 12-band reference array
    {tile_id}_ref_YYYYMMDD_geo.json   — geo metadata sidecar
    reference_tiles_manifest.json     — summary of what was downloaded

Pipeline compatibility:
  Reference .npy arrays have the same shape (H, W, 12) and uint8
  normalization as the primary acquisition arrays. The bitemporal
  difference is computed by apply_bitemporal_diff.py (next step).
"""

import os
import sys
import json
import getpass
import logging
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np

# ── Set up logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("results_analysis/ref_tile_download.log"),
    ],
)
log = logging.getLogger(__name__)

# ── Add project root to path ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from src.ingestion.copernicus_client import CopernicusClient
from src.ingestion.preprocessing import safe_to_npy

# ── Configuration ─────────────────────────────────────────────────────────────

# Search window: winter 2023/2024 — dormant vegetation, lower emissions
# Use December + January rather than November (still partly green)
REF_DATE_START = "2023-11-01T00:00:00.000Z"
REF_DATE_END   = "2024-02-28T23:59:59.000Z"
MAX_CLOUD      = 25.0   # %; prefer low cloud — will take best available

# Output directories
NPY_CACHE   = Path("data/npy_cache")
DOWNLOAD_DIR = Path("data/downloads/reference")
MANIFEST    = Path("results_analysis/reference_tiles_manifest.json")

NPY_CACHE.mkdir(parents=True, exist_ok=True)
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
Path("results_analysis").mkdir(exist_ok=True)

# ── Target sites ──────────────────────────────────────────────────────────────
# Each entry ties the site name to:
#   tile_id   — Sentinel-2 Military Grid Reference tile (e.g. T32ULB)
#   lat / lon — centroid of the facility (used to build the WKT intersect polygon)
#   note      — why this site is a priority

SITES = [
    {
        "name": "weisweiler",
        "tile_id": "T31UGS",
        "lat": 50.837,
        "lon": 6.322,
        "note": "Weisweiler lignite plant (confirmed S/C=2.091 on two dates) — primary true positive. T31UGS orbit R008 covers western position; T32ULB R108 is zero-padded at this lon.",
    },
    {
        "name": "boxberg",
        "tile_id": "T33UVT",
        "lat": 51.413,
        "lon": 14.582,
        "note": "Boxberg/Lausitz (S/C=1.517, marginal above Lausitz background 1.397)",
    },
    {
        "name": "rybnik",
        "tile_id": "T34UCA",
        "lat": 50.090,
        "lon": 18.529,
        "note": "Rybnik (ring profile increases outward — terrain artifact, bi-temporal should suppress)",
    },
    {
        "name": "groningen",
        "tile_id": "T31UGV",
        "lat": 53.252,
        "lon": 6.682,
        "note": "Groningen gas field (TROPOMI confirmed non-detection — terrain artifact)",
    },
]


# ── Helper: build WKT point-polygon around a lat/lon centroid ─────────────────
def site_bbox_wkt(lat: float, lon: float, margin_deg: float = 0.3) -> str:
    """
    Build a small WKT POLYGON bounding box around a site centroid.

    Sentinel-2 tiles are 100x100 km each. A 0.3° margin (~33 km) is
    more than sufficient to intersect the correct tile while excluding
    adjacent tiles from a different granule.
    """
    min_lon = lon - margin_deg
    max_lon = lon + margin_deg
    min_lat = lat - margin_deg
    max_lat = lat + margin_deg
    return (
        f"POLYGON(({min_lon} {min_lat},{max_lon} {min_lat},"
        f"{max_lon} {max_lat},{min_lon} {max_lat},{min_lon} {min_lat}))"
    )


# ── Main download loop ────────────────────────────────────────────────────────
def download_reference_tiles(client: CopernicusClient) -> dict:
    """
    Search, download, and convert reference tiles for all sites.

    Returns:
        manifest dict mapping site_name → result info
    """
    manifest = {}

    for site in SITES:
        name    = site["name"]
        tile_id = site["tile_id"]
        log.info("=" * 60)
        log.info("Site: %s  (tile %s)", name.upper(), tile_id)
        log.info("Note: %s", site["note"])

        # ── Check if already done ─────────────────────────────────────────
        existing = list(NPY_CACHE.glob(f"{tile_id}_ref_*.npy"))
        if existing:
            npy_path = existing[0]
            log.info("  Already cached: %s — skipping download", npy_path.name)
            manifest[name] = {
                "tile_id":      tile_id,
                "npy_path":     str(npy_path),
                "status":       "cached",
            }
            continue

        # ── Build WKT for the site centroid ───────────────────────────────
        wkt = site_bbox_wkt(site["lat"], site["lon"])
        log.info("  Searching %s – %s (cloud ≤ %.0f%%)...",
                 REF_DATE_START[:10], REF_DATE_END[:10], MAX_CLOUD)

        # ── Search Copernicus for winter tiles ────────────────────────────
        try:
            products = client.search_products(
                wkt_polygon=wkt,
                start_date=REF_DATE_START,
                end_date=REF_DATE_END,
                collection="SENTINEL-2",
                max_cloud_cover=MAX_CLOUD,
                max_results=50,
            )
        except Exception as e:
            log.error("  Search failed for %s: %s", name, e)
            manifest[name] = {"tile_id": tile_id, "status": "search_error", "error": str(e)}
            continue

        # ── Filter to exact tile_id ───────────────────────────────────────
        tile_products = [p for p in products if p.tile_id == tile_id]
        log.info("  Found %d products total, %d for tile %s",
                 len(products), len(tile_products), tile_id)

        if not tile_products:
            # Relax cloud cover and try again
            log.warning("  No products with cloud ≤ %.0f%%. Trying ≤ 50%%...", MAX_CLOUD)
            try:
                products2 = client.search_products(
                    wkt_polygon=wkt,
                    start_date=REF_DATE_START,
                    end_date=REF_DATE_END,
                    collection="SENTINEL-2",
                    max_cloud_cover=50.0,
                    max_results=50,
                )
                tile_products = [p for p in products2 if p.tile_id == tile_id]
                log.info("  Relaxed search: %d products for tile %s", len(tile_products), tile_id)
            except Exception as e:
                log.error("  Relaxed search failed: %s", e)

        if not tile_products:
            log.error("  No products found for %s / %s — skipping", name, tile_id)
            manifest[name] = {"tile_id": tile_id, "status": "no_products"}
            continue

        # ── Sort by cloud cover (ascending), then by date (prefer Dec/Jan) ──
        # December and January acquisitions preferred: further from summer
        # vegetation green-up and from the summer acquisition dates.
        def sort_key(p):
            cloud = p.cloud_cover if p.cloud_cover is not None else 99.0
            # Prefer Dec 2023 (month 12) and Jan 2024 (month 1) over Feb 2024
            try:
                month = int(p.acquisition_date[5:7])
                # month score: Dec=0, Jan=1, Feb=2, Nov=3, Oct=4 (lower = better)
                month_score = {12: 0, 1: 1, 2: 2, 11: 3, 10: 4}.get(month, 5)
            except (ValueError, IndexError):
                month_score = 5
            return (cloud, month_score)

        tile_products.sort(key=sort_key)
        best = tile_products[0]

        log.info("  Best product: %s", best.name[:70])
        log.info("  Cloud cover: %.1f%%  Date: %s",
                 best.cloud_cover or 0, best.acquisition_date[:10])

        # ── Download ──────────────────────────────────────────────────────
        log.info("  Downloading (~500–700 MB)...")
        try:
            zip_path = client.download_product(best, str(DOWNLOAD_DIR))
        except Exception as e:
            log.error("  Download failed: %s", e)
            manifest[name] = {"tile_id": tile_id, "status": "download_error", "error": str(e)}
            continue

        if zip_path is None:
            log.error("  Download returned None for %s", name)
            manifest[name] = {"tile_id": tile_id, "status": "download_none"}
            continue

        log.info("  Downloaded: %s", zip_path)

        # ── Convert SAFE → .npy ───────────────────────────────────────────
        log.info("  Converting SAFE archive to .npy (rasterio required)...")
        extract_tmp = tempfile.mkdtemp(prefix=f"ref_{tile_id}_")
        try:
            npy_path, meta_path = safe_to_npy(
                zip_path=zip_path,
                output_dir=str(NPY_CACHE),
                tile_id=tile_id,
                acquisition_date=best.acquisition_date[:10],
                satellite=best.satellite,
                extract_dir=extract_tmp,
            )
        except Exception as e:
            log.error("  SAFE→NPY conversion failed: %s", e)
            manifest[name] = {
                "tile_id": tile_id, "status": "conversion_error", "error": str(e),
                "zip_path": str(zip_path),
            }
            shutil.rmtree(extract_tmp, ignore_errors=True)
            continue
        finally:
            shutil.rmtree(extract_tmp, ignore_errors=True)

        # ── Rename to reference tile convention ───────────────────────────
        acq_date_clean = best.acquisition_date[:10].replace("-", "")
        ref_npy  = NPY_CACHE / f"{tile_id}_ref_{acq_date_clean}.npy"
        ref_meta = NPY_CACHE / f"{tile_id}_ref_{acq_date_clean}_geo.json"

        npy_src  = Path(npy_path)
        meta_src = Path(meta_path)

        if npy_src != ref_npy:
            npy_src.rename(ref_npy)
            log.info("  Renamed → %s", ref_npy.name)
        if meta_src != ref_meta:
            meta_src.rename(ref_meta)

        # ── Quick sanity check ────────────────────────────────────────────
        arr = np.load(ref_npy, mmap_mode="r")
        log.info("  Shape: %s  dtype: %s  B11 mean: %.1f  B12 mean: %.1f",
                 arr.shape, arr.dtype,
                 float(arr[:, :, 10].mean()),
                 float(arr[:, :, 11].mean()))

        manifest[name] = {
            "tile_id":        tile_id,
            "npy_path":       str(ref_npy),
            "meta_path":      str(ref_meta),
            "product_name":   best.name,
            "acquisition_date": best.acquisition_date[:10],
            "cloud_cover":    best.cloud_cover,
            "shape":          list(arr.shape),
            "status":         "ok",
        }
        log.info("  ✓ Reference tile ready: %s", ref_npy.name)

    return manifest


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Reference Tile Download — Step 2 (Bi-temporal)")
    print("  Window: Nov 2023 – Feb 2024")
    print("  Sites: Weisweiler, Boxberg, Rybnik, Groningen")
    print("=" * 60)

    # Credentials — env vars preferred for non-interactive (nohup) runs
    username = os.environ.get("COPERNICUS_USER") or input("\nCopernicus username (email): ").strip()
    password = os.environ.get("COPERNICUS_PASS") or getpass.getpass("Copernicus password: ")

    print("\n[1] Authenticating...")
    try:
        client = CopernicusClient(username, password)
        _ = client.token   # trigger auth to catch bad credentials early
        print("    OK")
    except Exception as e:
        print(f"    Auth failed: {e}")
        sys.exit(1)

    print("\n[2] Downloading reference tiles...")
    manifest = download_reference_tiles(client)

    print("\n[3] Results summary:")
    ok     = [s for s, r in manifest.items() if r.get("status") in ("ok", "cached")]
    failed = [s for s, r in manifest.items() if r.get("status") not in ("ok", "cached")]

    for site, result in manifest.items():
        status = result.get("status", "?")
        icon   = "✓" if status in ("ok", "cached") else "✗"
        extra  = ""
        if status == "ok":
            extra = f"  {result.get('acquisition_date','?')}  cloud={result.get('cloud_cover',0):.0f}%"
        elif status == "cached":
            extra = f"  (cached) {result.get('npy_path','').split('/')[-1]}"
        elif "error" in result:
            extra = f"  ERROR: {result.get('error','')[:60]}"
        print(f"  {icon} {site:<15} {status:<20}{extra}")

    print(f"\n  Succeeded: {len(ok)}/{len(SITES)}  Failed: {len(failed)}/{len(SITES)}")

    # Save manifest
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\n  Manifest saved → {MANIFEST}")

    if ok:
        print("\n[4] Next step: run apply_bitemporal_diff.py to compute")
        print("    delta_B11 / delta_B12 difference arrays, then re-run")
        print("    the CH4Net v2 analysis on the difference imagery.")

    sys.exit(0 if len(failed) == 0 else 1)
