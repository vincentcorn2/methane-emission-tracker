"""
download_eval_tiles.py
======================
Downloads summer 2024 Sentinel-2 L1C target tiles for priority evaluation
sites that are not yet in data/npy_cache/.

Current targets:
  - Bełchatów  (T34UDA) — Europe's #1 CO2 emitter; fresh tile needed
  - Lippendorf (T33UUS) — 891 MW lignite x2; was incorrectly mapped to T33UUT

Usage:
    export COPERNICUS_USER=your@email.com
    export COPERNICUS_PASS='yourpassword'
    conda activate methane
    python download_eval_tiles.py

    # Dry run (search only):
    python download_eval_tiles.py --dry-run

    # One site only:
    python download_eval_tiles.py --site belchatow

Outputs (written to data/npy_cache/):
    <product_name>.npy          — 12-band uint8 array
    <product_name>_geo.json     — geo metadata sidecar

Estimated time: ~40 min per tile (download + rasterio conversion)
Use 'caffeinate -i python download_eval_tiles.py' to prevent sleep.
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

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("results_analysis/eval_tile_download.log"),
    ],
)
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))
from src.ingestion.copernicus_client import CopernicusClient
from src.ingestion.preprocessing import safe_to_npy

NPY_CACHE = Path("data/npy_cache")
DOWNLOAD_DIR = Path("data/downloads/eval")

# Search window: summer 2024 (when most industrial emissions are detectable)
TARGET_DATE_START = "2024-06-01T00:00:00.000Z"
TARGET_DATE_END   = "2024-09-30T23:59:59.000Z"
MAX_CLOUD = 15.0  # % — strict; relax to 25 if nothing found

# ── Target sites ───────────────────────────────────────────────────────────────
SITES = [
    {
        "name": "belchatow",
        # T34UDA was wrong — MGRS calc puts Bełchatów (51.27°N, 19.32°E, Zone 34N)
        # at easting ~382k, northing ~5681k. Try T34UCB (Silesia is T34UCB at 50°N),
        # T34UCC (Warsaw region), T34UCD in order — discover_tile_ids below finds the real one.
        "tile_id": None,  # will be discovered from catalog
        "tile_candidates": ["T34UCB", "T34UCC", "T34UCD"],  # T34UCB confirmed correct
        "lat": 51.266,
        "lon": 19.315,
        "note": "Bełchatów lignite plant — Europe's #1 CO2 emitter (858 MW, Poland)",
    },
    {
        "name": "lippendorf",
        "tile_id": "T33UUS",
        "lat": 51.178,
        "lon": 12.378,
        "note": "Lippendorf lignite plant — 891 MW x2 near Leipzig (T33UUS, not T33UUT)",
    },
]


def site_bbox_wkt(lat: float, lon: float, margin_deg: float = 0.3) -> str:
    """Build a WKT POLYGON bounding box around a site centroid."""
    min_lat, max_lat = lat - margin_deg, lat + margin_deg
    min_lon, max_lon = lon - margin_deg, lon + margin_deg
    return (
        f"POLYGON(({min_lon} {min_lat},"
        f"{max_lon} {min_lat},"
        f"{max_lon} {max_lat},"
        f"{min_lon} {max_lat},"
        f"{min_lon} {min_lat}))"
    )


def find_cached(tile_id: str) -> list:
    """Return L1C .npy files already cached for this tile ID (exclude L2A)."""
    all_matches = sorted(NPY_CACHE.glob(f"*{tile_id}*.npy"))
    l1c = [p for p in all_matches if "MSIL1C" in p.name]
    return l1c


def discover_tile_id(client, lat, lon, candidates):
    """
    Find which MGRS tile ID the catalog actually uses for this location.
    Searches a broad area and returns the first candidate tile_id found.
    Returns (tile_id, products) or (None, []).
    """
    wkt = site_bbox_wkt(lat, lon, margin_deg=0.5)
    try:
        products = client.search_products(
            wkt_polygon=wkt,
            start_date=TARGET_DATE_START,
            end_date=TARGET_DATE_END,
            collection="SENTINEL-2",
            max_cloud_cover=30.0,
        )
    except Exception as e:
        log.error("  Discovery search failed: %s", e)
        return None, []

    # Print all unique tile IDs found so operator can verify
    found_tiles = sorted({p.tile_id for p in products if p.tile_id != "unknown"})
    log.info("  Tiles found in search area: %s", found_tiles)

    # Try each candidate in order
    for candidate in candidates:
        matching = [p for p in products if p.tile_id == candidate and "MSIL1C" in p.name]
        if matching:
            log.info("  Resolved tile: %s (%d L1C products)", candidate, len(matching))
            return candidate, matching

    # No candidate matched — show what we found to help diagnose
    log.warning("  None of the candidates %s matched. Tiles found: %s", candidates, found_tiles)
    return None, []


def download_and_convert(product, client, site, extract_root):
    """Download a Sentinel-2 product, convert to .npy, save to npy_cache."""
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    log.info("  Downloading %s ...", product.name[:70])
    zip_path = client.download_product(product, str(DOWNLOAD_DIR))
    if zip_path is None:
        raise RuntimeError(f"Download returned None for {product.name}")

    log.info("  Converting to .npy ...")
    extract_dir = tempfile.mkdtemp(prefix="s2_eval_", dir=str(extract_root))
    try:
        npy_path, meta_path = safe_to_npy(
            zip_path=zip_path,
            output_dir=str(NPY_CACHE),
            tile_id=site["tile_id"],
            acquisition_date=product.acquisition_date,
            satellite=product.satellite,
            extract_dir=extract_dir,
        )
        log.info("  Saved: %s", Path(npy_path).name)
        return npy_path, meta_path
    finally:
        shutil.rmtree(extract_dir, ignore_errors=True)
        # Keep the zip in case re-run needed; can delete manually
        log.info("  Zip retained at: %s", zip_path)


def main(dry_run: bool = False, site_filter: str = None):
    print("=" * 60)
    print("  Eval Tile Download — Priority European Sites")
    print(f"  Window: Jun 2024 – Sep 2024")
    if dry_run:
        print("  MODE: DRY RUN (search only, no downloads)")
    print("=" * 60)

    # ── Credentials ───────────────────────────────────────────────────────────
    username = os.environ.get("COPERNICUS_USER", "").strip()
    password = os.environ.get("COPERNICUS_PASS", "").strip()
    if not username:
        username = input("\nCopernicus username (email): ").strip()
    if not password:
        password = getpass.getpass("Copernicus password: ")

    print("\n[1] Authenticating...")
    client = CopernicusClient(username, password)
    _ = client.token  # trigger auth immediately to fail fast
    print("    OK\n")

    sites_to_run = [s for s in SITES if site_filter is None or s["name"] == site_filter]
    if not sites_to_run:
        log.error("No sites matched filter '%s'", site_filter)
        sys.exit(1)

    extract_root = Path(tempfile.gettempdir())
    results = {}

    print("[2] Downloading target tiles...\n")
    for site in sites_to_run:
        name = site["name"]
        tile_id = site.get("tile_id")
        candidates = site.get("tile_candidates", [tile_id] if tile_id else [])
        log.info("=" * 60)
        log.info("Site: %s  (tile: %s)", name.upper(), tile_id or "discovering...")
        log.info("Note: %s", site["note"])

        # Discover tile ID if not known
        if tile_id is None:
            log.info("  Running tile ID discovery...")
            tile_id, tile_products = discover_tile_id(
                client, site["lat"], site["lon"], candidates
            )
            if tile_id is None:
                log.error("  Could not determine tile ID for %s", name)
                results[name] = {"status": "no_tile_id", "candidates_tried": candidates}
                continue
            site["tile_id"] = tile_id  # persist for later steps
        else:
            tile_products = []  # will search below

        # Check cache first (L1C only)
        cached = find_cached(tile_id)
        if cached:
            log.info("  Already cached (L1C): %s — skipping download", cached[0].name)
            results[name] = {"status": "cached", "file": str(cached[0]), "tile_id": tile_id}
            continue

        # Search Copernicus catalog — L1C only
        if not tile_products:
            wkt = site_bbox_wkt(site["lat"], site["lon"])
            try:
                products = client.search_products(
                    wkt_polygon=wkt,
                    start_date=TARGET_DATE_START,
                    end_date=TARGET_DATE_END,
                    collection="SENTINEL-2",
                    max_cloud_cover=MAX_CLOUD,
                )
            except Exception as e:
                log.error("  Search failed: %s", e)
                results[name] = {"status": "search_error", "error": str(e)}
                continue

            # Filter to exact tile AND L1C only (L2A has atmospheric correction
            # that destroys the methane signal — CH4Net requires raw L1C TOA reflectance)
            tile_products = [
                p for p in products
                if p.tile_id == tile_id and "MSIL1C" in p.name
            ]
            log.info("  Found %d L1C products for tile %s (of %d total)",
                     len(tile_products), tile_id, len(products))

            if not tile_products:
                # Relax cloud cover and retry
                log.warning("  No L1C products at %.0f%% cloud — retrying at 30%%...", MAX_CLOUD)
                try:
                    products2 = client.search_products(
                        wkt_polygon=wkt,
                        start_date=TARGET_DATE_START,
                        end_date=TARGET_DATE_END,
                        collection="SENTINEL-2",
                        max_cloud_cover=30.0,
                    )
                    tile_products = [
                        p for p in products2
                        if p.tile_id == tile_id and "MSIL1C" in p.name
                    ]
                    log.info("  Relaxed search: %d L1C products for tile %s",
                             len(tile_products), tile_id)
                except Exception as e:
                    log.error("  Relaxed search failed: %s", e)

        if not tile_products:
            log.error("  No L1C products found for %s / %s — skipping", name, tile_id)
            results[name] = {"status": "no_products", "tile_id": tile_id}
            continue

        # Sort by cloud cover, pick the best L1C
        tile_products.sort(key=lambda p: (p.cloud_cover or 99))
        best = tile_products[0]
        log.info(
            "  Best match: %s  (cloud: %.1f%%)",
            best.name[:65],
            best.cloud_cover or 0,
        )

        if dry_run:
            log.info("  [dry-run] Would download %s", best.name[:65])
            results[name] = {"status": "dry_run", "would_download": best.name}
            continue

        try:
            npy_path, meta_path = download_and_convert(best, client, site, extract_root)
            results[name] = {
                "status": "ok", "tile_id": tile_id,
                "npy": npy_path, "meta": meta_path,
            }
        except Exception as e:
            log.error("  Failed: %s", e)
            results[name] = {"status": "error", "tile_id": tile_id, "error": str(e)}

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Results:")
    for name, r in results.items():
        status = r["status"]
        if status == "ok":
            print(f"  ✓ {name:<15} downloaded → {Path(r['npy']).name}")
        elif status == "cached":
            print(f"  ✓ {name:<15} already cached → {Path(r['file']).name}")
        elif status == "dry_run":
            print(f"  ~ {name:<15} [dry-run] would download: {r['would_download'][:50]}")
        else:
            print(f"  ✗ {name:<15} {status}: {r.get('error', r.get('tile_id', ''))}")

    manifest_path = "results_analysis/eval_tile_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nManifest saved → {manifest_path}")

    if not dry_run:
        ok_count = sum(1 for r in results.values() if r["status"] in ("ok", "cached"))
        print(f"\n[3] Next step: run apply_bitemporal_diff.py to evaluate new sites:")
        if ok_count > 0:
            sites_ready = " ".join(
                s["name"] for s in sites_to_run
                if results.get(s["name"], {}).get("status") in ("ok", "cached")
            )
            print(f"    python apply_bitemporal_diff.py --sites {sites_ready} "
                  f"--weights weights/european_model_v8.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download eval tiles for priority European sites")
    parser.add_argument("--dry-run", action="store_true", help="Search only, no downloads")
    parser.add_argument("--site", default=None, help="Download only one site (belchatow or lippendorf)")
    args = parser.parse_args()
    main(dry_run=args.dry_run, site_filter=args.site)
