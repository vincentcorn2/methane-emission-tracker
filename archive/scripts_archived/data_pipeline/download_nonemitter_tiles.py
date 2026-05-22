"""
scripts/download_nonemitter_tiles.py
======================================
Download one Sentinel-2 L1C scene per non-emitter reference location for
WS5 conformal calibration and FPR characterisation.

Reads results_analysis/nonemitter_manifest.json (produced by
ws5_sample_nonemitters.py) and for each location:

  1. Searches the Copernicus catalog for the tile that contains the
     location's MGRS cell during the target month.
  2. Selects the scene with the lowest cloud cover (≤ 10% preferred,
     ≤ 25% fallback).
  3. Downloads and converts to .npy in data/nonemitter_tiles/.
  4. Updates results_analysis/nonemitter_download_manifest.json.

Each .npy is the full-tile array (same format as data/npy_cache/) so that
run_nonemitter_inference.py can run CH4Net v8 in the standard crop-inference
loop without any format changes.

Usage:
    conda activate methane
    export COPERNICUS_USER=your@email.com
    export COPERNICUS_PASS='yourpassword'

    python scripts/download_nonemitter_tiles.py [--dry-run] [--ids nonemit_001 nonemit_002]
    python scripts/download_nonemitter_tiles.py --max-cloud 25 --max-scenes-per-site 2

Notes:
    - Tiles already present in data/nonemitter_tiles/ are skipped.
    - ~40 min per tile — use 'caffeinate -i python ...' on macOS.
    - HARD_EXCLUDE proximity-flagged locations are skipped automatically.
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
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS_DIR = Path("results_analysis")
RESULTS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(RESULTS_DIR / "nonemitter_download.log")),
    ],
)
log = logging.getLogger(__name__)

from src.ingestion.copernicus_client import CopernicusClient
from src.ingestion.preprocessing import safe_to_npy

# ── Paths ──────────────────────────────────────────────────────────────────────
MANIFEST_IN  = RESULTS_DIR / "nonemitter_manifest.json"
MANIFEST_OUT = RESULTS_DIR / "nonemitter_download_manifest.json"
TILE_DIR     = Path("data/nonemitter_tiles")
DOWNLOAD_DIR = Path("data/downloads/nonemitter")

# ── Cloud cover limits ─────────────────────────────────────────────────────────
DEFAULT_MAX_CLOUD  = 10.0
FALLBACK_MAX_CLOUD = 25.0

# ── Search window: ±30 days around 15th of target month ───────────────────────
SEARCH_WINDOW_DAYS = 30
# Fallback years to try when target year has no L1C (ESA only archives L2A for some tiles)
FALLBACK_YEARS = [2023, 2022]


def date_window(target_ym: str, search_window_days: int = SEARCH_WINDOW_DAYS) -> tuple[str, str]:
    """
    Given "2024-07", return OData date strings for ±N days around the 15th.
    E.g. "2024-07" → ("2024-06-15T00:00:00.000Z", "2024-08-14T23:59:59.000Z")
    """
    mid = datetime.strptime(f"{target_ym}-15", "%Y-%m-%d")
    start = (mid - timedelta(days=search_window_days)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    end   = (mid + timedelta(days=search_window_days)).strftime("%Y-%m-%dT%H:%M:%S.999Z")
    return start, end


def substitute_year(target_ym: str, year: int) -> str:
    """Replace the year in a 'YYYY-MM' string: '2024-07' → '2023-07'."""
    month = target_ym.split("-")[1]
    return f"{year}-{month}"


def bbox_wkt(lat: float, lon: float, margin: float = 0.25) -> str:
    return (
        f"POLYGON(({lon-margin} {lat-margin},{lon+margin} {lat-margin},"
        f"{lon+margin} {lat+margin},{lon-margin} {lat+margin},"
        f"{lon-margin} {lat-margin}))"
    )


def find_cached(location_id: str, mgrs_tile: str) -> list:
    """Return .npy files already downloaded for this location.

    Matches on location_id prefix — accepts any tile suffix so that files
    downloaded via a neighboring tile (fallback) are also recognised.
    """
    return sorted(TILE_DIR.glob(f"{location_id}_*.npy"))


def search_for_location(client: CopernicusClient, loc: dict,
                        max_cloud: float,
                        search_window_days: int = SEARCH_WINDOW_DAYS,
                        target_ym: str | None = None) -> list:
    """Search Copernicus catalog for L1C scenes covering a non-emitter location.

    Args:
        target_ym: Override for the target year-month (e.g. '2023-07').
                   Defaults to loc['target_date'].
    """
    ym = target_ym or loc["target_date"]
    start, end = date_window(ym, search_window_days)
    wkt = bbox_wkt(loc["lat"], loc["lon"])

    try:
        products = client.search_products(
            wkt_polygon=wkt,
            start_date=start,
            end_date=end,
            collection="SENTINEL-2",
            max_cloud_cover=max_cloud,
        )
    except Exception as e:
        log.error("  Search failed for %s: %s", loc["id"], e)
        return []

    # Filter to L1C only.
    # Strategy: prefer the exact MGRS tile specified in the manifest, but fall back
    # to ANY L1C tile that covers the coordinate.  CDSE only archives L1C online for
    # high-demand tiles; for others, only L2A exists.  Any S2 L1C tile returned by
    # an intersects-bbox search necessarily covers the target coordinate, so a
    # neighboring tile is a valid substitute for the crop-inference step.
    mgrs      = loc["mgrs_tile"]
    all_count = len(products)
    all_l1c   = [p for p in products if "MSIL1C" in p.name]
    tile_l1c  = [p for p in all_l1c  if mgrs in p.tile_id]

    if tile_l1c:
        l1c = tile_l1c
        log.info("  Found %d L1C scenes (tile %s, of %d total) for %s (cloud ≤ %.0f%%, window ±%dd of %s)",
                 len(l1c), mgrs, all_count, loc["id"], max_cloud, search_window_days, ym)
    elif all_l1c:
        l1c = all_l1c
        actual_tiles = sorted({p.tile_id for p in all_l1c})
        log.info("  Found %d L1C scenes (fallback tiles %s — %s has no online L1C) "
                 "for %s (cloud ≤ %.0f%%, window ±%dd of %s)",
                 len(l1c), actual_tiles, mgrs, loc["id"], max_cloud, search_window_days, ym)
    else:
        l1c = []
        log.info("  Found 0 L1C scenes (of %d total, only L2A on CDSE) for %s "
                 "(tile %s, cloud ≤ %.0f%%, window ±%dd of %s)",
                 all_count, loc["id"], mgrs, max_cloud, search_window_days, ym)
    return l1c


def download_one(product, client: CopernicusClient,
                 location_id: str, mgrs_tile: str) -> dict:
    """Download a single product and convert to .npy. Returns a result dict."""
    TILE_DIR.mkdir(parents=True, exist_ok=True)
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already cached under npy_cache/ (full tile from a previous run)
    from pathlib import Path as _P
    npy_cache = _P("data/npy_cache")
    existing_cache = list(npy_cache.glob(f"*{product.name[:44]}*.npy"))
    if existing_cache:
        log.info("  Already in npy_cache: %s — symlinking", existing_cache[0].name)
        dest = TILE_DIR / f"{location_id}_{existing_cache[0].name}"
        if not dest.exists():
            dest.symlink_to(existing_cache[0].resolve())
        return {
            "status": "cached",
            "npy": str(dest),
            "source": "npy_cache_symlink",
            "cloud_cover": product.cloud_cover,
            "acquisition_date": product.acquisition_date,
            "product_name": product.name,
        }

    log.info("  Downloading %s ...", product.name[:70])
    zip_path = client.download_product(product, str(DOWNLOAD_DIR))
    if zip_path is None:
        raise RuntimeError(f"Download returned None for {product.name}")

    log.info("  Converting to .npy ...")
    extract_dir = tempfile.mkdtemp(prefix=f"s2_{location_id}_")
    try:
        # Use the actual tile_id from the product (may differ from manifest's mgrs_tile
        # when we fall back to a neighboring tile that has online L1C).
        actual_tile = product.tile_id.lstrip("T") if product.tile_id else mgrs_tile
        npy_path, meta_path = safe_to_npy(
            zip_path=zip_path,
            output_dir=str(TILE_DIR),
            tile_id=actual_tile,
            acquisition_date=product.acquisition_date,
            satellite=product.satellite,
            extract_dir=extract_dir,
        )
        # Rename so the file carries the location_id prefix for easy lookup
        final_npy = TILE_DIR / f"{location_id}_{Path(npy_path).name}"
        Path(npy_path).rename(final_npy)
        log.info("  Saved: %s", final_npy.name)
        return {
            "status": "ok",
            "npy": str(final_npy),
            "meta": meta_path,
            "cloud_cover": product.cloud_cover,
            "acquisition_date": product.acquisition_date,
            "product_name": product.name,
            "actual_tile": product.tile_id,   # record when a fallback tile was used
        }
    finally:
        shutil.rmtree(extract_dir, ignore_errors=True)


def process_location(loc: dict, client: CopernicusClient,
                     max_cloud: float, dry_run: bool,
                     max_scenes: int = 1,
                     search_window_days: int = SEARCH_WINDOW_DAYS,
                     fallback_years: list | None = None) -> dict:
    """Handle one non-emitter location end-to-end.

    When no L1C is found in the target year, automatically retries with
    fallback_years (same month, earlier years).  This handles the ESA CDSE
    tile-level deprecation of L1C — many European tiles only have L2A in the
    standard archive for 2024 but have L1C from 2022/2023.
    """
    loc_id    = loc["id"]
    mgrs      = loc["mgrs_tile"]
    prox_flag = loc.get("proximity_flag", "OK")
    fb_years  = fallback_years if fallback_years is not None else FALLBACK_YEARS

    log.info("── %s  (%s)  %s ──", loc_id, loc["label"][:40], mgrs)

    # Hard-exclude locations that are too close to an emitter
    if prox_flag == "HARD_EXCLUDE":
        log.warning("  SKIP: HARD_EXCLUDE (%.0f km from %s)",
                    loc["min_dist_to_emitter_km"],
                    loc["nearest_emitter_site"])
        return {"location_id": loc_id, "status": "skipped_hard_exclude",
                "proximity_flag": prox_flag}

    # Check local cache first
    cached = find_cached(loc_id, mgrs)
    if cached:
        log.info("  Already downloaded: %s", cached[0].name)
        return {"location_id": loc_id, "status": "cached", "npy": str(cached[0]),
                "proximity_flag": prox_flag}

    # ── Search: target year first, then fallback years ────────────────────────
    year_sequence = [loc["target_date"]] + [
        substitute_year(loc["target_date"], yr) for yr in fb_years
    ]
    products = []
    used_ym  = None

    for ym in year_sequence:
        products = search_for_location(client, loc, max_cloud, search_window_days, ym)
        if not products and max_cloud < FALLBACK_MAX_CLOUD:
            log.warning("  No L1C at %.0f%% cloud in %s — retrying at %.0f%%",
                        max_cloud, ym, FALLBACK_MAX_CLOUD)
            products = search_for_location(client, loc, FALLBACK_MAX_CLOUD,
                                           search_window_days, ym)
        if products:
            used_ym = ym
            if ym != loc["target_date"]:
                log.info("  Using fallback year %s (target year had no L1C)", ym)
            break
        else:
            log.warning("  No L1C found in %s (±%dd window)", ym, search_window_days)

    if not products:
        log.warning("  No scenes found for %s after trying years: %s",
                    mgrs, [y.split("-")[0] for y in year_sequence])
        return {"location_id": loc_id, "status": "not_found",
                "proximity_flag": prox_flag}

    # Sort by cloud cover, pick best
    products.sort(key=lambda p: (p.cloud_cover or 99, p.acquisition_date or ""))
    candidates = products[:max_scenes]

    log.info("  Best candidate: %s  cloud=%.1f%%",
             candidates[0].name[:65], candidates[0].cloud_cover or 0)

    if dry_run:
        return {
            "location_id": loc_id,
            "status":       "dry_run",
            "would_download": candidates[0].name,
            "cloud_cover":    candidates[0].cloud_cover,
            "acquisition_date": candidates[0].acquisition_date,
            "fallback_year":  used_ym if used_ym != loc["target_date"] else None,
            "actual_tile":    candidates[0].tile_id,
            "proximity_flag": prox_flag,
        }

    # Download
    try:
        result = download_one(candidates[0], client, loc_id, mgrs)
        result["location_id"]  = loc_id
        result["proximity_flag"] = prox_flag
        if used_ym and used_ym != loc["target_date"]:
            result["fallback_year"] = used_ym
        return result
    except Exception as e:
        log.error("  Download failed: %s", e)
        return {"location_id": loc_id, "status": "error", "error": str(e),
                "proximity_flag": prox_flag}


def main():
    parser = argparse.ArgumentParser(
        description="Download S2 L1C tiles for WS5 non-emitter reference locations"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Search catalog only, no downloads")
    parser.add_argument("--ids", nargs="+", default=None,
                        help="Process only specific location IDs (e.g. nonemit_001 nonemit_005)")
    parser.add_argument("--max-cloud", type=float, default=DEFAULT_MAX_CLOUD,
                        help=f"Preferred max cloud cover %% (default: {DEFAULT_MAX_CLOUD})")
    parser.add_argument("--max-scenes-per-site", type=int, default=1,
                        help="Max scenes to download per location (default: 1)")
    parser.add_argument("--search-window", type=int, default=SEARCH_WINDOW_DAYS,
                        help=f"Search window ±N days around target month (default: {SEARCH_WINDOW_DAYS})")
    parser.add_argument("--no-fallback-years", action="store_true",
                        help="Disable automatic 2023/2022 fallback when target year has no L1C")
    parser.add_argument(
        "--manifest", default=str(MANIFEST_IN),
        help=f"Input manifest (default: {MANIFEST_IN})"
    )
    args = parser.parse_args()

    # ── Load manifest ─────────────────────────────────────────────────────────
    if not Path(args.manifest).exists():
        print(f"ERROR: Manifest not found: {args.manifest}")
        print("Run: python scripts/ws5_sample_nonemitters.py")
        sys.exit(1)

    with open(args.manifest) as f:
        manifest = json.load(f)
    locations = manifest["locations"]

    if args.ids:
        locations = [l for l in locations if l["id"] in args.ids]
        if not locations:
            print(f"ERROR: No locations matched IDs: {args.ids}")
            sys.exit(1)

    fb_years = [] if args.no_fallback_years else FALLBACK_YEARS

    print("=" * 70)
    print("  WS5 Non-Emitter S2 Download")
    print(f"  Locations: {len(locations)}   Cloud limit: {args.max_cloud:.0f}%")
    print(f"  Search window: ±{args.search_window}d   Fallback years: {fb_years or 'disabled'}")
    print(f"  Output: {TILE_DIR}")
    if args.dry_run:
        print("  MODE: DRY RUN")
    print("=" * 70)

    # ── Authenticate ──────────────────────────────────────────────────────────
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

    # ── Process each location ─────────────────────────────────────────────────
    results = {}
    for loc in locations:
        result = process_location(
            loc, client, args.max_cloud, args.dry_run, args.max_scenes_per_site,
            search_window_days=args.search_window,
            fallback_years=fb_years,
        )
        results[loc["id"]] = result

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Download Summary:")
    counts = {}
    for loc_id, r in results.items():
        status = r.get("status", "?")
        counts[status] = counts.get(status, 0) + 1
        icon = {"ok": "✓", "cached": "✓", "dry_run": "○", "not_found": "—",
                "error": "✗", "skipped_hard_exclude": "⊘"}.get(status, "?")
        npy_name = Path(r["npy"]).name if r.get("npy") else r.get("would_download", "")[:50]
        cloud = r.get("cloud_cover")
        cloud_str = f"  cloud={cloud:.1f}%" if cloud is not None else ""
        print(f"  {icon} {loc_id:<14}  {status:<20}  {npy_name[:40]}{cloud_str}")

    print(f"\n  Totals: {counts}")
    print()

    # ── Write output manifest ─────────────────────────────────────────────────
    with open(MANIFEST_OUT, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Download manifest saved → %s", MANIFEST_OUT)

    n_ready = sum(1 for r in results.values() if r.get("status") in ("ok", "cached"))
    if n_ready > 0:
        print(f"  {n_ready} location(s) ready for inference.")
        print(f"  Next: python scripts/run_nonemitter_inference.py")


if __name__ == "__main__":
    main()
