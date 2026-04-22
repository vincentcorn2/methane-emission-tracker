"""
scripts/historical_backfill_download.py
========================================
WS4 — Download historical S2 L1C scenes (2020–2023) for confirmed emitter sites.

For each priority site and each year from 2020 to MAX_YEAR (inclusive), this
script searches the Copernicus CDSE catalog for the best cloud-free summer
acquisition and downloads it to data/npy_cache/ — the same directory that
apply_bitemporal_diff.py reads from.

After running this script, run:
    python apply_bitemporal_diff.py --sites weisweiler rybnik belchatow ...
    python scripts/historical_backfill_timeseries.py

to produce the multi-year S/C time series per site (WS4 deliverable: Gap 4
out-of-time validation).

Priority sites (confirmed emitters and key non-detections):
    weisweiler, rybnik, belchatow, lippendorf, neurath, boxberg, groningen, maasvlakte

Years: 2020, 2021, 2022, 2023  (2024 already in npy_cache)

Selection criterion: lowest cloud cover scene in months June–August.
Summer window is Jun–Aug rather than Jun–Sep to:
  (a) avoid the Weisweiler Sep terrain-CV problem (winter-wheat harvest complete,
      uniform stubble by July but mixed-crop rebound by late Sep),
  (b) maximise solar zenith margin.

Usage:
    conda activate methane
    export COPERNICUS_USER=your@email.com
    export COPERNICUS_PASS='yourpassword'

    python scripts/historical_backfill_download.py
    python scripts/historical_backfill_download.py --sites weisweiler rybnik
    python scripts/historical_backfill_download.py --years 2022 2023
    python scripts/historical_backfill_download.py --dry-run
    python scripts/historical_backfill_download.py --max-cloud 35

Estimated time: ~40 min per tile.  Use 'caffeinate -i python ...' on macOS.
"""

import os
import sys
import re
import json
import getpass
import logging
import tempfile
import shutil
import argparse
from pathlib import Path
from datetime import datetime, timedelta

# Matches the ACQUISITION date in S2 product names:
#   S2A_MSIL1C_20200601T095041_N0500_R079_T34UCB_20230402T070951.npy
#                ^^^^^^^^ acquisition             ^^^^^^^^ processing
_ACQ_YEAR_RE = re.compile(r"S2[AB]_MSIL1C_(\d{4})\d{4}T")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

Path("results_analysis").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("results_analysis/historical_backfill_download.log"),
    ],
)
log = logging.getLogger(__name__)

from src.ingestion.copernicus_client import CopernicusClient
from src.ingestion.preprocessing import safe_to_npy

# ── Site catalogue ─────────────────────────────────────────────────────────────
# Skip sites with tile_id=None (not yet commissioned).
SITES = {
    "weisweiler":   dict(lat=50.837, lon=6.322,  tile_id="T31UGS"),
    "rybnik":       dict(lat=50.135, lon=18.522, tile_id="T34UCA"),
    "belchatow":    dict(lat=51.266, lon=19.315, tile_id="T34UCB"),
    "lippendorf":   dict(lat=51.178, lon=12.378, tile_id="T33UUS"),
    "neurath":      dict(lat=51.038, lon=6.616,  tile_id="T32ULB"),
    "boxberg":      dict(lat=51.416, lon=14.565, tile_id="T33UVT"),
    "groningen":    dict(lat=53.252, lon=6.682,  tile_id="T31UGV"),
    "maasvlakte":   dict(lat=51.944, lon=4.067,  tile_id="T31UET"),
}

# Tiles already saturated with 2024 data — don't re-download 2024
ALREADY_CACHED_YEAR = 2024

# Search window: whole summer (1 Jun – 31 Aug each year)
SEARCH_MONTHS = [(6, 1, 8, 31)]   # (start_month, start_day, end_month, end_day)

DEFAULT_MAX_CLOUD  = 20.0
FALLBACK_MAX_CLOUD = 35.0
MIN_YEAR           = 2020
MAX_YEAR           = 2023   # 2024 already in npy_cache

NPY_CACHE    = Path("data/npy_cache")
DOWNLOAD_DIR = Path("data/downloads/historical")
MANIFEST_OUT = Path("results_analysis/historical_backfill_manifest.json")


def summer_window(year: int) -> tuple[str, str]:
    start = f"{year}-06-01T00:00:00.000Z"
    end   = f"{year}-08-31T23:59:59.999Z"
    return start, end


def bbox_wkt(lat: float, lon: float, margin: float = 0.25) -> str:
    return (
        f"POLYGON(({lon-margin} {lat-margin},{lon+margin} {lat-margin},"
        f"{lon+margin} {lat+margin},{lon-margin} {lat+margin},"
        f"{lon-margin} {lat-margin}))"
    )


def tile_already_cached(tile_id: str, year: int) -> list[Path]:
    """Return .npy files for this tile and year already in npy_cache.

    Matches on ACQUISITION year only — S2 filenames contain both an
    acquisition timestamp and a processing timestamp (e.g.
    S2A_MSIL1C_20200601T..._T34UCB_20230402T....npy).  A naive glob for
    '*2023*' would falsely match the 2023 processing date on a 2020 scene.
    """
    year_str = str(year)
    result = []
    for p in NPY_CACHE.glob(f"*{tile_id}*.npy"):
        if "MSIL1C" not in p.name or "_ref_" in p.name:
            continue
        m = _ACQ_YEAR_RE.search(p.name)
        if m and m.group(1) == year_str:
            result.append(p)
    return sorted(result)


def search_site_year(client: CopernicusClient, site: str, cfg: dict,
                     year: int, max_cloud: float) -> list:
    """Search CDSE for L1C scenes for a site in a given summer."""
    start, end = summer_window(year)
    wkt = bbox_wkt(cfg["lat"], cfg["lon"])
    tile_id = cfg["tile_id"]

    try:
        products = client.search_products(
            wkt_polygon=wkt,
            start_date=start,
            end_date=end,
            collection="SENTINEL-2",
            max_cloud_cover=max_cloud,
        )
    except Exception as e:
        log.error("  Search failed for %s/%d: %s", site, year, e)
        return []

    l1c = [p for p in products if tile_id in p.tile_id and "MSIL1C" in p.name]
    log.info("  %s %d: %d L1C scenes (of %d total) at cloud ≤ %.0f%%",
             site, year, len(l1c), len(products), max_cloud)
    return l1c


def download_and_convert(product, client: CopernicusClient,
                         site: str, tile_id: str) -> dict:
    """Download a single product and convert to .npy in npy_cache."""
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    NPY_CACHE.mkdir(parents=True, exist_ok=True)

    log.info("  Downloading %s ...", product.name[:70])
    zip_path = client.download_product(product, str(DOWNLOAD_DIR))
    if zip_path is None:
        raise RuntimeError(f"Download returned None for {product.name}")

    log.info("  Converting to .npy ...")
    extract_dir = tempfile.mkdtemp(prefix=f"s2_{site}_hist_")
    try:
        npy_path, meta_path = safe_to_npy(
            zip_path=zip_path,
            output_dir=str(NPY_CACHE),
            tile_id=tile_id,
            acquisition_date=product.acquisition_date,
            satellite=product.satellite,
            extract_dir=extract_dir,
        )
        log.info("  Saved: %s", Path(npy_path).name)
        return {
            "status":           "ok",
            "npy":              npy_path,
            "meta":             meta_path,
            "cloud_cover":      product.cloud_cover,
            "acquisition_date": product.acquisition_date,
            "product_name":     product.name,
        }
    finally:
        shutil.rmtree(extract_dir, ignore_errors=True)


def process_site_year(site: str, cfg: dict, year: int,
                      client: CopernicusClient,
                      max_cloud: float, dry_run: bool) -> dict:
    """Handle one (site, year) pair."""
    tile_id = cfg["tile_id"]

    log.info("── %s  %d  (%s) ──", site, year, tile_id)

    # Skip if already cached for this year
    cached = tile_already_cached(tile_id, year)
    if cached:
        log.info("  Already cached: %s", cached[0].name)
        return {
            "site": site, "year": year, "tile_id": tile_id,
            "status": "cached", "npy": str(cached[0]),
        }

    # Search
    products = search_site_year(client, site, cfg, year, max_cloud)
    if not products and max_cloud < FALLBACK_MAX_CLOUD:
        log.warning("  No L1C at %.0f%% — retrying at %.0f%%",
                    max_cloud, FALLBACK_MAX_CLOUD)
        products = search_site_year(client, site, cfg, year, FALLBACK_MAX_CLOUD)

    if not products:
        log.warning("  No L1C found for %s in summer %d", site, year)
        return {"site": site, "year": year, "tile_id": tile_id, "status": "not_found"}

    # Pick best (lowest cloud)
    products.sort(key=lambda p: (p.cloud_cover or 99, p.acquisition_date or ""))
    best = products[0]
    log.info("  Best: %s  cloud=%.1f%%  date=%s",
             best.name[:60], best.cloud_cover or 0, best.acquisition_date or "?")

    if dry_run:
        return {
            "site": site, "year": year, "tile_id": tile_id,
            "status": "dry_run",
            "would_download": best.name,
            "cloud_cover":    best.cloud_cover,
            "acquisition_date": best.acquisition_date,
        }

    # Check if already in npy_cache under original product name
    existing = list(NPY_CACHE.glob(f"*{best.name[:44]}*.npy"))
    if existing:
        log.info("  Already in npy_cache: %s", existing[0].name)
        return {
            "site": site, "year": year, "tile_id": tile_id,
            "status": "cached", "npy": str(existing[0]),
            "cloud_cover": best.cloud_cover,
            "acquisition_date": best.acquisition_date,
        }

    try:
        result = download_and_convert(best, client, site, tile_id)
        result.update({"site": site, "year": year, "tile_id": tile_id})
        return result
    except Exception as e:
        log.error("  Download failed: %s", e)
        return {
            "site": site, "year": year, "tile_id": tile_id,
            "status": "error", "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(
        description="WS4: download historical S2 L1C scenes for emitter sites (2020-2023)"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Search catalog only, no downloads")
    parser.add_argument("--sites", nargs="+", default=None,
                        choices=list(SITES.keys()),
                        help="Process only these sites (default: all)")
    parser.add_argument("--years", nargs="+", type=int, default=None,
                        help=f"Process only these years (default: {MIN_YEAR}-{MAX_YEAR})")
    parser.add_argument("--max-cloud", type=float, default=DEFAULT_MAX_CLOUD,
                        help=f"Max cloud cover %% (default: {DEFAULT_MAX_CLOUD})")
    args = parser.parse_args()

    sites_to_run = {k: v for k, v in SITES.items()
                    if args.sites is None or k in args.sites}
    years_to_run = args.years if args.years else list(range(MIN_YEAR, MAX_YEAR + 1))

    total_pairs = len(sites_to_run) * len(years_to_run)

    print("=" * 72)
    print("  WS4 Historical Backfill — S2 L1C Download")
    print(f"  Sites: {list(sites_to_run.keys())}")
    print(f"  Years: {years_to_run}  (2024 already cached)")
    print(f"  Cloud limit: {args.max_cloud:.0f}%   Fallback: {FALLBACK_MAX_CLOUD:.0f}%")
    print(f"  Total (site, year) pairs: {total_pairs}")
    print(f"  Estimated time (if all new): ~{total_pairs * 40 // 60}h {total_pairs * 40 % 60}min")
    if args.dry_run:
        print("  MODE: DRY RUN")
    print("=" * 72)

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

    # ── Load existing manifest (incremental resume) ───────────────────────────
    manifest: dict = {}
    if MANIFEST_OUT.exists():
        try:
            manifest = json.load(open(MANIFEST_OUT))
            already = sum(1 for r in manifest.values()
                          if r.get("status") in ("ok", "cached"))
            log.info("Resuming: %d (site, year) pairs already done", already)
        except Exception:
            pass

    # ── Process ───────────────────────────────────────────────────────────────
    for site, cfg in sites_to_run.items():
        for year in years_to_run:
            key = f"{site}_{year}"
            if key in manifest and manifest[key].get("status") in ("ok", "cached"):
                log.info("Skip %s — already done", key)
                continue
            result = process_site_year(site, cfg, year, client, args.max_cloud, args.dry_run)
            manifest[key] = result
            # Write after every tile so progress is preserved
            if not args.dry_run:
                with open(MANIFEST_OUT, "w") as f:
                    json.dump(manifest, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  Download Summary:")
    counts = {}
    for key, r in sorted(manifest.items()):
        status = r.get("status", "?")
        counts[status] = counts.get(status, 0) + 1
        icon = {"ok": "✓", "cached": "✓", "dry_run": "○",
                "not_found": "—", "error": "✗"}.get(status, "?")
        site  = r.get("site", key.split("_")[0])
        year  = r.get("year", "")
        cloud = r.get("cloud_cover")
        npy_name = Path(r["npy"]).name[:45] if r.get("npy") else r.get("would_download", "")[:45]
        cloud_str = f"  cloud={cloud:.1f}%" if cloud is not None else ""
        print(f"  {icon} {site:<14} {year}  {status:<12}  {npy_name}{cloud_str}")

    print(f"\n  Totals: {counts}")
    n_ready = sum(1 for r in manifest.values() if r.get("status") in ("ok", "cached"))
    print(f"\n  {n_ready}/{total_pairs} (site, year) pairs ready for inference.")
    print()

    with open(MANIFEST_OUT, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("Manifest saved → %s", MANIFEST_OUT)

    if n_ready > 0 and not args.dry_run:
        print("  Next steps:")
        print("    python apply_bitemporal_diff.py --sites weisweiler rybnik belchatow lippendorf neurath boxberg groningen maasvlakte")
        print("    python scripts/historical_backfill_timeseries.py")


if __name__ == "__main__":
    main()
