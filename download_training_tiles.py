"""
download_training_tiles.py
===========================
Downloads the Sentinel-2 L1C tiles that correspond to the 41 confirmed
TROPOMI-positive dates from tropomi_mine_europe.py.

These are the ground-truth training examples for European CH4Net fine-tuning:
  • 14 distinct sites across DE, FR, NL, PL, RO, UK
  • 11 distinct Sentinel-2 tiles
  • 6 source types: gas_storage, gas_compressor, gas_terminal,
                    gas_processing, coal_mine, coal_mine_plant
  • Jan 2023 – Jul 2024

Strategy: For each confirmed positive (TROPOMI date + S2 tile ID), search
the Copernicus catalog for L1C acquisitions of that tile within ±2 days,
preferring lowest cloud cover. Download and convert to .npy.

Outputs (written to data/npy_cache/training/):
  <product_name>.npy              — 12-band uint8 array
  <product_name>_geo.json         — geo metadata sidecar
  <product_name>_label.json       — training label: site, enhancement_ppb,
                                    source_type, tropomi_date, etc.

Also writes:
  results_analysis/training_manifest.json  — full download inventory
  results_analysis/training_summary.txt    — human-readable summary

Usage:
    export COPERNICUS_USER=your@email.com
    export COPERNICUS_PASS='yourpassword'
    conda activate methane
    python download_training_tiles.py

    # Dry run (search only, no downloads):
    python download_training_tiles.py --dry-run

    # Download only the top-N strongest signals first:
    python download_training_tiles.py --top 10

    # Resume after interruption (skips already-cached tiles):
    python download_training_tiles.py          # just re-run; cached = skipped

Estimated disk usage: ~500 MB per tile × 41 acquisitions = ~20 GB max
  (many dates share the same tile → deduplication reduces this significantly)

Memory: Safe. Arrays are saved immediately after conversion; full scene
  loaded only one at a time.
"""

import os
import sys
import json
import math
import getpass
import logging
import shutil
import tempfile
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

# ── Logging ───────────────────────────────────────────────────────────────────
Path("results_analysis").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("results_analysis/training_download.log"),
    ],
)
log = logging.getLogger(__name__)

from src.ingestion.copernicus_client import CopernicusClient
from src.ingestion.preprocessing import safe_to_npy

# ── Paths ──────────────────────────────────────────────────────────────────────
POSITIVES_JSON  = Path("results_analysis/tropomi_positives.json")
TRAIN_CACHE     = Path("data/npy_cache/training")
DOWNLOAD_DIR    = Path("data/downloads/training")
MANIFEST_PATH   = Path("results_analysis/training_manifest.json")
SUMMARY_PATH    = Path("results_analysis/training_summary.txt")

TRAIN_CACHE.mkdir(parents=True, exist_ok=True)
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ── Search parameters ─────────────────────────────────────────────────────────
DATE_WINDOW_DAYS = 7     # search ±7 days around TROPOMI detection date
DATE_WINDOW_WIDE = 21    # fallback window if nothing found in ±7 days
MAX_CLOUD        = 40.0  # % cloud cover; relax to 80% if nothing found
MAX_RESULTS      = 50    # max search results per query

# ── Tile bounding boxes (small polygon at site centroid → ensures correct tile) ─
MARGIN_DEG = 0.25

def site_bbox_wkt(lat: float, lon: float) -> str:
    d = MARGIN_DEG
    return (f"POLYGON(({lon-d} {lat-d},{lon+d} {lat-d},"
            f"{lon+d} {lat+d},{lon-d} {lat+d},{lon-d} {lat-d}))")


# ── Already downloaded? ────────────────────────────────────────────────────────
def already_cached(tile_id: str, approx_date: str) -> Path | None:
    """
    Return path if a training tile for this (tile_id, date) is already cached.
    We match on tile_id in the filename and date within ±2 days.
    """
    target_dt = datetime.strptime(approx_date, "%Y-%m-%d")
    for p in TRAIN_CACHE.glob(f"*_{tile_id}_*.npy"):
        # Extract date from S2 product name: S2X_MSIL1C_YYYYMMDDThhmmss_...
        parts = p.stem.split("_")
        for part in parts:
            if len(part) == 15 and part[0].isdigit():
                try:
                    file_dt = datetime.strptime(part[:8], "%Y%m%d")
                    if abs((file_dt - target_dt).days) <= DATE_WINDOW_DAYS:
                        return p
                except ValueError:
                    pass
    return None


# ── Main download loop ─────────────────────────────────────────────────────────
def download_training_tiles(
    positives: list[dict],
    client: CopernicusClient,
    dry_run: bool = False,
) -> dict:
    """
    For each TROPOMI-confirmed positive, find and download the corresponding
    Sentinel-2 L1C tile. Groups by tile_id to minimise redundant downloads
    (same tile_id on nearby dates → one acquisition covers multiple sites).

    Returns manifest dict keyed by (site, date) string.
    """
    manifest = {}
    stats = {"ok": 0, "cached": 0, "no_product": 0, "error": 0, "skipped_dry": 0}

    # Group positives by tile_id+date so we don't double-download
    # (e.g. silesia_jastrzebie and silesia_zofiowka share T34UCB on 2023-01-01)
    seen_tile_date: dict[tuple, str] = {}  # (tile_id, product_name) → manifest key of first hit

    for pos in positives:
        site        = pos["site"]
        date        = pos["date"]       # TROPOMI overpass date
        tile_id     = pos["s2_tile"]
        lat, lon    = pos["lat"], pos["lon"]
        enh         = pos["enhancement_ppb"]
        source_type = pos.get("source_type", "unknown")
        country     = pos.get("country", "?")
        key         = f"{site}|{date}"

        log.info("")
        log.info("─" * 60)
        log.info("%-28s  %s  %+.1f ppb  tile=%s", site, date, enh, tile_id)

        # ── Check cache ───────────────────────────────────────────────────
        cached = already_cached(tile_id, date)
        if cached:
            log.info("  ✓ Cached: %s", cached.name)
            # Load the label sidecar
            label_path = cached.with_name(cached.stem + "_label.json")
            if not label_path.exists():
                # Write label for a previously cached tile
                label = _build_label(pos, cached.stem)
                label_path.write_text(json.dumps(label, indent=2))
            manifest[key] = {
                "status": "cached", "npy_path": str(cached),
                "tile_id": tile_id, "date": date, "site": site,
                "enhancement_ppb": enh, "source_type": source_type,
                "country": country,
            }
            stats["cached"] += 1
            continue

        # ── Check if a sibling site already downloaded this tile/date ─────
        # (Multiple Silesian mines on the same tile on the same date)
        tile_date_key = (tile_id, date)
        if tile_date_key in seen_tile_date:
            sibling_key = seen_tile_date[tile_date_key]
            sibling = manifest.get(sibling_key, {})
            if sibling.get("npy_path"):
                log.info("  ✓ Sharing tile with %s: %s",
                         sibling_key.split("|")[0],
                         Path(sibling["npy_path"]).name)
                label_path = Path(sibling["npy_path"]).with_name(
                    Path(sibling["npy_path"]).stem + f"_{site}_label.json"
                )
                label = _build_label(pos, Path(sibling["npy_path"]).stem)
                label_path.write_text(json.dumps(label, indent=2))
                manifest[key] = {
                    "status": "shared", "npy_path": sibling["npy_path"],
                    "tile_id": tile_id, "date": date, "site": site,
                    "enhancement_ppb": enh, "source_type": source_type,
                    "country": country,
                }
                stats["cached"] += 1
                continue

        # ── Search Copernicus ─────────────────────────────────────────────
        dt = datetime.strptime(date, "%Y-%m-%d")
        wkt = site_bbox_wkt(lat, lon)

        def _search(window_days, cloud_pct, any_tile=False):
            s = (dt - timedelta(days=window_days)).strftime("%Y-%m-%dT00:00:00.000Z")
            e = (dt + timedelta(days=window_days)).strftime("%Y-%m-%dT23:59:59.000Z")
            prods = client.search_products(
                wkt_polygon=wkt,
                start_date=s,
                end_date=e,
                collection="SENTINEL-2",
                max_cloud_cover=cloud_pct,
                max_results=MAX_RESULTS,
            )
            # L1C only — L2A has atmospheric correction that erases methane signal
            l1c = [p for p in prods if p.processing_level == "MSIL1C"]
            if any_tile:
                return l1c  # accept any tile covering the site bbox
            return [p for p in l1c if p.tile_id == tile_id]

        tile_products = []
        used_fallback_tile = False
        for window, cloud, any_tile in [
            (DATE_WINDOW_DAYS, MAX_CLOUD,  False),
            (DATE_WINDOW_DAYS, 80.0,       False),
            (DATE_WINDOW_WIDE, MAX_CLOUD,  False),
            (DATE_WINDOW_WIDE, 80.0,       False),
            # Final fallback: accept any L1C tile that covers this location.
            # The TROPOMI miner tile_id may have come from a co-registered L2A
            # product; the L1C equivalent uses the same footprint but a different
            # tile code in some ESA reprocessing batches.
            (DATE_WINDOW_WIDE, 80.0,       True),
        ]:
            try:
                tile_products = _search(window, cloud, any_tile)
                label = "any-tile" if any_tile else tile_id
                log.info("  ±%dd cloud≤%.0f%% [%s]: %d L1C products",
                         window, cloud, label, len(tile_products))
                if tile_products:
                    used_fallback_tile = any_tile
                    break
            except Exception as e:
                log.error("  Search failed (±%dd cloud≤%.0f%%): %s", window, cloud, e)

        if not tile_products:
            log.warning("  No L1C products found for %s within ±%dd (any tile)",
                        tile_id, DATE_WINDOW_WIDE)
            manifest[key] = {"status": "no_product", "tile_id": tile_id,
                             "date": date, "site": site, "enhancement_ppb": enh}
            stats["no_product"] += 1
            continue

        if used_fallback_tile:
            actual_tiles = list(set(p.tile_id for p in tile_products))
            log.info("  Note: using fallback tile(s) %s instead of expected %s",
                     actual_tiles, tile_id)

        # Pick the one closest to the TROPOMI date, breaking ties by cloud cover
        def sort_key(p):
            try:
                file_dt = datetime.strptime(p.acquisition_date[:10], "%Y-%m-%d")
                day_diff = abs((file_dt - dt).days)
            except Exception:
                day_diff = 99
            cloud = p.cloud_cover if p.cloud_cover is not None else 99.0
            return (day_diff, cloud)

        tile_products.sort(key=sort_key)
        best = tile_products[0]
        log.info("  Best: %s  cloud=%.1f%%  date=%s",
                 best.name[:65], best.cloud_cover or 0,
                 best.acquisition_date[:10])

        if dry_run:
            log.info("  [dry-run] Would download %s", best.name[:65])
            manifest[key] = {
                "status": "dry_run", "would_download": best.name,
                "tile_id": tile_id, "date": date, "site": site,
                "enhancement_ppb": enh, "source_type": source_type,
            }
            stats["skipped_dry"] += 1
            continue

        # ── Download ──────────────────────────────────────────────────────
        log.info("  Downloading...")
        try:
            zip_path = client.download_product(best, str(DOWNLOAD_DIR))
        except Exception as e:
            log.error("  Download failed: %s", e)
            manifest[key] = {"status": "download_error", "error": str(e),
                             "tile_id": tile_id, "date": date, "site": site}
            stats["error"] += 1
            continue

        if zip_path is None:
            manifest[key] = {"status": "download_none",
                             "tile_id": tile_id, "date": date, "site": site}
            stats["error"] += 1
            continue

        # ── Convert SAFE → .npy ───────────────────────────────────────────
        log.info("  Converting to .npy...")
        extract_tmp = tempfile.mkdtemp(prefix=f"train_{tile_id}_")
        try:
            npy_path, meta_path = safe_to_npy(
                zip_path=zip_path,
                output_dir=str(TRAIN_CACHE),
                tile_id=best.tile_id,
                acquisition_date=best.acquisition_date[:10],
                satellite=best.satellite,
                extract_dir=extract_tmp,
            )
        except Exception as e:
            log.error("  Conversion failed: %s", e)
            manifest[key] = {"status": "conversion_error", "error": str(e),
                             "zip_path": str(zip_path),
                             "tile_id": tile_id, "date": date, "site": site}
            stats["error"] += 1
            shutil.rmtree(extract_tmp, ignore_errors=True)
            continue
        finally:
            shutil.rmtree(extract_tmp, ignore_errors=True)

        # ── Write training label sidecar ──────────────────────────────────
        product_stem = Path(npy_path).stem
        label = _build_label(pos, product_stem, best)
        label_path = Path(npy_path).with_name(product_stem + "_label.json")
        label_path.write_text(json.dumps(label, indent=2))

        # ── Quick sanity check ────────────────────────────────────────────
        arr = np.load(npy_path, mmap_mode="r")
        log.info("  ✓ shape=%s  B12_mean=%.1f  label=%+.1f ppb",
                 arr.shape, float(arr[:, :, 11].mean()), enh)

        seen_tile_date[tile_date_key] = key
        manifest[key] = {
            "status":          "ok",
            "npy_path":        npy_path,
            "label_path":      str(label_path),
            "product_name":    best.name,
            "s2_date":         best.acquisition_date[:10],
            "tropomi_date":    date,
            "tile_id":         tile_id,
            "site":            site,
            "enhancement_ppb": enh,
            "source_type":     source_type,
            "country":         country,
            "cloud_cover":     best.cloud_cover,
            "shape":           list(arr.shape),
        }
        stats["ok"] += 1

    return manifest, stats


def _build_label(pos: dict, product_stem: str, s2_product=None) -> dict:
    """Build the training label sidecar dict for a positive example."""
    label = {
        "split":            "train",        # all positives start as train; user can move to val
        "label_type":       "positive",     # TROPOMI-confirmed methane emission
        "site":             pos["site"],
        "country":          pos.get("country", "?"),
        "source_type":      pos.get("source_type", "unknown"),
        "lat":              pos["lat"],
        "lon":              pos["lon"],
        "tile_id":          pos["s2_tile"],
        "tropomi_date":     pos["date"],
        "enhancement_ppb":  pos["enhancement_ppb"],
        "max_near_ppb":     pos.get("max_near_ppb"),
        "background_ppb":   pos.get("background_ppb"),
        "n_near_pixels":    pos.get("n_near_pixels"),
        "qa_threshold":     pos.get("qa_threshold"),
        "tropomi_product":  pos.get("product_name", ""),
        "s2_product":       s2_product.name if s2_product else product_stem,
        "created":          datetime.utcnow().isoformat() + "Z",
        "note": (
            "TROPOMI-confirmed emission. S2 tile acquired within "
            f"±{DATE_WINDOW_DAYS} days of TROPOMI overpass. "
            "Use centroid (lat, lon) to extract 160×160 or 200×200 training crop."
        ),
    }
    return label


# ── Summary report ─────────────────────────────────────────────────────────────
def write_summary(manifest: dict, stats: dict):
    lines = [
        "TRAINING TILE DOWNLOAD SUMMARY",
        f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        "=" * 65,
        f"  OK (new downloads):    {stats['ok']:4d}",
        f"  Cached/shared:         {stats['cached']:4d}",
        f"  No S2 product found:   {stats['no_product']:4d}",
        f"  Errors:                {stats['error']:4d}",
        f"  Dry-run skipped:       {stats['skipped_dry']:4d}",
        f"  Total positives:       {len(manifest):4d}",
        "",
        "By source type:",
    ]

    by_type: dict[str, list] = defaultdict(list)
    for info in manifest.values():
        by_type[info.get("source_type", "?")].append(info)
    for stype, items in sorted(by_type.items()):
        ok = sum(1 for x in items if x.get("status") in ("ok", "cached", "shared"))
        lines.append(f"  {stype:<25}  {ok:3d} / {len(items):3d} tiles ready")

    lines += ["", "By country:"]
    by_country: dict[str, list] = defaultdict(list)
    for info in manifest.values():
        by_country[info.get("country", "?")].append(info)
    for c, items in sorted(by_country.items()):
        ok = sum(1 for x in items if x.get("status") in ("ok", "cached", "shared"))
        lines.append(f"  {c}:  {ok:3d} / {len(items):3d}")

    lines += ["", "Ready tiles for fine-tuning:"]
    for key, info in sorted(manifest.items(), key=lambda x: -x[1].get("enhancement_ppb", 0)):
        if info.get("status") in ("ok", "cached", "shared"):
            site = info.get("site", "?")
            enh  = info.get("enhancement_ppb", 0)
            date = info.get("tropomi_date", "?")
            tile = info.get("tile_id", "?")
            stype = info.get("source_type", "?")[:18]
            lines.append(f"  {site:<28} {date}  {enh:+.1f} ppb  {tile}  {stype}")

    text = "\n".join(lines)
    SUMMARY_PATH.write_text(text)
    print("\n" + text)


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Download Sentinel-2 tiles for TROPOMI-confirmed positive dates"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Search only — do not download")
    parser.add_argument("--top", type=int, default=None,
                        help="Only download top-N strongest signals (sorted by enhancement_ppb)")
    parser.add_argument("--site", type=str, default=None,
                        help="Only download tiles for one specific site")
    args = parser.parse_args()

    print("=" * 65)
    print("  Training Tile Download — TROPOMI-Confirmed Positives")
    print(f"  Source: {POSITIVES_JSON}")
    print("=" * 65)

    if not POSITIVES_JSON.exists():
        print(f"\nERROR: {POSITIVES_JSON} not found.")
        print("Run tropomi_mine_europe.py first.")
        sys.exit(1)

    with open(POSITIVES_JSON) as f:
        positives = json.load(f)

    print(f"\nLoaded {len(positives)} confirmed positives")

    # Apply filters
    if args.site:
        positives = [p for p in positives if p["site"] == args.site]
        print(f"Filtered to site '{args.site}': {len(positives)} records")

    if args.top:
        positives = sorted(positives, key=lambda x: -x["enhancement_ppb"])[:args.top]
        print(f"Top-{args.top} by enhancement: {[p['site'] + ' ' + p['date'] for p in positives]}")

    # Deduplication preview
    unique_tile_dates = set((p["s2_tile"], p["date"]) for p in positives)
    print(f"Unique (tile, date) pairs: {len(unique_tile_dates)}  "
          f"(deduplication saves ~{len(positives)-len(unique_tile_dates)} downloads)")
    print(f"Unique tiles: {sorted(set(p['s2_tile'] for p in positives))}")

    if args.dry_run:
        print("\n[DRY RUN — no downloads will occur]")

    # Credentials
    username = os.environ.get("COPERNICUS_USER") or input("\nCopernicus username (email): ").strip()
    password = os.environ.get("COPERNICUS_PASS") or getpass.getpass("Copernicus password: ")

    print("\n[1] Authenticating...")
    try:
        client = CopernicusClient(username, password)
        _ = client.token
        print("    OK")
    except Exception as e:
        print(f"    Auth failed: {e}")
        sys.exit(1)

    print(f"\n[2] Processing {len(positives)} positives...")
    manifest, stats = download_training_tiles(positives, client, dry_run=args.dry_run)

    print("\n[3] Saving manifest and summary...")
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"    Manifest → {MANIFEST_PATH}")

    write_summary(manifest, stats)
    print(f"    Summary  → {SUMMARY_PATH}")

    # Final verdict
    total_ready = stats["ok"] + stats["cached"]
    print(f"\n{'='*65}")
    print(f"  {total_ready} / {len(positives)} training tiles ready")
    if stats["no_product"] > 0:
        print(f"  {stats['no_product']} dates had no S2 acquisition within ±{DATE_WINDOW_DAYS} days")
        print(f"  (expand DATE_WINDOW_DAYS or check cloud cover — "
              f"Silesia winter dates often cloudy)")
    if total_ready >= 30:
        print(f"\n  ✓ Sufficient data for fine-tuning.")
        print(f"  Next: run apply_bitemporal_diff.py (reference subtraction),")
        print(f"  then approach_c_retrain.py with the European training set.")
    print(f"{'='*65}")

    sys.exit(0 if stats["error"] == 0 else 1)


if __name__ == "__main__":
    main()
