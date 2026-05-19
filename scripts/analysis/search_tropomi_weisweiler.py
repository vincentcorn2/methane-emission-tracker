"""
scripts/search_tropomi_weisweiler.py
=====================================
Broad TROPOMI S5P L2 CH4 search over Weisweiler (50.837°N, 6.322°E) for the
full summer 2024 window (2024-06-01 – 2024-09-30).

Unlike validate_tropomi.py (which only checks the 4 S2 acquisition dates),
this script systematically checks every date in the window, so we can find
any day TROPOMI saw a Weisweiler methane plume — regardless of whether S2
was acquiring on that date.

Strategy (two-phase):
  Phase 1  --catalog-only   (default, fast)
           Queries the Copernicus OData catalog for the EU daytime orbit
           on each date.  Prints a date × orbit table with estimated swath
           coverage (from product bounding box intersection).  No downloads.

  Phase 2  --download       Fetches the top candidates sorted by likelihood
           of capturing a real plume (nearest to Sep-18 detection, then all
           others).  Extracts ΔXCH4 from each file.  Downloads stop as soon
           as --max-downloads is reached or the window is exhausted.

Output:
  results_analysis/tropomi_weisweiler_search.json
    date → { orbit_name, swath_covers_site, enhancement_ppb, ... }

Usage:
  conda activate methane
  export COPERNICUS_USER=your@email.com
  export COPERNICUS_PASS='yourpassword'

  # Phase 1: catalog only (no download, fast)
  python scripts/search_tropomi_weisweiler.py --catalog-only

  # Phase 2: download up to 15 orbits, extract ΔXCH4
  python scripts/search_tropomi_weisweiler.py --download --max-downloads 15

  # Re-run extraction from previously downloaded files only
  python scripts/search_tropomi_weisweiler.py --no-download

  # Focus on a specific date range (e.g., around the Sep-18 detection)
  python scripts/search_tropomi_weisweiler.py --download --start 2024-09-10 --end 2024-09-25
"""

import os
import sys
import json
import getpass
import logging
import argparse
import re
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Logging ────────────────────────────────────────────────────────────────────
Path("results_analysis").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("results_analysis/tropomi_weisweiler_search.log"),
    ],
)
log = logging.getLogger(__name__)

MISSING = []
try:
    import netCDF4 as nc
except ImportError:
    MISSING.append("netCDF4")
try:
    import requests  # noqa: F401 (needed by CopernicusClient)
except ImportError:
    MISSING.append("requests")

if MISSING:
    print(f"Missing packages: {', '.join(MISSING)}")
    print(f"Install with: pip install {' '.join(MISSING)} --break-system-packages")
    sys.exit(1)

from src.ingestion.copernicus_client import CopernicusClient

# ── Constants ─────────────────────────────────────────────────────────────────
SITE_LAT = 50.837
SITE_LON = 6.322
SITE_NAME = "weisweiler"

TROPOMI_DIR   = Path("data/downloads/tropomi")
RESULTS_JSON  = Path("results_analysis/tropomi_weisweiler_search.json")

SEARCH_START  = "2024-06-01"
SEARCH_END    = "2024-09-30"

# Max ±distance from site for the nearest valid pixel to be counted as "covered"
MAX_DIST_DEG  = 0.15   # ~16 km

# Enhancement threshold for reporting
ENH_THRESHOLD = 5.0    # ppb

# Background annulus parameters (consistent with validate_tropomi.py)
BG_INNER_DEG  = 0.25
BG_OUTER_DEG  = 1.00
BG_PERCENTILE = 20
QA_THRESHOLD  = 0.5

# EU daytime overpass window (hour, UTC)
EU_OVP_START  = 11
EU_OVP_END    = 14

# Detection date — sort downloads so we check this date and neighbours first
DETECTION_DATE = datetime(2024, 9, 18)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _orbit_start_dt(name: str) -> datetime | None:
    m = re.search(r"CH4[_]{4}(\d{8}T\d{6})", name)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y%m%dT%H%M%S")
        except ValueError:
            pass
    return None


def _is_eu_daytime(name: str) -> bool:
    t = _orbit_start_dt(name)
    if t is None:
        return True  # unknown — include as candidate
    return EU_OVP_START <= t.hour < EU_OVP_END


def site_bbox_wkt(margin: float = 0.20) -> str:
    lat, lon = SITE_LAT, SITE_LON
    return (f"POLYGON(({lon-margin} {lat-margin},{lon+margin} {lat-margin},"
            f"{lon+margin} {lat+margin},{lon-margin} {lat+margin},"
            f"{lon-margin} {lat-margin}))")


def search_one_day(client: CopernicusClient, date_str: str) -> list:
    """Return EU daytime S5P L2 CH4 products over Weisweiler bbox for one date."""
    start = f"{date_str}T00:00:00.000Z"
    end   = f"{date_str}T23:59:59.000Z"
    wkt   = site_bbox_wkt()

    params = {
        "$filter": (
            f"Collection/Name eq 'SENTINEL-5P' and "
            f"OData.CSC.Intersects(area=geography'SRID=4326;{wkt}') and "
            f"ContentDate/Start ge {start} and "
            f"ContentDate/Start le {end} and "
            f"contains(Name,'L2__CH4___')"
        ),
        "$top": 20,
        "$orderby": "ContentDate/Start asc",
    }

    headers = {"Authorization": f"Bearer {client.token}"}
    r = client._session.get(client._catalog_url, params=params,
                            headers=headers, timeout=60)
    if r.status_code == 401:
        client._refresh_token()
        headers = {"Authorization": f"Bearer {client.token}"}
        r = client._session.get(client._catalog_url, params=params,
                                headers=headers, timeout=60)
    r.raise_for_status()

    products = []
    for item in r.json().get("value", []):
        name = item.get("Name", "")
        if "L2__CH4___" in name and "OFFL" in name and _is_eu_daytime(name):
            products.append({
                "id":    item["Id"],
                "name":  name,
                "start": item.get("ContentDate", {}).get("Start", ""),
            })
    return products


def download_orbit(product: dict, client: CopernicusClient) -> Path | None:
    """Download a TROPOMI orbit .nc file. Returns path or None."""
    TROPOMI_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = product["name"].replace(" ", "_")
    if not safe_name.endswith(".nc"):
        safe_name += ".nc"
    out = TROPOMI_DIR / safe_name
    if out.exists():
        log.info("    Cached: %s", out.name)
        return out

    url = (
        f"https://catalogue.dataspace.copernicus.eu/odata/v1/"
        f"Products({product['id']})/$value"
    )
    headers = {"Authorization": f"Bearer {client.token}"}
    try:
        head = client._session.head(url, headers=headers, allow_redirects=False)
        redir = head.headers.get("Location", url) if head.status_code in (301,302,303,307,308) else url
        log.info("    Downloading %s (~400–600 MB)...", product["name"][:65])
        with client._session.get(redir, headers=headers, stream=True, timeout=600) as resp:
            resp.raise_for_status()
            size = 0
            with open(out, "wb") as f:
                for chunk in resp.iter_content(65536):
                    if chunk:
                        f.write(chunk)
                        size += len(chunk)
        log.info("    Saved: %s (%.1f MB)", out.name, size / 1e6)
        return out
    except Exception as e:
        log.error("    Download failed: %s", e)
        if out.exists():
            out.unlink()
        return None


def extract_xch4(nc_path: Path) -> dict:
    """
    Extract XCH4 bias-corrected enhancement at Weisweiler from a TROPOMI .nc file.
    Returns dict with keys: site_xch4, bg_xch4, enhancement, qa_at_site, n_bg_pixels,
    dist_to_site_deg — or {"error": "..."} if the site is not in the swath.
    """
    try:
        ds = nc.Dataset(str(nc_path), "r")
    except Exception as e:
        return {"error": f"open failed: {e}"}

    try:
        grp  = ds["PRODUCT"]
        lats = np.array(grp["latitude"][:])
        lons = np.array(grp["longitude"][:])
        xch4_raw = np.array(grp["methane_mixing_ratio_bias_corrected"][:])
        qa_raw   = np.array(grp["qa_value"][:])
        ds.close()

        # Squeeze time dimension
        xch4 = xch4_raw[0] if xch4_raw.ndim == 3 else xch4_raw
        qa   = qa_raw[0]   if qa_raw.ndim   == 3 else qa_raw
        if lats.ndim == 3:
            lats, lons = lats[0], lons[0]

        xch4 = xch4.astype(np.float32)
        xch4[xch4 > 1e29] = np.nan

        valid    = (qa >= QA_THRESHOLD) & np.isfinite(xch4)
        dist_deg = np.sqrt((lats - SITE_LAT)**2 + (lons - SITE_LON)**2)

        dist_masked = np.where(valid, dist_deg, np.inf)
        flat_idx    = int(np.argmin(dist_masked))
        si, sj      = np.unravel_index(flat_idx, dist_masked.shape)

        min_dist = float(dist_masked[si, sj])
        if min_dist > MAX_DIST_DEG:
            return {"error": f"site not in swath (nearest valid: {min_dist:.3f}°)"}

        site_val = float(xch4[si, sj])
        site_qa  = float(qa[si, sj])

        bg_mask   = (dist_deg >= BG_INNER_DEG) & (dist_deg <= BG_OUTER_DEG) & valid
        bg_pixels = xch4[bg_mask]
        if len(bg_pixels) < 10:
            return {"error": f"too few background pixels ({len(bg_pixels)})"}

        bg_val = float(np.percentile(bg_pixels, BG_PERCENTILE))
        enh    = round(site_val - bg_val, 2)

        return {
            "site_xch4":        round(site_val, 2),
            "bg_xch4":          round(bg_val, 2),
            "enhancement":      enh,
            "qa_at_site":       round(site_qa, 3),
            "n_bg_pixels":      int(bg_mask.sum()),
            "dist_to_site_deg": round(min_dist, 4),
        }
    except Exception as e:
        try:
            ds.close()
        except Exception:
            pass
        return {"error": str(e)}


# ── Date sequence helpers ─────────────────────────────────────────────────────

def date_range(start_str: str, end_str: str):
    """Yield YYYY-MM-DD strings from start to end inclusive."""
    d = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d")
    while d <= end:
        yield d.strftime("%Y-%m-%d")
        d += timedelta(days=1)


def sorted_by_priority(dates: list) -> list:
    """
    Sort dates so that days closest to the Sep-18 detection come first.
    This ensures the most informative downloads happen first when capped by
    --max-downloads.
    """
    def _priority(d):
        dt = datetime.strptime(d, "%Y-%m-%d")
        return abs((dt - DETECTION_DATE).days)
    return sorted(dates, key=_priority)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Broad TROPOMI S5P CH4 search over Weisweiler — summer 2024"
    )
    parser.add_argument("--catalog-only", action="store_true",
                        help="Only query the catalog; print available orbits without downloading")
    parser.add_argument("--download", action="store_true",
                        help="Download orbit files and extract ΔXCH4")
    parser.add_argument("--no-download", action="store_true",
                        help="Extract from already-cached files only (no network)")
    parser.add_argument("--max-downloads", type=int, default=20,
                        help="Maximum number of orbit files to download (default: 20)")
    parser.add_argument("--start", default=SEARCH_START,
                        help=f"Start date YYYY-MM-DD (default: {SEARCH_START})")
    parser.add_argument("--end", default=SEARCH_END,
                        help=f"End date YYYY-MM-DD (default: {SEARCH_END})")
    args = parser.parse_args()

    if not any([args.catalog_only, args.download, args.no_download]):
        print("Specify one of: --catalog-only, --download, --no-download")
        parser.print_help()
        sys.exit(1)

    all_dates = list(date_range(args.start, args.end))
    prioritised = sorted_by_priority(all_dates)

    print("=" * 70)
    print(f"  TROPOMI Weisweiler Search — {args.start} to {args.end}")
    print(f"  Site: {SITE_LAT}°N, {SITE_LON}°E  (Weisweiler / T31UGS)")
    if args.download:
        print(f"  Mode: download (max {args.max_downloads} orbits, priority-sorted)")
    elif args.catalog_only:
        print("  Mode: catalog only (no download)")
    else:
        print("  Mode: extract from cache only")
    print("=" * 70)

    # ── Authenticate (skip for --no-download) ─────────────────────────────────
    client = None
    if args.download or args.catalog_only:
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

    results = {}
    n_downloaded = 0
    n_covered    = 0
    dates_with_enhancement = []

    # ── Phase 1: catalog search ────────────────────────────────────────────────
    if args.catalog_only or args.download:
        log.info("[2] Querying catalog for %d dates...", len(all_dates))
        catalog = {}    # date → list of product dicts
        for date in sorted(all_dates):    # chronological for catalog scan
            try:
                products = search_one_day(client, date)
            except Exception as e:
                log.warning("  %s  catalog error: %s", date, e)
                products = []

            if products:
                catalog[date] = products
                log.info("  %s  →  %d EU daytime orbit(s): %s",
                         date, len(products), products[0]["name"][:50])
            else:
                log.debug("  %s  →  no orbit found", date)

        print(f"\nCatalog: {len(catalog)}/{len(all_dates)} dates have a EU daytime S5P orbit over Weisweiler bbox")

        if args.catalog_only:
            print("\nTop candidates (sorted by proximity to Sep-18 detection):")
            for date in prioritised:
                if date in catalog:
                    prods = catalog[date]
                    for p in prods:
                        dt = _orbit_start_dt(p["name"])
                        utc = dt.strftime("%H:%M") if dt else "?"
                        print(f"  {date}  {utc} UTC  {p['name'][:65]}")
            print(f"\nRun with --download --max-downloads N to fetch orbit files and extract ΔXCH4")
            return

    # ── Phase 2: download + extract ───────────────────────────────────────────

    # For --no-download: enumerate cached .nc files
    if args.no_download:
        cached_files = sorted(TROPOMI_DIR.glob("*L2__CH4___*.nc")) if TROPOMI_DIR.exists() else []
        log.info("[2] Found %d cached TROPOMI files", len(cached_files))
        for nc_path in cached_files:
            dt  = _orbit_start_dt(nc_path.name)
            key = dt.strftime("%Y-%m-%d") if dt else nc_path.stem[:10]
            log.info("  Processing %s ...", nc_path.name[:65])
            extraction = extract_xch4(nc_path)
            enh = extraction.get("enhancement")
            if "error" in extraction:
                log.info("    → %s", extraction["error"])
            else:
                log.info("    → ΔXCH4=%+.1f ppb  qa=%.2f  n_bg=%d",
                         enh, extraction["qa_at_site"], extraction["n_bg_pixels"])
                n_covered += 1
                if enh is not None and enh >= ENH_THRESHOLD:
                    dates_with_enhancement.append((key, enh, nc_path.name))
                    log.info("    ★ ENHANCEMENT DETECTED: %+.1f ppb", enh)
            results[key] = {"orbit": nc_path.name, **extraction}

    else:
        # Download + extract (prioritised order)
        log.info("[2] Processing %d dates (prioritised by proximity to Sep-18)...", len(prioritised))
        for date in prioritised:
            if date not in catalog:
                continue

            products = catalog[date]

            # Check if already cached
            nc_path = None
            for p in products:
                safe = p["name"].replace(" ", "_")
                if not safe.endswith(".nc"):
                    safe += ".nc"
                candidate = TROPOMI_DIR / safe
                if candidate.exists():
                    nc_path = candidate
                    log.info("  %s → cached", date)
                    break

            if nc_path is None:
                if n_downloaded >= args.max_downloads:
                    log.info("  %s → download limit reached (%d); skipping", date, args.max_downloads)
                    results[date] = {"status": "skipped_limit"}
                    continue

                nc_path = download_orbit(products[0], client)
                if nc_path:
                    n_downloaded += 1
                else:
                    results[date] = {"status": "download_failed"}
                    continue

            log.info("  %s → extracting ΔXCH4...", date)
            extraction = extract_xch4(nc_path)
            enh = extraction.get("enhancement")

            if "error" in extraction:
                log.info("    → %s", extraction["error"])
                results[date] = {"orbit": nc_path.name, **extraction}
            else:
                n_covered += 1
                log.info("    → ΔXCH4=%+.1f ppb  qa=%.2f  dist=%.3f°  n_bg=%d",
                         enh, extraction["qa_at_site"],
                         extraction["dist_to_site_deg"], extraction["n_bg_pixels"])
                if enh is not None and enh >= ENH_THRESHOLD:
                    dates_with_enhancement.append((date, enh, nc_path.name))
                    log.info("    ★ ENHANCEMENT DETECTED: %+.1f ppb on %s", enh, date)
                results[date] = {"orbit": nc_path.name, **extraction}

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY — Weisweiler TROPOMI Search")
    print("=" * 70)
    print(f"  Dates with orbit covering site: {n_covered}")
    print(f"  Dates with ΔXCH4 ≥ {ENH_THRESHOLD} ppb:  {len(dates_with_enhancement)}")

    if dates_with_enhancement:
        print("\n  ★ POSITIVE DETECTIONS:")
        for date, enh, orbit in sorted(dates_with_enhancement, key=lambda x: -x[1]):
            print(f"    {date}   ΔXCH4={enh:+.1f} ppb   {orbit[:60]}")
        print("\n  → These dates can be used for wind-independent flow quantification")
        print("    via the IME approach: Q̂ ≈ U × ΔXCH4_column × pixel_area / H_mix")
    else:
        print("\n  No TROPOMI enhancement ≥ 5 ppb found in the processed dates.")
        if n_downloaded < args.max_downloads:
            print("  All available orbits processed — Weisweiler plume below TROPOMI sensitivity.")
        else:
            print(f"  Only {n_downloaded} of {len([d for d in catalog if d not in results])+n_downloaded}"
                  f" orbits downloaded so far.")
            print("  Run with larger --max-downloads to check remaining dates.")

    # Save
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    log.info("\nResults saved → %s", RESULTS_JSON)


if __name__ == "__main__":
    main()
