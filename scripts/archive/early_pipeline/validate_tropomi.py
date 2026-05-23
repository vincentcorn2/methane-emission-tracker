"""
validate_tropomi.py
===================
Cross-validates CH4Net S/C detections against TROPOMI S5P L2 CH4 column data.

For each site × detection date this script:
  1. Finds the TROPOMI S5P/OFFL/L2__CH4___ orbit file whose swath covers the
     site on or within ±2 days of the Sentinel-2 acquisition date.
  2. Downloads the NetCDF file (~400–600 MB per orbit; one file covers all EU sites
     on a given day, so at most N_DATES files are downloaded regardless of
     how many sites share that date).
  3. Extracts the bias-corrected XCH4 column (ppb) at the site pixel and
     computes background as the 20th percentile of a 0.25–1.0° annulus.
  4. Enhancement = site_xch4 - background_xch4 (ppb).
  5. Flags bad S2 scenes (site_mean ≈ 0.4257) so they are excluded from
     the correlation analysis.
  6. Saves results to results_analysis/tropomi_validation.json and prints
     a cross-validation table: site × date × S/C × ΔXCH4 (ppb).

Signal interpretation:
  ΔXCHₓ > +5 ppb  + S/C > 1.15  →  high-confidence detection (dual-sensor)
  ΔXCHₓ ≈ 0      + S/C > 1.15  →  S2-only detection (plume below TROPOMI sensitivity)
  ΔXCHₓ > +5 ppb  + S/C ≈ 1.0  →  TROPOMI-only (S2 bad scene or missed detection)
  Both ≈ background               →  non-detection or genuine non-emitter

Prerequisites:
  conda activate methane
  pip install netCDF4 --break-system-packages   # if not already installed
  export COPERNICUS_USER=your@email.com
  export COPERNICUS_PASS='yourpassword'

Usage:
  caffeinate -i python validate_tropomi.py                    # all sites, all dates
  python validate_tropomi.py --no-download                    # use cached NetCDF only
  python validate_tropomi.py --sites belchatow neurath        # specific sites
  python validate_tropomi.py --date-window 3                  # ±3 day TROPOMI match
"""

import os
import sys
import json
import math
import getpass
import logging
import argparse
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

# ── Logging ────────────────────────────────────────────────────────────────────
Path("results_analysis").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("results_analysis/tropomi_validation.log"),
    ],
)
log = logging.getLogger(__name__)

# ── Dependencies ───────────────────────────────────────────────────────────────
MISSING = []
try:
    import netCDF4 as nc
except ImportError:
    MISSING.append("netCDF4")
try:
    import requests
except ImportError:
    MISSING.append("requests")

if MISSING:
    print(f"Missing packages: {', '.join(MISSING)}")
    print(f"Install with: pip install {' '.join(MISSING)} --break-system-packages")
    sys.exit(1)

from src.ingestion.copernicus_client import CopernicusClient

# ── Config ────────────────────────────────────────────────────────────────────
TROPOMI_DIR      = Path("data/downloads/tropomi")
RESULTS_JSON     = Path("results_analysis/tropomi_validation.json")
MULTIDATE_JSON   = Path("results_analysis/multidate_validation.json")

# Site background annulus: exclude inner_deg, include outer_deg
BG_INNER_DEG = 0.25   # ~27 km  — exclude immediate site area
BG_OUTER_DEG = 1.00   # ~110 km — background annulus width
BG_PERCENTILE = 20    # use 20th pctile of annulus as background (robust to other emitters)

# TROPOMI EU daytime overpass window (UTC hours, inclusive).
# Sentinel-5P crosses central Europe (~5–20°E) at roughly 11:00–13:30 UTC.
# Orbits outside this window are Atlantic/nighttime passes and cover nothing useful.
EU_OVERPASS_UTC_START = 11  # inclusive lower bound (hour)
EU_OVERPASS_UTC_END   = 14  # exclusive upper bound (hour)

# Bad-scene sentinel value — S2 scenes producing this site_mean are unusable
BAD_SCENE_MEAN = 0.4257   # model degenerate output; matches 0.425702 in results

# TROPOMI quality flag threshold (ESA recommends 0.5 for scientific use)
QA_THRESHOLD = 0.5

# Detection threshold (S/C)
CLASSIC_THRESH = 1.15

# Date window: look for TROPOMI overpass within ±N days of S2 acquisition
DEFAULT_DATE_WINDOW = 2

# ── Site registry ─────────────────────────────────────────────────────────────
# Top-10 EU emitters by CO2 (from EU-ETS/E-PRTR, used as proxy for CH4 scale).
# Includes our 7 existing sites + 3 new ones (Jänschwalde, Schwarze Pumpe, Turów).
SITES = {
    # ── Existing confirmed / evaluated sites ──────────────────────────────────
    "weisweiler":    dict(lat=50.837, lon=6.322,  tile_id="T31UGS",
                          operator="RWE",          country="DE",
                          mw=1060,   fuel="lignite",
                          label="✓ confirmed",
                          note="S/C=23.461 on 2024-09-18"),
    "neurath":       dict(lat=51.038, lon=6.616,  tile_id="T32ULB",
                          operator="RWE",          country="DE",
                          mw=4400,   fuel="lignite",
                          label="✓ confirmed",
                          note="S/C=67.205 on 2024-08-29, S/C=23.039 on 2024-06-25"),
    "niederaussem":  dict(lat=50.971, lon=6.667,  tile_id="T32ULB",
                          operator="RWE",          country="DE",
                          mw=3827,   fuel="lignite",
                          label="✗ non-detection",
                          note="0/4 dates detected — likely reduced operation 2024"),
    "belchatow":     dict(lat=51.266, lon=19.315, tile_id="T34UCB",
                          operator="PGE",          country="PL",
                          mw=4830,   fuel="lignite",
                          label="✓ confirmed",
                          note="3/4 dates detected, mean S/C=45"),
    "boxberg":       dict(lat=51.416, lon=14.565, tile_id="T33UVT",
                          operator="LEAG",         country="DE",
                          mw=2575,   fuel="lignite",
                          label="✗ unreliable",
                          note="Open-pit mine — wildly variable S/C, discard"),
    "lippendorf":    dict(lat=51.178, lon=12.378, tile_id="T33UUS",
                          operator="LEAG",         country="DE",
                          mw=1866,   fuel="lignite",
                          label="✗ artifact",
                          note="BT collapse 155→0.195; seasonal SWIR vegetation artifact"),
    "rybnik":        dict(lat=50.135, lon=18.522, tile_id="T34UCA",
                          operator="PGE",          country="PL",
                          mw=1775,   fuel="coal",
                          label="~ control",
                          note="Ring profile increases outward — terrain artifact"),
    "groningen":     dict(lat=53.252, lon=6.682,  tile_id="T31UGV",
                          operator="NAM",          country="NL",
                          mw=None,   fuel="gas",
                          label="✗ control",
                          note="TROPOMI benchmark: −0.99 ppb (confirmed non-detection)"),
    "maasvlakte":    dict(lat=51.944, lon=4.067,  tile_id="T31UET",
                          operator="Vattenfall",   country="NL",
                          mw=1070,   fuel="coal",
                          label="✗ non-detection",
                          note="S/C=0.210 on v8 — Rotterdam port"),

    # ── New sites: completing top-10 EU CO2 emitters list ────────────────────
    # Jänschwalde: RWE lignite plant, Brandenburg. ~22 Mt CO2/yr (EU #2).
    # tile_id will be discovered from catalog (likely T33UUT).
    "jaenschwalde":  dict(lat=51.838, lon=14.621, tile_id=None,
                          tile_candidates=["T33UUT", "T33UUS", "T33UVT"],
                          operator="LEAG",         country="DE",
                          mw=3000,   fuel="lignite",
                          label="? new",
                          note="EU #2 CO2 emitter, ~22 Mt/yr — not yet evaluated"),

    # Schwarze Pumpe: LEAG lignite plant, Brandenburg. ~13 Mt CO2/yr.
    "schwarze_pumpe": dict(lat=51.579, lon=14.328, tile_id=None,
                           tile_candidates=["T33UUT", "T33UUS"],
                           operator="LEAG",         country="DE",
                           mw=1600,   fuel="lignite",
                           label="? new",
                           note="~13 Mt CO2/yr Brandenburg cluster"),

    # Turów: PGE lignite plant, SW Poland (Zgorzelec). ~10 Mt CO2/yr.
    "turow":          dict(lat=51.003, lon=14.988, tile_id=None,
                           tile_candidates=["T33UUT", "T33UVT"],
                           operator="PGE",          country="PL",
                           mw=1938,   fuel="lignite",
                           label="? new",
                           note="~10 Mt CO2/yr SW Poland — adjacent to DE/CZ border"),
}


# ── TROPOMI catalog search ─────────────────────────────────────────────────────

def _orbit_start_dt(name_or_path: str) -> datetime | None:
    """
    Parse the orbit start UTC datetime from a TROPOMI product name or filename.

    TROPOMI product names follow the convention:
      S5P_OFFL_L2__CH4____<YYYYMMDDTHHmmss>_<YYYYMMDDTHHmmss>_<orbit>_...
    The first timestamp is the orbit start.
    """
    import re
    m = re.search(r"CH4[_]{4}(\d{8}T\d{6})", str(name_or_path))
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y%m%dT%H%M%S")
        except ValueError:
            pass
    return None


def _is_eu_daytime_orbit(name_or_path: str) -> bool:
    """
    Return True if the orbit start time falls within the EU daytime overpass window.

    Sentinel-5P crosses central Europe (~5–20°E) at roughly 11:00–13:30 UTC
    (ascending node).  Orbits starting before 11:xx UTC are Atlantic/western
    passes and will contain no valid pixels over central European sites.
    """
    t = _orbit_start_dt(name_or_path)
    if t is None:
        return False  # Can't determine — treat as potentially valid
    return EU_OVERPASS_UTC_START <= t.hour < EU_OVERPASS_UTC_END


def europe_bbox_wkt(margin_deg: float = 0.3) -> str:
    """Broad WKT polygon covering all EU sites (for TROPOMI orbit search)."""
    # Bounding box: 4°W–20°E, 49°N–55°N  (covers DE + PL sites)
    return "POLYGON((-4 49, 20 49, 20 55, -4 55, -4 49))"


def site_bbox_wkt(lat: float, lon: float, margin: float = 0.3) -> str:
    return (f"POLYGON(({lon-margin} {lat-margin},{lon+margin} {lat-margin},"
            f"{lon+margin} {lat+margin},{lon-margin} {lat+margin},"
            f"{lon-margin} {lat-margin}))")


def search_tropomi_orbits(client: CopernicusClient,
                          date_str: str,
                          window_days: int = DEFAULT_DATE_WINDOW) -> list:
    """
    Find TROPOMI S5P L2 CH4 products covering Europe on date_str ± window_days.

    Returns list of (product_id, product_name, start_time) tuples, sorted by
    closeness to date_str.
    """
    acq_date = datetime.strptime(date_str, "%Y%m%d")
    start = (acq_date - timedelta(days=window_days)).strftime("%Y-%m-%dT00:00:00.000Z")
    end   = (acq_date + timedelta(days=window_days)).strftime("%Y-%m-%dT23:59:59.000Z")

    wkt = europe_bbox_wkt()

    # S5P products live under collection "SENTINEL-5P" in CDSE
    params = {
        "$filter": (
            f"Collection/Name eq 'SENTINEL-5P' and "
            f"OData.CSC.Intersects(area=geography'SRID=4326;{wkt}') and "
            f"ContentDate/Start ge {start} and "
            f"ContentDate/Start le {end} and "
            f"contains(Name,'L2__CH4___')"
        ),
        "$top": 50,
        "$orderby": "ContentDate/Start desc",
    }

    headers = {"Authorization": f"Bearer {client.token}"}
    r = client._session.get(client._catalog_url, params=params, headers=headers, timeout=60)
    if r.status_code == 401:
        client._refresh_token()
        headers = {"Authorization": f"Bearer {client.token}"}
        r = client._session.get(client._catalog_url, params=params, headers=headers, timeout=60)
    r.raise_for_status()

    items = r.json().get("value", [])
    log.info("  Found %d S5P L2 CH4 products for %s ± %dd", len(items), date_str, window_days)

    results = []
    for item in items:
        name  = item.get("Name", "")
        pid   = item.get("Id", "")
        start_t = item.get("ContentDate", {}).get("Start", "")
        if "L2__CH4___" in name and "OFFL" in name:
            results.append({"id": pid, "name": name, "start": start_t})

    # Sort by:
    #   1. Whether the orbit falls in the EU daytime overpass window (prefer daytime)
    #   2. Closeness in total seconds to target 12:30 UTC on the acquisition date
    # This avoids selecting Pacific nighttime orbits (~22:xx UTC) over the correct
    # European daytime orbit (~11:xx–13:xx UTC) when multiple products share a date.
    target_dt = acq_date.replace(hour=12, minute=30, second=0)

    def _sort_key(x: dict):
        t = _orbit_start_dt(x.get("name", "") or x.get("start", ""))
        if t is None:
            # Fall back to parsing from the start date string
            try:
                t = datetime.strptime(x["start"][:19], "%Y-%m-%dT%H:%M:%S")
            except Exception:
                return (1, float("inf"))
        is_daytime = (EU_OVERPASS_UTC_START <= t.hour < EU_OVERPASS_UTC_END)
        secs_from_target = abs((t - target_dt).total_seconds())
        # Daytime orbits rank first (0), nighttime second (1);
        # within each group sort by proximity to 12:30 UTC on acquisition date.
        return (0 if is_daytime else 1, secs_from_target)

    results.sort(key=_sort_key)
    return results


def download_tropomi(product: dict, client: CopernicusClient) -> Path | None:
    """Download a TROPOMI NetCDF file. Returns local path or None on failure."""
    TROPOMI_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = product["name"].replace(" ", "_")
    if not safe_name.endswith(".nc"):
        safe_name += ".nc"
    out_path = TROPOMI_DIR / safe_name

    if out_path.exists():
        log.info("    Cached: %s", out_path.name)
        return out_path

    # Try .zip first (some products are delivered zipped), then direct download
    download_url = (
        f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products({product['id']})/$value"
    )
    headers = {"Authorization": f"Bearer {client.token}"}

    try:
        head = client._session.head(download_url, headers=headers, allow_redirects=False)
        redirect_url = head.headers.get("Location", download_url) if head.status_code in (301,302,303,307,308) else download_url

        log.info("    Downloading %s (~400–600 MB)...", product["name"][:70])
        with client._session.get(redirect_url, headers=headers, stream=True, timeout=600) as r:
            r.raise_for_status()
            size = 0
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
                        size += len(chunk)
        log.info("    Saved: %s (%.1f MB)", out_path.name, size / 1e6)
        return out_path
    except Exception as e:
        log.error("    Download failed: %s", e)
        if out_path.exists():
            out_path.unlink()
        return None


# ── TROPOMI NetCDF extraction ──────────────────────────────────────────────────

def extract_xch4(nc_path: Path, lat: float, lon: float) -> dict:
    """
    Extract XCH4 at (lat, lon) from a TROPOMI S5P L2 CH4 NetCDF file.

    Returns dict with:
      site_xch4:   bias-corrected column mixing ratio at site pixel (ppb)
      bg_xch4:     20th-percentile of annulus background (ppb)
      enhancement: site_xch4 - bg_xch4 (ppb)
      qa_at_site:  quality flag at site pixel
      n_bg_pixels: number of valid background pixels used
      overpass_utc: UTC time of site pixel overpass

    Returns {"error": "..."} on failure.
    """
    try:
        ds = nc.Dataset(str(nc_path), "r")
    except Exception as e:
        return {"error": f"netCDF4 open failed: {e}"}

    try:
        grp = ds["PRODUCT"]

        # ── Core arrays ────────────────────────────────────────────────────
        # TROPOMI L2 CH4 product structure:
        #   latitude / longitude:  (n_scanlines, n_ground_pixels)   — 2D
        #   methane_mixing_ratio_bias_corrected: (time=1, n_scanlines, n_ground_pixels) — 3D
        #   qa_value:              (time=1, n_scanlines, n_ground_pixels) — 3D
        # We squeeze the time dimension from 3D variables so all arrays are 2D.
        lats = np.array(grp["latitude"][:])   # always (n, m)
        lons = np.array(grp["longitude"][:])

        xch4_raw = np.array(grp["methane_mixing_ratio_bias_corrected"][:])
        qa_raw   = np.array(grp["qa_value"][:])

        # Squeeze leading time dimension if present (shape (1, n, m) → (n, m))
        xch4 = xch4_raw[0] if xch4_raw.ndim == 3 else xch4_raw
        qa   = qa_raw[0]   if qa_raw.ndim   == 3 else qa_raw
        if lats.ndim == 3:
            lats = lats[0]
            lons = lons[0]

        # Fill values → NaN
        xch4 = xch4.astype(np.float32)
        xch4[xch4 > 1e29] = np.nan

        ds.close()

        # ── Quality mask ───────────────────────────────────────────────────
        valid = (qa >= QA_THRESHOLD) & np.isfinite(xch4)

        # ── Site pixel ────────────────────────────────────────────────────
        dist_deg = np.sqrt((lats - lat)**2 + (lons - lon)**2)
        # Only consider valid pixels
        dist_masked = np.where(valid, dist_deg, np.inf)
        flat_idx    = int(np.argmin(dist_masked))
        si, sj      = np.unravel_index(flat_idx, dist_masked.shape)  # 2D now

        min_dist = float(dist_masked[si, sj])
        if min_dist > 0.15:  # ~16 km — site not covered by this orbit
            return {"error": f"site not in orbit swath (nearest valid pixel: {min_dist:.2f}°)"}

        site_val = float(xch4[si, sj])
        site_qa  = float(qa[si, sj])

        # ── Background annulus ────────────────────────────────────────────
        bg_mask = (
            (dist_deg >= BG_INNER_DEG) &
            (dist_deg <= BG_OUTER_DEG) &
            valid
        )
        bg_pixels = xch4[bg_mask]
        if len(bg_pixels) < 10:
            return {"error": f"too few background pixels ({len(bg_pixels)} < 10)"}

        bg_val   = float(np.percentile(bg_pixels, BG_PERCENTILE))
        enh      = round(site_val - bg_val, 2)

        return {
            "site_xch4":   round(site_val, 2),
            "bg_xch4":     round(bg_val, 2),
            "enhancement": enh,
            "qa_at_site":  round(site_qa, 3),
            "n_bg_pixels": int(bg_mask.sum()),
            "dist_to_site_deg": round(min_dist, 4),
        }

    except Exception as e:
        try:
            ds.close()
        except Exception:
            pass
        log.debug("extract_xch4 error at (%.3f, %.3f) in %s: %s", lat, lon, nc_path.name, e)
        return {"error": str(e)}


# ── Load S/C results from multidate_validation.json ───────────────────────────

def load_sc_results() -> dict:
    """Load multidate S/C results, keyed by (site_name, date_str)."""
    if not MULTIDATE_JSON.exists():
        log.warning("multidate_validation.json not found — run validate_multidate.py first")
        return {}

    with open(MULTIDATE_JSON) as f:
        data = json.load(f)

    sc_map = {}
    for site, res in data.items():
        for date, dr in res.get("dates", {}).items():
            sc_map[(site, date)] = dr
    return sc_map


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TROPOMI cross-validation for CH4Net detections")
    parser.add_argument("--sites", nargs="+", default=None,
                        help="Limit to specific sites (default: all)")
    parser.add_argument("--no-download", action="store_true",
                        help="Use only cached NetCDF files; skip Copernicus download")
    parser.add_argument("--date-window", type=int, default=DEFAULT_DATE_WINDOW,
                        help=f"±days to search for TROPOMI overpass (default: {DEFAULT_DATE_WINDOW})")
    args = parser.parse_args()

    sites_to_run = {k: v for k, v in SITES.items()
                    if args.sites is None or k in args.sites}

    print("=" * 70)
    print("  TROPOMI S5P L2 CH4 Cross-Validation — CH4Net v2")
    print(f"  Sites:    {', '.join(sites_to_run)}")
    print(f"  Window:   ±{args.date_window} days for TROPOMI match")
    print(f"  Download: {'disabled' if args.no_download else 'enabled'}")
    print("=" * 70)

    # Load Sentinel-2 S/C results
    sc_results = load_sc_results()
    if not sc_results:
        print("\nNo S/C results found. Run validate_multidate.py first.")
        sys.exit(1)

    # Authenticate
    client = None
    if not args.no_download:
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

    # Collect all unique dates we need.  For each date we keep a *list* of candidate
    # orbit files sorted by proximity to 12:30 UTC.  Extraction will try each in turn
    # and use the first one whose swath actually covers the target site — necessary
    # because TROPOMI's ~2600 km swath may cover western EU (6°E) but miss eastern
    # sites (19°E) on a given orbit, and vice-versa.
    N_ORBITS_PER_DATE = 3          # download/cache at most this many daytime orbits
    date_to_ncs: dict[str, list[Path]] = {}
    all_dates = sorted({date for (site, date) in sc_results if site in sites_to_run})

    log.info("\n[2] Finding TROPOMI orbits for %d unique acquisition dates...", len(all_dates))

    for date_str in all_dates:
        # Check cache: collect all .nc files whose date falls within ±window_days
        candidate_files: list[Path] = []
        if TROPOMI_DIR.exists():
            acq = datetime.strptime(date_str, "%Y%m%d")
            for delta in range(-args.date_window, args.date_window + 1):
                alt = (acq + timedelta(days=delta)).strftime("%Y%m%d")
                candidate_files.extend(TROPOMI_DIR.glob(f"*L2__CH4___{alt}*.nc"))

        # Prefer daytime orbits; discard nighttime/Atlantic passes
        daytime_files = [f for f in candidate_files if _is_eu_daytime_orbit(f.name)]
        any_files     = candidate_files  # fallback if --no-download

        target_dt = datetime.strptime(date_str, "%Y%m%d").replace(hour=12, minute=30)

        if daytime_files:
            # Sort ALL cached daytime files by proximity to 12:30 UTC; keep all of them
            # so the extraction loop can fall back to the next orbit if swath misses the site.
            daytime_files.sort(key=lambda f: abs(
                ((_orbit_start_dt(f.name) or target_dt) - target_dt).total_seconds()
            ))
            log.info("  %s → %d daytime cached: %s%s", date_str, len(daytime_files),
                     daytime_files[0].name[:60],
                     " (+more)" if len(daytime_files) > 1 else "")
            date_to_ncs[date_str] = daytime_files
            # Even if we have cached files, fall through to download more if we have < N_ORBITS_PER_DATE
            # and downloading is enabled — this ensures multi-track coverage.
            if len(daytime_files) >= N_ORBITS_PER_DATE or args.no_download:
                continue

        elif any_files and args.no_download:
            # --no-download: fall back to any cached orbit even if nighttime
            log.info("  %s → nighttime cached (--no-download): %s", date_str, any_files[0].name)
            date_to_ncs[date_str] = [any_files[0]]
            continue

        elif any_files:
            # Nighttime orbits already cached — need to download the correct daytime orbits
            log.info("  %s → cached files are nighttime orbits; downloading EU daytime orbits", date_str)

        if args.no_download:
            log.info("  %s → no cached file; skipping (--no-download)", date_str)
            date_to_ncs.setdefault(date_str, [])
            continue

        # Search Copernicus and download up to N_ORBITS_PER_DATE daytime orbits
        log.info("  %s → searching catalog...", date_str)
        orbits = search_tropomi_orbits(client, date_str, args.date_window)

        if not orbits:
            log.warning("  %s → no TROPOMI CH4 orbit found in ±%dd window", date_str, args.date_window)
            date_to_ncs.setdefault(date_str, [])
            continue

        # Download the top N daytime orbits (they're already sorted daytime-first)
        already_cached = set(p.name for p in date_to_ncs.get(date_str, []))
        downloaded = list(date_to_ncs.get(date_str, []))
        for orbit in orbits:
            if len(downloaded) >= N_ORBITS_PER_DATE:
                break
            # Skip if we already have this file cached
            safe_name = orbit["name"].replace(" ", "_")
            if not safe_name.endswith(".nc"):
                safe_name += ".nc"
            if safe_name in already_cached:
                continue
            # Only download daytime orbits
            if not _is_eu_daytime_orbit(orbit.get("name", "")):
                continue
            log.info("  %s [%d/%d] → %s", date_str, len(downloaded) + 1,
                     N_ORBITS_PER_DATE, orbit["name"][:70])
            nc_path = download_tropomi(orbit, client)
            if nc_path:
                downloaded.append(nc_path)
        date_to_ncs[date_str] = downloaded

    # Extract XCH4 for each site × date
    log.info("\n[3] Extracting XCH4 at each site...")
    all_results = {}

    for site_name, meta in sites_to_run.items():
        lat, lon = meta["lat"], meta["lon"]
        log.info("\n  Site: %-16s  (%.3f°N, %.3f°E)", site_name.upper(), lat, lon)

        site_dates = {date: dr for (s, date), dr in sc_results.items() if s == site_name}
        if not site_dates:
            log.warning("    No S/C results found for %s", site_name)
            continue

        date_records = {}
        for date_str, sc_data in sorted(site_dates.items()):
            sc_ratio   = sc_data.get("sc_ratio")
            site_mean  = sc_data.get("site_mean")

            # Flag bad S2 scenes (degenerate model output)
            is_bad_scene = (site_mean is not None and abs(site_mean - BAD_SCENE_MEAN) < 0.01)
            s2_detect    = (sc_ratio is not None and sc_ratio > CLASSIC_THRESH and not is_bad_scene)

            # Try each cached orbit in turn; use the first whose swath covers the site.
            # Multiple orbits are needed because TROPOMI's ~2600 km swath may cover
            # western EU sites (6°E) but not eastern ones (19°E) on a given pass.
            nc_paths = date_to_ncs.get(date_str) or []
            trop = {"error": "no_orbit_file"}
            for nc_path in nc_paths:
                if nc_path is None or not nc_path.exists():
                    continue
                candidate = extract_xch4(nc_path, lat, lon)
                if "error" not in candidate:
                    trop = candidate
                    break
                # Keep trying if the only issue is the swath miss; bail on other errors
                err = candidate.get("error", "")
                log.debug("    %s  orbit %s  error: %s", date_str, nc_path.name[:30], err)
                if "site not in orbit swath" in err:
                    trop = candidate   # update best-so-far; continue trying
                else:
                    trop = candidate
                    break  # netCDF4 open failure etc. — no point trying more

            enh = trop.get("enhancement")

            # Dual-sensor assessment
            trop_detect = (enh is not None and enh >= 5.0)
            if is_bad_scene:
                assessment = "— bad S2 scene"
            elif s2_detect and trop_detect:
                assessment = "✓✓ DUAL-SENSOR CONFIRM"
            elif s2_detect and not trop_detect:
                assessment = "✓  S2-only (plume below TROPOMI sensitivity or no orbit)"
            elif not s2_detect and trop_detect:
                assessment = "~  TROPOMI-only (S2 bad/miss)"
            else:
                assessment = "✗  non-detection"

            log.info("    %s  S/C=%-8s  ΔXCH4=%s ppb  %s",
                     date_str,
                     f"{sc_ratio:.3f}" if sc_ratio else "—",
                     f"{enh:+.1f}" if enh is not None else "—",
                     assessment)

            date_records[date_str] = {
                "sc_ratio":       sc_ratio,
                "site_mean":      site_mean,
                "is_bad_scene":   is_bad_scene,
                "s2_detect":      s2_detect,
                "tropomi":        trop,
                "trop_detect":    trop_detect,
                "nc_file":        str(nc_path) if nc_path else None,
                "assessment":     assessment,
            }

        # Compute site-level detection statistics (exclude bad scenes)
        valid_dates = {d: r for d, r in date_records.items() if not r["is_bad_scene"] and "error" not in r.get("tropomi", {}).get("error", "VALID")}
        # Actually, even dates without TROPOMI are valid for S/C stats
        valid_s2 = {d: r for d, r in date_records.items() if not r["is_bad_scene"] and r.get("sc_ratio") is not None}
        n_s2_detect = sum(1 for r in valid_s2.values() if r["s2_detect"])
        sc_vals_when_detected = [r["sc_ratio"] for r in valid_s2.values() if r["s2_detect"]]

        p_detect = n_s2_detect / len(valid_s2) if valid_s2 else None

        # Wilson 95% CI for p_detect
        if valid_s2:
            n = len(valid_s2)
            p = n_s2_detect / n
            z = 1.96
            denom = 1 + z**2 / n
            centre = (p + z**2 / (2 * n)) / denom
            margin = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
            p_detect_lo = max(0.0, round(centre - margin, 3))
            p_detect_hi = min(1.0, round(centre + margin, 3))
        else:
            p_detect_lo = p_detect_hi = None

        all_results[site_name] = {
            "lat": lat,
            "lon": lon,
            "operator": meta.get("operator"),
            "country":  meta.get("country"),
            "mw":       meta.get("mw"),
            "fuel":     meta.get("fuel"),
            "label":    meta.get("label"),
            "p_detect":         round(p_detect, 3) if p_detect is not None else None,
            "p_detect_lo_95":   p_detect_lo,
            "p_detect_hi_95":   p_detect_hi,
            "n_valid_dates":    len(valid_s2),
            "n_detections":     n_s2_detect,
            "mean_sc_detected": round(float(np.mean(sc_vals_when_detected)), 3) if sc_vals_when_detected else None,
            "dates": date_records,
        }

    # Print cross-validation table
    print("\n")
    print("=" * 110)
    print("  TROPOMI CROSS-VALIDATION TABLE — CH4Net v2 vs S5P L2 XCH4")
    print("=" * 110)
    print(f"  {'Site':<16} {'Label':<14} {'Date':>10}  {'S/C':>8}  {'ΔXCH4 (ppb)':>12}  {'Assessment'}")
    print("  " + "-" * 107)

    for site_name, res in all_results.items():
        label = res.get("label", "")
        for i, (date, dr) in enumerate(sorted(res["dates"].items())):
            prefix = site_name if i == 0 else ""
            lab    = label     if i == 0 else ""
            sc  = dr.get("sc_ratio")
            enh = dr.get("tropomi", {}).get("enhancement")
            asm = dr.get("assessment", "—")
            sc_str  = f"{sc:>8.3f}" if sc else f"{'—':>8}"
            enh_str = f"{enh:>+12.1f}" if enh is not None else f"{'—':>12}"
            print(f"  {prefix:<16} {lab:<14} {date:>10}  {sc_str}  {enh_str}  {asm}")

        # Summary row
        p = res.get("p_detect")
        lo = res.get("p_detect_lo_95")
        hi = res.get("p_detect_hi_95")
        n  = res.get("n_valid_dates", 0)
        nd = res.get("n_detections", 0)
        msc = res.get("mean_sc_detected")
        if p is not None:
            ci_str = f"[{lo:.2f}, {hi:.2f}]" if lo is not None else ""
            print(f"  {'':16} {'':14} {'SUMMARY':>10}  "
                  f"  p={p:.2f} {ci_str}  {nd}/{n} S2 detections"
                  + (f"  mean S/C={msc:.1f} when detected" if msc else ""))
        print("  " + "-" * 107)

    print("=" * 110)
    print("\n  Dual-sensor threshold: S/C > 1.15  AND  ΔXCHₓ ≥ +5 ppb")
    print("  Background: 20th percentile of 0.25–1.0° annulus around site")
    print("  TROPOMI quality flag: qa_value ≥ 0.5 (ESA scientific standard)")

    # Save
    with open(RESULTS_JSON, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info("\nResults saved → %s", RESULTS_JSON)


if __name__ == "__main__":
    main()
