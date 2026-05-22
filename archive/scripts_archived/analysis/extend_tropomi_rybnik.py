"""
scripts/extend_tropomi_rybnik.py
=================================
Extends the TROPOMI methane mining record for Rybnik and the four other
Silesian JSW coal mine sites from October 2024 through the current date.

The original tropomi_mine_europe.py covered 2023-01-01 → 2024-09-30.
This script picks up from 2024-10-01 and appends to the SAME output files:
  results_analysis/tropomi_mining_results.csv
  results_analysis/tropomi_positives.json

It is fully resumable — already-processed (site, date) pairs are skipped.
After writing new data it also rewrites results_analysis/rybnik_tropomi_summary.json
with the combined view across the full time window.

Usage
-----
  # From the methane-api root directory:
  export COPERNICUS_USER="vcc2127@columbia.edu"
  export COPERNICUS_PASS="<password>"
  python scripts/extend_tropomi_rybnik.py

  # Or let it prompt:
  python scripts/extend_tropomi_rybnik.py

  # Dry run (shows what would be fetched, no downloads):
  python scripts/extend_tropomi_rybnik.py --dry-run

  # Only Rybnik, skip the other Silesian sites:
  python scripts/extend_tropomi_rybnik.py --rybnik-only

Runtime
-------
Roughly 2–5 minutes per site per month (dominated by CDSE search latency on
days with no overpass). For ~7 months × 5 sites ≈ 30-90 minutes total.
Use nohup or a tmux session.
"""

import argparse
import csv
import getpass
import json
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import requests

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT / "data" / "tropomi"
RESULTS_DIR = ROOT / "results_analysis"
CSV_PATH    = RESULTS_DIR / "tropomi_mining_results.csv"
JSON_PATH   = RESULTS_DIR / "tropomi_positives.json"
SUMMARY_OUT = RESULTS_DIR / "rybnik_tropomi_summary.json"

# ── Extension date window ──────────────────────────────────────────────────────
EXTEND_START = "2024-10-01"
EXTEND_END   = date.today().strftime("%Y-%m-%d")   # up to today

# ── Detection parameters (match original script) ──────────────────────────────
THRESHOLD_PPB = 10.0
QA_THRESHOLD  = 0.5
BOX_DEG       = 1.0
NEAR_DEG      = 0.3

# ── Sites: Rybnik + four other Silesian JSW mines ─────────────────────────────
SILESIAN_SITES = {
    "silesia_rybnik": dict(
        lat=50.135, lon=18.522,
        s2_tile="T34UCA",
        source_type="coal_mine_plant",
        country="PL",
        note="Rybnik area — Chwałowice/Jankowice mines (PGG), co-located with Rybnik power plant",
    ),
    "silesia_jastrzebie": dict(
        lat=49.950, lon=18.590,
        s2_tile="T34UCB",
        source_type="coal_mine",
        country="PL",
        note="Jastrzebie-Zdroj complex (JSW) — active ventilation shafts",
    ),
    "silesia_knurow": dict(
        lat=50.220, lon=18.670,
        s2_tile="T34UCA",
        source_type="coal_mine",
        country="PL",
        note="Knurow-Szczyglowice (JSW) — documented TROPOMI anomalies 2019–2022",
    ),
    "silesia_pniowek": dict(
        lat=49.930, lon=18.640,
        s2_tile="T34UCB",
        source_type="coal_mine",
        country="PL",
        note="Pniowek (JSW) — among highest CMM ventilation rates in Poland",
    ),
    "silesia_zofiowka": dict(
        lat=49.980, lon=18.600,
        s2_tile="T34UCB",
        source_type="coal_mine",
        country="PL",
        note="Zofiowka (JSW) — active deep coal mine, persistent TROPOMI signal",
    ),
}

# ── CDSE endpoints ─────────────────────────────────────────────────────────────
AUTH_URL     = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
SEARCH_URL   = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
DOWNLOAD_URL = "https://zipper.dataspace.copernicus.eu/odata/v1/Products"

FIELDS = [
    "site", "date", "country", "source_type", "lat", "lon",
    "product_name", "status",
    "background_ppb", "mean_near_ppb", "max_near_ppb",
    "enhancement_ppb", "n_box_pixels", "n_near_pixels",
    "near_radius_deg", "qa_threshold",
    "validated", "s2_tile",
]


# ── Auth ───────────────────────────────────────────────────────────────────────
def get_token(username: str, password: str) -> str:
    r = requests.post(AUTH_URL, data={
        "grant_type": "password", "username": username,
        "password": password, "client_id": "cdse-public",
    }, timeout=30)
    r.raise_for_status()
    return r.json()["access_token"]


# ── TROPOMI product search ─────────────────────────────────────────────────────
def search_tropomi(lat: float, lon: float, date_str: str) -> list:
    start = f"{date_str}T00:00:00.000Z"
    end   = f"{date_str}T23:59:59.000Z"
    bbox  = (f"POLYGON(({lon-BOX_DEG} {lat-BOX_DEG},{lon+BOX_DEG} {lat-BOX_DEG},"
             f"{lon+BOX_DEG} {lat+BOX_DEG},{lon-BOX_DEG} {lat+BOX_DEG},"
             f"{lon-BOX_DEG} {lat-BOX_DEG}))")
    params = {
        "$filter": (
            f"Collection/Name eq 'SENTINEL-5P' "
            f"and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' "
            f"  and att/OData.CSC.StringAttribute/Value eq 'L2__CH4___') "
            f"and OData.CSC.Intersects(area=geography'SRID=4326;{bbox}') "
            f"and ContentDate/Start gt {start} "
            f"and ContentDate/Start lt {end}"
        ),
        "$top": 10,
        "$orderby": "ContentDate/Start asc",
    }
    r = requests.get(SEARCH_URL, params=params, timeout=30)
    r.raise_for_status()
    products = r.json().get("value", [])
    # Keep only daytime European overpasses (T09–T13 UTC)
    return [p for p in products
            if any(f"T{h:02d}" in p["Name"] for h in range(9, 14))]


# ── Download ───────────────────────────────────────────────────────────────────
def download_product(product: dict, token: str) -> Path:
    pid  = product["Id"]
    name = product["Name"]
    out  = DATA_DIR / (name if name.endswith(".nc") else name + ".nc")
    if out.exists():
        print("(cached)", end=" ", flush=True)
        return out
    headers = {"Authorization": f"Bearer {token}"}
    url = f"{DOWNLOAD_URL}({pid})/$value"
    with requests.get(url, headers=headers, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done  = 0
        with open(out, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"{100*done/total:.0f}%", end="\r", flush=True)
    print()
    return out


# ── XCH4 extraction ────────────────────────────────────────────────────────────
def extract_enhancement(nc_path: Path, lat: float, lon: float) -> dict:
    try:
        import netCDF4 as nc_lib
    except ImportError:
        return {"error": "netCDF4 not installed — run: pip install netCDF4"}

    def to_array(var):
        arr = var[0]
        return np.array(arr.filled(np.nan) if hasattr(arr, "filled") else arr,
                        dtype=np.float32)
    try:
        ds   = nc_lib.Dataset(nc_path)
        grp  = ds.groups["PRODUCT"]
        lats = to_array(grp.variables["latitude"])
        lons = to_array(grp.variables["longitude"])
        xch4 = to_array(grp.variables["methane_mixing_ratio"])
        qa   = to_array(grp.variables["qa_value"])
        ds.close()
    except Exception as e:
        return {"error": f"read_failed: {e}"}

    # Background box with adaptive QA
    for qa_thr in [QA_THRESHOLD, 0.3, 0.0]:
        in_box = (
            (lats >= lat - BOX_DEG) & (lats <= lat + BOX_DEG) &
            (lons >= lon - BOX_DEG) & (lons <= lon + BOX_DEG) &
            (qa   >= qa_thr)
        )
        if in_box.sum() >= 3:
            break

    if in_box.sum() == 0:
        return {"error": "orbit_miss"}
    if float(np.nanmax(qa[in_box])) < 0.01:
        return {"error": "cloudy"}

    background = float(np.nanmedian(xch4[in_box]))

    # Near-field with expanding radius
    xch4_near, used_radius, used_qa_thr = None, None, qa_thr
    for radius in [NEAR_DEG, 0.50, 0.75]:
        mask = (
            (lats >= lat - radius) & (lats <= lat + radius) &
            (lons >= lon - radius) & (lons <= lon + radius) &
            (qa   >= qa_thr)
        )
        if mask.sum() >= 1:
            xch4_near = xch4[mask]
            used_radius = radius
            break

    if xch4_near is None:
        return {"error": "no_near_pixels", "background_ppb": round(background, 2)}

    enhancement = float(np.nanmean(xch4_near)) - background
    return {
        "background_ppb":  round(background, 2),
        "mean_near_ppb":   round(float(np.nanmean(xch4_near)), 2),
        "max_near_ppb":    round(float(np.nanmax(xch4_near)), 2),
        "enhancement_ppb": round(enhancement, 2),
        "n_box_pixels":    int(in_box.sum()),
        "n_near_pixels":   int(mask.sum()),
        "near_radius_deg": used_radius,
        "qa_threshold":    used_qa_thr,
        "validated":       enhancement >= THRESHOLD_PPB,
    }


# ── Resume support ─────────────────────────────────────────────────────────────
def load_done() -> set:
    done = set()
    if CSV_PATH.exists():
        with open(CSV_PATH) as f:
            for row in csv.DictReader(f):
                done.add((row["site"], row["date"]))
    return done


# ── Summary generation ─────────────────────────────────────────────────────────
def write_rybnik_summary():
    """
    Rebuild rybnik_tropomi_summary.json from the full CSV (original + extension).
    Called at the end of a run to give a single clean view for the paper.
    """
    if not CSV_PATH.exists():
        return

    rybnik_rows = []
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            if row["site"] == "silesia_rybnik" and row.get("enhancement_ppb") not in ("nan", "", None):
                try:
                    rybnik_rows.append({
                        "date": row["date"],
                        "enhancement_ppb": float(row["enhancement_ppb"]),
                        "n_near_pixels":   int(row["n_near_pixels"]) if row["n_near_pixels"] else 0,
                        "validated":       row["validated"] == "True",
                        "product":         row.get("product_name", ""),
                    })
                except (ValueError, TypeError):
                    pass

    confirmed = [r for r in rybnik_rows if r["validated"]]
    strong    = [r for r in rybnik_rows if r["enhancement_ppb"] >= 5.0]

    summary = {
        "site": "silesia_rybnik",
        "coords": {"lat": SILESIAN_SITES["silesia_rybnik"]["lat"],
                   "lon": SILESIAN_SITES["silesia_rybnik"]["lon"]},
        "full_period": {
            "start": min(r["date"] for r in rybnik_rows) if rybnik_rows else None,
            "end":   max(r["date"] for r in rybnik_rows) if rybnik_rows else None,
        },
        "n_dates_with_valid_retrieval": len(rybnik_rows),
        "n_confirmed_positives_ge10ppb": len(confirmed),
        "n_moderate_positives_5to10ppb": len([r for r in strong if r["enhancement_ppb"] < 10]),
        "confirmed_events": sorted(confirmed, key=lambda x: x["date"]),
        "all_positive_events_ge5ppb": sorted(strong, key=lambda x: x["date"]),
        "max_enhancement_ppb": max((r["enhancement_ppb"] for r in rybnik_rows), default=None),
        "mean_enhancement_ppb_when_positive": (
            round(sum(r["enhancement_ppb"] for r in confirmed) / len(confirmed), 2)
            if confirmed else None
        ),
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    SUMMARY_OUT.write_text(json.dumps(summary, indent=2))
    print(f"\n✓ Rybnik summary written → {SUMMARY_OUT.name}")
    print(f"  Confirmed events (≥10 ppb): {len(confirmed)}")
    if confirmed:
        for ev in confirmed:
            print(f"    {ev['date']}  {ev['enhancement_ppb']:+.2f} ppb  n_pixels={ev['n_near_pixels']}")


# ── Date iterator ──────────────────────────────────────────────────────────────
def date_range(start_str: str, end_str: str):
    d   = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str,   "%Y-%m-%d")
    while d <= end:
        yield d.strftime("%Y-%m-%d")
        d += timedelta(days=1)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Extend TROPOMI Rybnik record to present.")
    parser.add_argument("--dry-run",      action="store_true",
                        help="Print plan without downloading anything.")
    parser.add_argument("--rybnik-only",  action="store_true",
                        help="Only process silesia_rybnik, skip the other Silesian sites.")
    parser.add_argument("--start",        default=EXTEND_START,
                        help=f"Start date (default: {EXTEND_START})")
    parser.add_argument("--end",          default=EXTEND_END,
                        help=f"End date (default: today = {EXTEND_END})")
    args = parser.parse_args()

    sites = ({"silesia_rybnik": SILESIAN_SITES["silesia_rybnik"]}
             if args.rybnik_only else SILESIAN_SITES)

    done = load_done()
    new_dates = {
        (s, d) for s in sites for d in date_range(args.start, args.end)
        if (s, d) not in done
    }

    print("=" * 70)
    print("  TROPOMI Rybnik Extension")
    print(f"  Sites:  {', '.join(sites.keys())}")
    print(f"  Period: {args.start} → {args.end}")
    print(f"  Already processed (skipping): {len(done)} site-dates")
    print(f"  New site-dates to fetch:      {len(new_dates)}")
    print("=" * 70)

    if args.dry_run:
        sample = sorted(new_dates)[:12]
        print("\nDRY RUN — sample of (site, date) pairs that would be fetched:")
        for s, d in sample:
            print(f"  {s}  {d}")
        if len(new_dates) > 12:
            print(f"  … and {len(new_dates) - 12} more")
        print("\nNo downloads performed. Remove --dry-run to execute.")
        # Still write summary from existing data
        write_rybnik_summary()
        return

    # Auth
    username = os.environ.get("COPERNICUS_USER") or input("Copernicus username: ")
    password = os.environ.get("COPERNICUS_PASS") or getpass.getpass("Password: ")
    try:
        token = get_token(username, password)
        print("  ✓ Authenticated\n")
    except Exception as e:
        print(f"  ✗ Auth failed: {e}")
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    positives = json.loads(JSON_PATH.read_text()) if JSON_PATH.exists() else []
    write_header = not CSV_PATH.exists()

    total_new   = 0
    total_pos   = 0
    token_refresh_counter = 0   # refresh token every 250 requests

    with open(CSV_PATH, "a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()

        for site_name, site_meta in sites.items():
            lat, lon = site_meta["lat"], site_meta["lon"]
            print(f"\n{'─'*60}")
            print(f"  {site_name.upper()}  ({lat:.3f}°N, {lon:.3f}°E)")
            print(f"{'─'*60}")

            for date_str in date_range(args.start, args.end):
                if (site_name, date_str) in done:
                    continue

                # Refresh token every 250 API calls (tokens expire in ~10 min)
                token_refresh_counter += 1
                if token_refresh_counter % 250 == 0:
                    try:
                        token = get_token(username, password)
                    except Exception:
                        pass  # continue with existing token

                try:
                    products = search_tropomi(lat, lon, date_str)
                except Exception as e:
                    print(f"  [{date_str}] search error: {e}")
                    continue

                if not products:
                    row = {f: "" for f in FIELDS}
                    row.update(site=site_name, date=date_str,
                               country=site_meta["country"],
                               source_type=site_meta["source_type"],
                               lat=lat, lon=lon, status="no_overpass")
                    writer.writerow(row)
                    csv_file.flush()
                    done.add((site_name, date_str))
                    continue

                best = None
                for product in products:
                    pname = product["Name"]
                    print(f"  [{date_str}] {pname[:55]}…", end=" ", flush=True)
                    try:
                        nc_path = download_product(product, token)
                    except Exception as e:
                        print(f"dl_err: {e}")
                        continue

                    result = extract_enhancement(nc_path, lat, lon)
                    if "error" in result:
                        print(result["error"])
                        continue

                    enh = result["enhancement_ppb"]
                    marker = "★ POSITIVE" if result["validated"] else ("~" if enh > 5 else "·")
                    print(f"{marker}  enh={enh:+.1f} ppb  n={result['n_near_pixels']}")

                    if best is None or enh > best.get("enhancement_ppb", -999):
                        best = {**result, "product_name": pname}

                if best is None:
                    row = {f: "" for f in FIELDS}
                    row.update(site=site_name, date=date_str,
                               country=site_meta["country"],
                               source_type=site_meta["source_type"],
                               lat=lat, lon=lon,
                               status="cloudy_or_miss",
                               product_name=products[0]["Name"][:60])
                    writer.writerow(row)
                    csv_file.flush()
                    done.add((site_name, date_str))
                    total_new += 1
                    continue

                row = {f: "" for f in FIELDS}
                row.update(
                    site=site_name, date=date_str,
                    country=site_meta["country"],
                    source_type=site_meta["source_type"],
                    lat=lat, lon=lon,
                    status="ok",
                    **{k: best[k] for k in FIELDS if k in best},
                )
                writer.writerow(row)
                csv_file.flush()
                done.add((site_name, date_str))
                total_new += 1

                if best.get("validated"):
                    total_pos += 1
                    positives.append({
                        "site":           site_name,
                        "date":           date_str,
                        "lat":            lat,
                        "lon":            lon,
                        "country":        site_meta["country"],
                        "source_type":    site_meta["source_type"],
                        "s2_tile":        site_meta.get("s2_tile", ""),
                        "background_ppb": best["background_ppb"],
                        "enhancement_ppb": best["enhancement_ppb"],
                        "max_near_ppb":   best["max_near_ppb"],
                        "n_near_pixels":  best["n_near_pixels"],
                        "qa_threshold":   best["qa_threshold"],
                        "product_name":   best["product_name"],
                    })
                    JSON_PATH.write_text(json.dumps(positives, indent=2))

    print(f"\n{'='*70}")
    print(f"  Done.  New records written: {total_new}  |  New positives: {total_pos}")
    print(f"{'='*70}")

    write_rybnik_summary()


if __name__ == "__main__":
    main()
