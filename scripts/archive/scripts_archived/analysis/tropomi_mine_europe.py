"""
tropomi_mine_europe.py
======================
Systematically mine Copernicus Data Space TROPOMI L2 CH4 products for
confirmed methane enhancements over known European emitter sites.

For each site × month in 2023–2024:
  1. Query TROPOMI products covering that site
  2. Filter to daytime European orbits (T10–T13 UTC)
  3. Download the product (cached — skips if already on disk)
  4. Extract XCH4 enhancement vs. local background
  5. If enhancement > THRESHOLD_PPB, log as a CONFIRMED POSITIVE and
     record the coincident Sentinel-2 tile ID

Output:
  results_analysis/tropomi_mining_results.csv  — one row per site-date
  results_analysis/tropomi_positives.json      — confirmed positives only

Usage:
  conda activate methane
  pip install netCDF4 requests tqdm
  python tropomi_mine_europe.py

Runtime: 4–12 hours depending on cloud cover (most products cloudy → fast skip).
Can be interrupted and restarted — all downloads are cached in data/tropomi/.
"""

import os, sys, json, csv, getpass, requests, numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from itertools import product as iterproduct

# ── Detection threshold ───────────────────────────────────────────────────────
THRESHOLD_PPB = 10.0      # ppb above background = "confirmed positive"
QA_THRESHOLD  = 0.5       # TROPOMI quality flag; 0.3 fallback
BOX_DEG       = 1.0       # ±1° search/background box per site
NEAR_DEG      = 0.3       # radius around centroid for near-field mean

# ── Date range ────────────────────────────────────────────────────────────────
DATE_START = "2023-01-01"
DATE_END   = "2024-09-30"

# ── European emitter sites ────────────────────────────────────────────────────
# Grouped by source type. Coordinates from literature / TROPOMI anomaly maps.
SITES = {

    # ── Upper Silesian Coal Mine Methane (Poland) ─────────────────────────────
    # TROPOMI confirmed top-5 European CMM emitters (Lavoie et al. 2022, GRL)
    "silesia_jastrzebie": dict(
        lat=49.950, lon=18.590,
        s2_tile="T34UCB",
        source_type="coal_mine",
        country="PL",
        note="Jastrzebie-Zdroj mining complex — JSW S.A., active ventilation shafts",
    ),
    "silesia_knurow": dict(
        lat=50.220, lon=18.670,
        s2_tile="T34UCA",
        source_type="coal_mine",
        country="PL",
        note="Knurow-Szczyglowice mine (JSW) — documented TROPOMI anomalies 2019–2022",
    ),
    "silesia_pniowek": dict(
        lat=49.930, lon=18.640,
        s2_tile="T34UCB",
        source_type="coal_mine",
        country="PL",
        note="Pniowek mine (JSW) — among highest CMM ventilation rates in Poland",
    ),
    "silesia_zofiowka": dict(
        lat=49.980, lon=18.600,
        s2_tile="T34UCB",
        source_type="coal_mine",
        country="PL",
        note="Zofiowka mine (JSW) — active deep coal mine, persistent TROPOMI signal",
    ),
    "silesia_rybnik": dict(
        lat=50.135, lon=18.522,
        s2_tile="T34UCA",
        source_type="coal_mine_plant",
        country="PL",
        note="Rybnik area — Rybnik power plant + Chwałowice/Jankowice mines co-located",
    ),

    # ── Groningen gas field (Netherlands) ─────────────────────────────────────
    "groningen": dict(
        lat=53.252, lon=6.682,
        s2_tile="T31UGV",
        source_type="gas_field",
        country="NL",
        note="Largest European gas field, curtailed 2023-2024. Our strongest S2 detection.",
    ),

    # ── Dutch gas transmission network ────────────────────────────────────────
    "nl_grijpskerk": dict(
        lat=53.270, lon=6.260,
        s2_tile="T31UGV",
        source_type="gas_compressor",
        country="NL",
        note="Grijpskerk compressor station (Gasunie) — major injection/withdrawal point",
    ),
    "nl_bergermeer": dict(
        lat=52.680, lon=4.700,
        s2_tile="T31UFU",
        source_type="gas_storage",
        country="NL",
        note="Bergermeer underground gas storage (TAQA) — largest UGS in NL",
    ),
    "nl_ravenstein": dict(
        lat=51.770, lon=5.650,
        s2_tile="T31UGS",
        source_type="gas_compressor",
        country="NL",
        note="Ravenstein compressor (Gasunie) — high-pressure trunk line node",
    ),

    # ── German gas transmission (ONTRAS / GASCADE corridor) ──────────────────
    "de_bad_lauchstaedt": dict(
        lat=51.390, lon=11.880,
        s2_tile="T32UQC",
        source_type="gas_compressor",
        country="DE",
        note="Bad Lauchstädt compressor station (ONTRAS) — persistent TROPOMI signal in literature",
    ),
    "de_sayda": dict(
        lat=50.740, lon=13.400,
        s2_tile="T33UUR",
        source_type="gas_compressor",
        country="DE",
        note="Sayda/Olbernhau compressor (GASCADE) — Thuringian corridor",
    ),

    # ── Romanian gas infrastructure ───────────────────────────────────────────
    "ro_medias": dict(
        lat=46.150, lon=24.370,
        s2_tile="T34TGR",
        source_type="gas_processing",
        country="RO",
        note="Medias gas processing (Romgaz) — Transylvania basin, known fugitive emissions",
    ),
    "ro_totea": dict(
        lat=44.480, lon=25.620,
        s2_tile="T35TLK",
        source_type="gas_compressor",
        country="RO",
        note="Totea compressor (Transgaz) — main transit corridor Moldova→Bulgaria",
    ),

    # ── French gas infrastructure ─────────────────────────────────────────────
    "fr_lacq": dict(
        lat=43.430, lon=-0.630,
        s2_tile="T30TXN",
        source_type="gas_processing",
        country="FR",
        note="Lacq gas processing complex (TotalEnergies) — historically largest French gas emitter",
    ),

    # ── UK North Sea / onshore ────────────────────────────────────────────────
    "uk_bacton": dict(
        lat=52.860, lon=1.480,
        s2_tile="T31UCB",
        source_type="gas_terminal",
        country="UK",
        note="Bacton gas terminal (BP/Shell) — major North Sea landing point",
    ),
}

# ── Sentinel-2 tile lookup by lat/lon ─────────────────────────────────────────
# For sites not matching the pre-filled tile ID we query the Copernicus catalog.
# This function returns the tile ID for a Sentinel-2 L1C product overlapping
# the given lat/lon on a given date (used when tif has been downloaded).
def query_s2_tile(lat, lon, date_str, token):
    """Return Sentinel-2 tile ID overlapping lat/lon on date_str, or None."""
    start = f"{date_str}T00:00:00.000Z"
    end   = f"{date_str}T23:59:59.000Z"
    bbox  = f"POLYGON(({lon-0.1} {lat-0.1},{lon+0.1} {lat-0.1},{lon+0.1} {lat+0.1},{lon-0.1} {lat+0.1},{lon-0.1} {lat-0.1}))"
    params = {
        "$filter": (
            f"Collection/Name eq 'SENTINEL-2' "
            f"and Attributes/OData.CSC.StringAttribute/any(a:a/Name eq 'productType' "
            f"  and a/OData.CSC.StringAttribute/Value eq 'S2MSI1C') "
            f"and OData.CSC.Intersects(area=geography'SRID=4326;{bbox}') "
            f"and ContentDate/Start gt {start} "
            f"and ContentDate/Start lt {end}"
        ),
        "$top": 1,
        "$select": "Name",
    }
    try:
        r = requests.get(
            "https://catalogue.dataspace.copernicus.eu/odata/v1/Products",
            params=params, timeout=20,
            headers={"Authorization": f"Bearer {token}"},
        )
        items = r.json().get("value", [])
        if items:
            name = items[0]["Name"]          # e.g. S2A_MSIL1C_20240628T105621_..._T31UGV_...
            parts = name.split("_")
            tile = next((p for p in parts if p.startswith("T") and len(p) == 6), None)
            return tile
    except Exception:
        pass
    return None


# ── Auth ──────────────────────────────────────────────────────────────────────
AUTH_URL     = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
SEARCH_URL   = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
DOWNLOAD_URL = "https://zipper.dataspace.copernicus.eu/odata/v1/Products"

def get_token(username, password):
    r = requests.post(AUTH_URL, data={
        "grant_type": "password", "username": username,
        "password": password, "client_id": "cdse-public",
    })
    r.raise_for_status()
    return r.json()["access_token"]


# ── TROPOMI product search ────────────────────────────────────────────────────
def search_tropomi_site(lat, lon, date_str):
    """
    Query all TROPOMI L2 CH4 products covering (lat, lon) on date_str.
    Returns list of product dicts (may be empty if no overpass or cloudy).
    """
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
    european = [p for p in products
                if any(f"T{h:02d}" in p["Name"] for h in range(9, 14))]
    return european


def download_product(product, token, out_dir):
    pid  = product["Id"]
    name = product["Name"]
    out  = out_dir / (name if name.endswith(".nc") else name + ".nc")
    if out.exists():
        return out
    headers = {"Authorization": f"Bearer {token}"}
    url = f"{DOWNLOAD_URL}({pid})/$value"
    with requests.get(url, headers=headers, stream=True, timeout=180) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done  = 0
        with open(out, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"    {100*done/total:.0f}%", end="\r")
    print()
    return out


# ── XCH4 extraction ───────────────────────────────────────────────────────────
def extract_enhancement(nc_path, lat, lon):
    """
    Extract XCH4 enhancement over (lat, lon) from a TROPOMI L2 product.
    Returns dict with enhancement stats, or {"error": ...}.
    """
    import netCDF4 as nc_lib

    def to_array(var):
        arr = var[0]
        if hasattr(arr, "filled"):
            arr = arr.filled(np.nan)
        return np.array(arr, dtype=np.float32)

    try:
        ds  = nc_lib.Dataset(nc_path)
        grp = ds.groups["PRODUCT"]
        lat_arr  = to_array(grp.variables["latitude"])
        lon_arr  = to_array(grp.variables["longitude"])
        xch4_arr = to_array(grp.variables["methane_mixing_ratio"])
        qa_arr   = to_array(grp.variables["qa_value"])
        ds.close()
    except Exception as e:
        return {"error": f"read_failed: {e}"}

    # Background box ── 1° around site, best available QA
    for qa_thr in [QA_THRESHOLD, 0.3, 0.0]:
        in_box = (
            (lat_arr >= lat - BOX_DEG) & (lat_arr <= lat + BOX_DEG) &
            (lon_arr >= lon - BOX_DEG) & (lon_arr <= lon + BOX_DEG) &
            (qa_arr  >= qa_thr)
        )
        if in_box.sum() >= 3:
            break

    if in_box.sum() == 0:
        return {"error": "orbit_miss"}

    if float(np.nanmax(qa_arr[in_box])) < 0.01:
        return {"error": "cloudy"}

    xch4_box = xch4_arr[in_box]
    background = float(np.nanmedian(xch4_box))

    # Near-field ── expanding radius
    xch4_near_vals = None
    used_radius    = None
    for radius in [NEAR_DEG, 0.50, 0.75]:
        mask = (
            (lat_arr >= lat - radius) & (lat_arr <= lat + radius) &
            (lon_arr >= lon - radius) & (lon_arr <= lon + radius) &
            (qa_arr  >= qa_thr)
        )
        if mask.sum() >= 1:
            xch4_near_vals = xch4_arr[mask]
            used_radius    = radius
            break

    if xch4_near_vals is None:
        return {"error": "no_near_pixels", "background_ppb": round(background, 2)}

    enhancement = float(np.nanmean(xch4_near_vals)) - background
    return {
        "background_ppb":  round(background, 2),
        "mean_near_ppb":   round(float(np.nanmean(xch4_near_vals)), 2),
        "max_near_ppb":    round(float(np.nanmax(xch4_near_vals)), 2),
        "enhancement_ppb": round(enhancement, 2),
        "n_box_pixels":    int(in_box.sum()),
        "n_near_pixels":   int(mask.sum()),
        "near_radius_deg": used_radius,
        "qa_threshold":    qa_thr,
        "validated":       enhancement >= THRESHOLD_PPB,
    }


# ── Date range iterator ───────────────────────────────────────────────────────
def date_range(start_str, end_str):
    """Yield date strings from start to end (inclusive), daily."""
    d = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d")
    while d <= end:
        yield d.strftime("%Y-%m-%d")
        d += timedelta(days=1)


# ── Resume support ────────────────────────────────────────────────────────────
def load_done(csv_path):
    """Return set of (site, date) pairs already processed."""
    done = set()
    if csv_path.exists():
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                done.add((row["site"], row["date"]))
    return done


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    out_dir = Path("data/tropomi")
    out_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path("results_analysis")
    results_dir.mkdir(exist_ok=True)

    csv_path  = results_dir / "tropomi_mining_results.csv"
    json_path = results_dir / "tropomi_positives.json"

    # ── CSV header ────────────────────────────────────────────────────────────
    FIELDS = [
        "site", "date", "country", "source_type", "lat", "lon",
        "product_name", "status",
        "background_ppb", "mean_near_ppb", "max_near_ppb",
        "enhancement_ppb", "n_box_pixels", "n_near_pixels",
        "near_radius_deg", "qa_threshold",
        "validated", "s2_tile",
    ]

    write_header = not csv_path.exists()
    done = load_done(csv_path)

    print("=" * 70)
    print("  TROPOMI European Methane Mining")
    print(f"  Sites: {len(SITES)}  |  Period: {DATE_START} → {DATE_END}")
    print(f"  Threshold: {THRESHOLD_PPB} ppb above background")
    print(f"  Resume: {len(done)} site-dates already processed")
    print("=" * 70)

    # Credentials: env vars preferred for nohup/background runs
    username = os.environ.get("COPERNICUS_USER") or input("Copernicus Data Space username: ")
    password = os.environ.get("COPERNICUS_PASS") or getpass.getpass("Password: ")
    token    = get_token(username, password)
    print("  ✓ Authenticated\n")

    positives = []
    if json_path.exists():
        with open(json_path) as f:
            positives = json.load(f)

    with open(csv_path, "a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()

        total_sites = len(SITES)
        for site_i, (site_name, site_meta) in enumerate(SITES.items(), 1):
            lat = site_meta["lat"]
            lon = site_meta["lon"]
            print(f"\n[{site_i}/{total_sites}] {site_name.upper()} "
                  f"({site_meta['country']}, {site_meta['source_type']})")
            print(f"  Coords: {lat:.3f}°N, {lon:.3f}°E")

            site_positives = 0
            site_processed = 0

            for date_str in date_range(DATE_START, DATE_END):
                if (site_name, date_str) in done:
                    continue

                # ── Query TROPOMI ─────────────────────────────────────────────
                try:
                    products = search_tropomi_site(lat, lon, date_str)
                except Exception as e:
                    print(f"  [{date_str}] Search error: {e}")
                    continue

                if not products:
                    # No overpass today — write a skip row quietly
                    row = {f: "" for f in FIELDS}
                    row.update(site=site_name, date=date_str,
                               country=site_meta["country"],
                               source_type=site_meta["source_type"],
                               lat=lat, lon=lon, status="no_overpass")
                    writer.writerow(row)
                    csv_file.flush()
                    done.add((site_name, date_str))
                    continue

                # ── Process each overpass ─────────────────────────────────────
                best = None  # best result for this site-date
                for product in products:
                    pname = product["Name"]
                    print(f"  [{date_str}] {pname[:50]}...", end=" ", flush=True)

                    try:
                        nc_path = download_product(product, token, out_dir)
                    except Exception as e:
                        print(f"download_err: {e}")
                        continue

                    result = extract_enhancement(nc_path, lat, lon)

                    if "error" in result:
                        print(result["error"])
                        continue

                    enh = result["enhancement_ppb"]
                    validated = result["validated"]
                    marker = "★ POSITIVE" if validated else ("~" if enh > 5 else "·")
                    print(f"{marker}  enh={enh:+.1f} ppb  n={result['n_near_pixels']}")

                    if best is None or enh > best.get("enhancement_ppb", -999):
                        best = {**result, "product_name": pname}

                if best is None:
                    row = {f: "" for f in FIELDS}
                    row.update(site=site_name, date=date_str,
                               country=site_meta["country"],
                               source_type=site_meta["source_type"],
                               lat=lat, lon=lon, status="cloudy_or_miss",
                               product_name=products[0]["Name"][:60])
                    writer.writerow(row)
                    csv_file.flush()
                    done.add((site_name, date_str))
                    site_processed += 1
                    continue

                # ── Sentinel-2 tile lookup for positives ──────────────────────
                s2_tile = site_meta.get("s2_tile", "")
                if best["validated"] and not s2_tile:
                    s2_tile = query_s2_tile(lat, lon, date_str, token) or ""

                row = {f: "" for f in FIELDS}
                row.update(
                    site=site_name, date=date_str,
                    country=site_meta["country"],
                    source_type=site_meta["source_type"],
                    lat=lat, lon=lon,
                    product_name=best["product_name"][:80],
                    status="ok",
                    background_ppb=best["background_ppb"],
                    mean_near_ppb=best["mean_near_ppb"],
                    max_near_ppb=best["max_near_ppb"],
                    enhancement_ppb=best["enhancement_ppb"],
                    n_box_pixels=best["n_box_pixels"],
                    n_near_pixels=best["n_near_pixels"],
                    near_radius_deg=best["near_radius_deg"],
                    qa_threshold=best["qa_threshold"],
                    validated=best["validated"],
                    s2_tile=s2_tile,
                )
                writer.writerow(row)
                csv_file.flush()
                done.add((site_name, date_str))
                site_processed += 1

                if best["validated"]:
                    site_positives += 1
                    pos_entry = {
                        "site": site_name,
                        "date": date_str,
                        "lat": lat, "lon": lon,
                        "country": site_meta["country"],
                        "source_type": site_meta["source_type"],
                        "s2_tile": s2_tile,
                        **{k: best[k] for k in [
                            "background_ppb", "enhancement_ppb",
                            "max_near_ppb", "n_near_pixels",
                            "qa_threshold", "product_name",
                        ]},
                    }
                    positives.append(pos_entry)
                    with open(json_path, "w") as jf:
                        json.dump(positives, jf, indent=2)
                    print(f"    ★ Logged positive → {json_path}")

            print(f"  Done: {site_processed} dates processed, "
                  f"{site_positives} positives found")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  MINING COMPLETE")
    print(f"  Total confirmed positives: {len(positives)}")
    print(f"  Results CSV:  {csv_path}")
    print(f"  Positives JSON: {json_path}")

    if positives:
        print("\n  Confirmed positive detections:")
        print(f"  {'Site':<28} {'Date':<12} {'Enh (ppb)':>10}  S2 Tile")
        print("  " + "-" * 65)
        for p in sorted(positives, key=lambda x: x["enhancement_ppb"], reverse=True):
            print(f"  {p['site']:<28} {p['date']:<12} "
                  f"{p['enhancement_ppb']:>+10.1f}  {p['s2_tile']}")
    print("=" * 70)
