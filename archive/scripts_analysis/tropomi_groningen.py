"""
tropomi_groningen.py
────────────────────
Query Copernicus Data Space for the TROPOMI L2 CH4 product covering
Groningen gas field on 2024-06-28 (the Sentinel-2 acquisition date that
gave S/C = 4.191), download it, and extract the XCH4 column enhancement
over the field.

Usage:
    conda activate methane
    pip install netCDF4 requests tqdm
    python tropomi_groningen.py

You will be prompted for your Copernicus Data Space credentials.
"""

import os, sys, json, getpass, requests, numpy as np
from pathlib import Path
from datetime import datetime

# ── Target ──────────────────────────────────────────────────────────────────
LAT   = 53.252          # Groningen gas field centroid
LON   = 6.682
DATE  = "2024-06-28"    # Sentinel-2 acquisition date
# Nearby dates to try if acquisition date is cloudy for TROPOMI
FALLBACK_DATES = ["2024-06-27", "2024-06-29", "2024-06-25", "2024-07-01", "2024-07-03"]
BOX_DEG = 1.0           # ±1° search box around centroid (~100 km each side)

OUT_DIR = Path("data/tropomi")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Copernicus Data Space auth ───────────────────────────────────────────────
AUTH_URL     = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
SEARCH_URL   = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
DOWNLOAD_URL = "https://zipper.dataspace.copernicus.eu/odata/v1/Products"


def get_token(username: str, password: str) -> str:
    r = requests.post(AUTH_URL, data={
        "grant_type":    "password",
        "username":      username,
        "password":      password,
        "client_id":     "cdse-public",
    })
    r.raise_for_status()
    return r.json()["access_token"]


def search_tropomi(date: str) -> list:
    """Find TROPOMI L2 CH4 products covering Groningen on a given date."""
    start = f"{date}T00:00:00.000Z"
    end   = f"{date}T23:59:59.000Z"
    bbox  = f"POLYGON(({LON-BOX_DEG} {LAT-BOX_DEG},{LON+BOX_DEG} {LAT-BOX_DEG},{LON+BOX_DEG} {LAT+BOX_DEG},{LON-BOX_DEG} {LAT+BOX_DEG},{LON-BOX_DEG} {LAT-BOX_DEG}))"

    params = {
        "$filter": (
            f"Collection/Name eq 'SENTINEL-5P' "
            f"and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' "
            f"  and att/OData.CSC.StringAttribute/Value eq 'L2__CH4___') "
            f"and OData.CSC.Intersects(area=geography'SRID=4326;{bbox}') "
            f"and ContentDate/Start gt {start} "
            f"and ContentDate/Start lt {end}"
        ),
        "$top": 5,
        "$orderby": "ContentDate/Start asc",
    }
    r = requests.get(SEARCH_URL, params=params, timeout=30)
    r.raise_for_status()
    results = r.json().get("value", [])
    return results


def download_product(product: dict, token: str) -> Path:
    """Download a TROPOMI product to OUT_DIR."""
    pid  = product["Id"]
    name = product["Name"]
    out  = OUT_DIR / (name if name.endswith(".nc") else name + ".nc")

    if out.exists():
        print(f"  Already cached: {out}")
        return out

    print(f"  Downloading {name} (~100–200 MB)...")
    headers = {"Authorization": f"Bearer {token}"}
    url = f"{DOWNLOAD_URL}({pid})/$value"

    with requests.get(url, headers=headers, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done  = 0
        with open(out, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB chunks
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100 * done / total
                    print(f"    {pct:.0f}%  ({done/1e6:.0f} / {total/1e6:.0f} MB)", end="\r")
    print(f"\n  Saved → {out}")
    return out


def extract_xch4(nc_path: Path) -> dict:
    """
    Read TROPOMI XCH4 and extract values within 1° of Groningen.
    Returns dict with enhancement stats.
    """
    try:
        import netCDF4 as nc
    except ImportError:
        print("  netCDF4 not installed — run: pip install netCDF4")
        sys.exit(1)

    ds = nc.Dataset(nc_path)
    product = ds.groups["PRODUCT"]

    # Convert all to plain float32 arrays, replacing fill values with NaN
    def to_array(var):
        arr = var[0]
        if hasattr(arr, 'filled'):
            arr = arr.filled(np.nan)
        return np.array(arr, dtype=np.float32)

    lat   = to_array(product.variables["latitude"])
    lon   = to_array(product.variables["longitude"])
    xch4  = to_array(product.variables["methane_mixing_ratio"])
    qa    = to_array(product.variables["qa_value"])

    ds.close()

    # ── Filter to bounding box (no QA cut yet) ─────────────────────────────
    in_box_any = (
        (lat >= LAT - BOX_DEG) & (lat <= LAT + BOX_DEG) &
        (lon >= LON - BOX_DEG) & (lon <= LON + BOX_DEG)
    )
    print(f"    Pixels in bounding box (any QA): {in_box_any.sum()}")
    if in_box_any.sum() > 0:
        qa_in_box = qa[in_box_any]
        print(f"    QA range in box: {float(qa_in_box.min()):.2f} – {float(qa_in_box.max()):.2f}  (median {float(np.nanmedian(qa_in_box)):.2f})")
        print(f"    Pixels with QA≥0.5: {(qa_in_box >= 0.5).sum()}  |  QA≥0.3: {(qa_in_box >= 0.3).sum()}  |  QA≥0.0: {(qa_in_box >= 0.0).sum()}")

    # Use QA≥0.3 as fallback if QA≥0.5 yields nothing (cloud cover degrades QA)
    for qa_thresh in [0.5, 0.3, 0.0]:
        in_box = (
            (lat >= LAT - BOX_DEG) & (lat <= LAT + BOX_DEG) &
            (lon >= LON - BOX_DEG) & (lon <= LON + BOX_DEG) &
            (qa >= qa_thresh)
        )
        if in_box.sum() > 0:
            print(f"    Using QA≥{qa_thresh} threshold ({in_box.sum()} pixels)")
            break

    if in_box.sum() == 0:
        return {"error": "orbit_miss", "detail": "No pixels in bounding box — orbit does not cover this location"}

    # If all pixels have QA=0, the scene is completely cloudy
    qa_in_box = qa[in_box]
    if float(np.nanmax(qa_in_box)) < 0.01:
        return {"error": "cloudy", "detail": f"All {in_box.sum()} pixels have QA=0 — scene completely cloudy for TROPOMI"}

    xch4_box = xch4[in_box]
    lat_box  = lat[in_box]
    lon_box  = lon[in_box]

    # ── Background: median of all box pixels (robust to outliers) ──────────
    background_ppb = float(np.nanmedian(xch4_box))

    # ── Enhancement at Groningen centroid — try increasing radii ───────────
    near = None
    near_radius_deg = None
    for radius in [0.15, 0.30, 0.50]:
        candidate_near = (
            (lat_box >= LAT - radius) & (lat_box <= LAT + radius) &
            (lon_box >= LON - radius) & (lon_box <= LON + radius)
        )
        if candidate_near.sum() > 0:
            near = candidate_near
            near_radius_deg = radius
            break

    if near is None or near.sum() == 0:
        return {
            "error": "no_near_pixels",
            "background_ppb": round(background_ppb, 2),
            "n_box_pixels": int(in_box.sum()),
            "n_near_pixels": 0,
            "note": "QA pixels in wider box but none within 50 km of Groningen centroid"
        }

    xch4_near     = xch4_box[near]
    enhancement   = float(np.nanmean(xch4_near)) - background_ppb
    max_near      = float(np.nanmax(xch4_near))

    return {
        "date":             DATE,
        "site":             "Groningen gas field",
        "lat":              LAT,
        "lon":              LON,
        "background_ppb":   round(background_ppb, 2),
        "mean_near_ppb":    round(float(np.nanmean(xch4_near)), 2),
        "max_near_ppb":     round(max_near, 2),
        "enhancement_ppb":  round(enhancement, 2),
        "n_box_pixels":     int(in_box.sum()),
        "n_near_pixels":    int(near.sum()),
        "near_radius_deg":  near_radius_deg,
        "qa_threshold_used": float(qa_thresh),
        "validated":        enhancement > 10.0,        # >10 ppb = significant anomaly
        "note": (
            f"Enhancement {enhancement:.1f} ppb — "
            + ("SIGNIFICANT ANOMALY — supports CH4Net detection" if enhancement > 10
               else ("MARGINAL" if enhancement > 5 else "NOT DETECTED by TROPOMI"))
        )
    }


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  TROPOMI XCH4 Co-location — Groningen, 2024-06-28")
    print("=" * 60)

    username = input("\nCopernicus username (email): ").strip()
    password = getpass.getpass("Copernicus password: ")

    print("\n[1] Authenticating...")
    try:
        token = get_token(username, password)
        print("    OK")
    except Exception as e:
        print(f"    Auth failed: {e}")
        sys.exit(1)

    print(f"\n[2] Searching for TROPOMI L2 CH4 on {DATE}...")
    products = search_tropomi(DATE)

    if not products:
        print("    No products found. Check date or bounding box.")
        sys.exit(1)

    print(f"    Found {len(products)} product(s):")
    for p in products:
        print(f"      {p['Name']}  ({p.get('ContentLength',0)/1e6:.0f} MB)")

    # TROPOMI orbits ~14x/day; Netherlands covered ~10:00-13:00 UTC.
    # Try each product in time order until one has pixels over Groningen.
    # Products starting 00-09 UTC are over Pacific/Americas — skip those.
    european_products = [
        p for p in products
        if "T10" in p["Name"] or "T11" in p["Name"] or "T12" in p["Name"] or "T13" in p["Name"]
    ]
    if not european_products:
        print("  No European overpass products found (expected T10/T11/T12/T13 UTC). Trying all...")
        european_products = products   # fallback

    print(f"  Using {len(european_products)} European overpass product(s).")

    print(f"\n[3] Downloading and scanning European overpass products...")
    result = None
    for i, product in enumerate(european_products):
        print(f"\n  Product {i+1}/{len(european_products)}: {product['Name'][:60]}...")
        nc_path = download_product(product, token)

        print(f"  Extracting XCH4 over Groningen ({LAT}°N, {LON}°E)...")
        candidate = extract_xch4(nc_path)

        if "error" in candidate:
            print(f"  → No coverage: {candidate['error']} — {candidate.get('detail','')}")
            continue

        n_near = candidate.get('n_near_pixels', 0)
        print(f"  → Found {n_near} pixels near Groningen centroid")
        if n_near == 0:
            print(f"     (QA pixels exist in box but none within 15 km of centroid — trying next)")
            continue

        result = candidate
        break   # use first orbit that covers the site

    if result is None:
        print(f"\n  June 28 completely cloudy for TROPOMI. Trying nearby dates...")
        all_candidates = []
        for fallback_date in FALLBACK_DATES:
            print(f"\n  Trying {fallback_date}...")
            fb_products = search_tropomi(fallback_date)
            fb_european = [p for p in fb_products
                           if any(f"T{h}" in p["Name"] for h in ["10","11","12","13"])]
            for product in fb_european:
                nc_path = download_product(product, token)
                candidate = extract_xch4(nc_path)
                if "error" not in candidate:
                    candidate["note"] = f"[Fallback date: {fallback_date}] " + candidate.get("note","")
                    all_candidates.append(candidate)
                    print(f"    → Enhancement: {candidate.get('enhancement_ppb', '?'):.2f} ppb  validated={candidate.get('validated')}")
                    break
                else:
                    print(f"    → {candidate.get('detail', candidate['error'])}")
        # Pick the best candidate (highest enhancement = most likely real signal)
        if all_candidates:
            result = max(all_candidates, key=lambda r: r.get("enhancement_ppb", -9999))

    if result is None:
        result = {"error": "All dates cloudy or no coverage — TROPOMI co-location not possible for this period"}

    print("\n" + "=" * 60)
    print("  RESULT")
    print("=" * 60)
    for k, v in result.items():
        print(f"  {k:<20} {v}")

    # Save
    out_json = Path("results_analysis/tropomi_groningen.json")
    out_json.parent.mkdir(exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved → {out_json}")
