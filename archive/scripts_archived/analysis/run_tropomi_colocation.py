#!/usr/bin/env python3
"""
TROPOMI CH4 Co-location — Groningen Gas Field
==============================================
Downloads TROPOMI Sentinel-5P L2 CH4 products for the same acquisition
date as the Sentinel-2 Groningen detection (2024-08-17) and checks for
co-located methane enhancement.

This transforms the 2.95× S/C result from a correlation into a validated
detection — the key step before any regression / quantification work.

Usage:
    python run_tropomi_colocation.py                   # full run
    python run_tropomi_colocation.py --search-only     # list products, no download
    python run_tropomi_colocation.py --date 2024-08-17 # explicit date override
    python run_tropomi_colocation.py --days-window 3   # ± days around S2 date

Output:
    results_tropomi/
        <product_name>.nc          — downloaded TROPOMI L2 file
        groningen_colocation.json  — statistics + verdict
        groningen_ch4_map.png      — spatial CH4 map (if matplotlib available)

Requirements (all in the `methane` conda env):
    netCDF4 or h5py, numpy, requests, python-dotenv
    matplotlib (optional, for map plot)
"""

import os
import sys
import json
import math
import glob
import argparse
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

# ── Constants ─────────────────────────────────────────────────────────────────

# Groningen gas field — Slochteren structure centre
GRONINGEN_LAT = 53.22
GRONINGEN_LON = 6.78
# Bounding box for CH4 extraction (west, south, east, north)
GRONINGEN_BBOX = (5.8, 52.7, 7.8, 53.8)

# Reference: clean North Sea strip west of Netherlands, no known point sources
REFERENCE_BBOX = (3.5, 53.0, 5.0, 54.2)
REFERENCE_LABEL = "North Sea reference (clean)"

# Minimum TROPOMI quality flag (0–1). ESA recommends ≥ 0.5 for science use.
QA_MIN = 0.5

S2_ACQUISITION_DATE = "2024-08-17"   # from Groningen events_*.json
RESULTS_DIR = "results_tropomi"

CDSE_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu"
    "/auth/realms/CDSE/protocol/openid-connect/token"
)
CDSE_CATALOGUE = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
CDSE_DOWNLOAD  = "https://zipper.dataspace.copernicus.eu/odata/v1/Products"


# ── Auth ──────────────────────────────────────────────────────────────────────

def get_token(user: str, password: str) -> str:
    """Fetch OAuth2 access token from CDSE."""
    r = requests.post(
        CDSE_TOKEN_URL,
        data={
            "grant_type":  "password",
            "username":    user,
            "password":    password,
            "client_id":   "cdse-public",
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["access_token"]


# ── Catalogue search ──────────────────────────────────────────────────────────

def search_tropomi_ch4(date_str: str, days_window: int = 1) -> list:
    """
    Search CDSE catalogue for TROPOMI L2 CH4 products covering Groningen
    within ± days_window of date_str.

    Returns list of dicts with id, name, size, beginningDateTime.
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    start = (dt - timedelta(days=days_window)).strftime("%Y-%m-%dT00:00:00.000Z")
    end   = (dt + timedelta(days=days_window)).strftime("%Y-%m-%dT23:59:59.000Z")

    # WKT point over Groningen for spatial filter
    wkt = f"POINT({GRONINGEN_LON} {GRONINGEN_LAT})"

    params = {
        "$filter": (
            "Collection/Name eq 'SENTINEL-5P' and "
            "Attributes/OData.CSC.StringAttribute/any("
            "  att:att/Name eq 'productType' and "
            "  att/OData.CSC.StringAttribute/Value eq 'L2__CH4___'"
            ") and "
            f"OData.CSC.Intersects(area=geography'SRID=4326;{wkt}') and "
            f"ContentDate/Start gt {start} and "
            f"ContentDate/Start lt {end}"
        ),
        "$orderby": "ContentDate/Start asc",
        "$top": 20,
    }

    r = requests.get(CDSE_CATALOGUE, params=params, timeout=30)
    r.raise_for_status()
    items = r.json().get("value", [])

    results = []
    for item in items:
        size_mb = item.get("ContentLength", 0) / 1e6
        results.append({
            "id":   item["Id"],
            "name": item["Name"],
            "size_mb": round(size_mb, 1),
            "date": item["ContentDate"]["Start"][:10],
            "time": item["ContentDate"]["Start"][11:19],
        })
    return results


# ── Download ──────────────────────────────────────────────────────────────────

def download_product(product_id: str, product_name: str, token: str,
                     out_dir: str) -> str:
    """
    Stream-download a TROPOMI product from CDSE zipper.
    Returns local file path.
    """
    out_path = os.path.join(out_dir, product_name + ".nc")
    if os.path.exists(out_path):
        print(f"  [CACHED] {product_name}.nc already downloaded.")
        return out_path

    url = f"{CDSE_DOWNLOAD}({product_id})/$value"
    headers = {"Authorization": f"Bearer {token}"}

    print(f"  Downloading {product_name}")
    print(f"  URL: {url}")

    r = requests.get(url, headers=headers, stream=True, timeout=120)
    r.raise_for_status()

    total = int(r.headers.get("Content-Length", 0))
    downloaded = 0
    chunk_size = 1024 * 1024  # 1 MB chunks

    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  Progress: {downloaded/1e6:.1f} / {total/1e6:.1f} MB  ({pct:.0f}%)",
                          end="", flush=True)
    print()
    print(f"  Saved → {out_path}")
    return out_path


# ── CH4 extraction ────────────────────────────────────────────────────────────

def extract_ch4(nc_path: str, bbox: tuple, qa_min: float = 0.5, verbose: bool = True):
    """
    Extract bias-corrected CH4 mixing ratios (ppb) inside bbox,
    filtered by qa_value >= qa_min.

    Returns (values_array, n_total_pixels, n_valid_pixels).
    Tries netCDF4 first, falls back to h5py.
    """
    west, south, east, north = bbox

    try:
        import netCDF4 as nc
        with nc.Dataset(nc_path, "r") as ds:
            grp = ds["PRODUCT"]
            lat = grp["latitude"][0].data          # shape: (scanline, pixel)
            lon = grp["longitude"][0].data
            ch4 = grp["methane_mixing_ratio_bias_corrected"][0].data
            qa  = grp["qa_value"][0].data

        if verbose:
            bbox_mask = (lat >= south) & (lat <= north) & (lon >= west) & (lon <= east)
            n_bbox = int(bbox_mask.sum())
            qa_in_bbox = qa[bbox_mask]
            print(f"   Pixels in bbox (no QA filter): {n_bbox:,}")
            if n_bbox > 0:
                print(f"   QA range in bbox: {float(qa_in_bbox.min()):.3f} – {float(qa_in_bbox.max()):.3f}  "
                      f"(mean {float(qa_in_bbox.mean()):.3f})")
                print(f"   QA ≥ {qa_min} in bbox: {int((qa_in_bbox >= qa_min).sum()):,}")

        mask = (
            (lat >= south) & (lat <= north) &
            (lon >= west)  & (lon <= east)  &
            (qa >= qa_min)
        )
        values = ch4[mask]
        # Remove fill values (usually 9.969e+36)
        values = values[values < 1e10]
        return values, int(mask.size), int(mask.sum())

    except ImportError:
        pass  # try h5py

    try:
        import h5py
        with h5py.File(nc_path, "r") as f:
            grp = f["PRODUCT"]
            lat = grp["latitude"][0][:]
            lon = grp["longitude"][0][:]
            ch4 = grp["methane_mixing_ratio_bias_corrected"][0][:]
            qa  = grp["qa_value"][0][:]

        if verbose:
            bbox_mask = (lat >= south) & (lat <= north) & (lon >= west) & (lon <= east)
            n_bbox = int(bbox_mask.sum())
            qa_in_bbox = qa[bbox_mask]
            print(f"   Pixels in bbox (no QA filter): {n_bbox:,}")
            if n_bbox > 0:
                print(f"   QA range in bbox: {float(qa_in_bbox.min()):.3f} – {float(qa_in_bbox.max()):.3f}  "
                      f"(mean {float(qa_in_bbox.mean()):.3f})")
                print(f"   QA ≥ {qa_min} in bbox: {int((qa_in_bbox >= qa_min).sum()):,}")

        mask = (
            (lat >= south) & (lat <= north) &
            (lon >= west)  & (lon <= east)  &
            (qa >= qa_min)
        )
        values = ch4[mask]
        values = values[values < 1e10]
        return values, int(mask.size), int(mask.sum())

    except ImportError:
        sys.exit(
            "ERROR: Neither netCDF4 nor h5py is installed.\n"
            "Install one: conda install -c conda-forge netCDF4\n"
            "        or:  conda install h5py"
        )


# ── Map plot (optional) ───────────────────────────────────────────────────────

def plot_ch4_map(nc_path: str, out_dir: str, s2_date: str):
    """
    Save a simple CH4 column map over the Netherlands / North Sea region.
    Silently skips if matplotlib is not installed.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("  (matplotlib not available — skipping map plot)")
        return

    try:
        import netCDF4 as nc
        with nc.Dataset(nc_path, "r") as ds:
            grp = ds["PRODUCT"]
            lat = grp["latitude"][0].data
            lon = grp["longitude"][0].data
            ch4 = grp["methane_mixing_ratio_bias_corrected"][0].data
            qa  = grp["qa_value"][0].data
    except ImportError:
        try:
            import h5py
            with h5py.File(nc_path, "r") as f:
                grp = f["PRODUCT"]
                lat = grp["latitude"][0][:]
                lon = grp["longitude"][0][:]
                ch4 = grp["methane_mixing_ratio_bias_corrected"][0][:]
                qa  = grp["qa_value"][0][:]
        except ImportError:
            return

    # Crop to Netherlands + surroundings
    region = (3.0, 52.0, 8.5, 54.5)
    west, south, east, north = region
    mask = (lat >= south) & (lat <= north) & (lon >= west) & (lon <= east) & (qa >= QA_MIN)
    mask &= (ch4 < 1e10)

    if mask.sum() < 5:
        print("  Too few valid pixels in region for map.")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(
        lon[mask], lat[mask], c=ch4[mask],
        s=30, cmap="YlOrRd",
        vmin=np.percentile(ch4[mask], 5),
        vmax=np.percentile(ch4[mask], 95),
        alpha=0.85, edgecolors="none",
    )
    plt.colorbar(sc, ax=ax, label="CH₄ mixing ratio (ppb, bias-corrected)")

    # Mark Groningen gas field
    ax.scatter(GRONINGEN_LON, GRONINGEN_LAT, marker="*", s=200,
               color="blue", zorder=5, label="Groningen gas field")
    ax.annotate("Groningen", (GRONINGEN_LON, GRONINGEN_LAT),
                textcoords="offset points", xytext=(6, 6), fontsize=9, color="blue")

    # Draw bounding boxes
    from matplotlib.patches import Rectangle
    gw, gs, ge, gn = GRONINGEN_BBOX
    rw, rs, re_, rn = REFERENCE_BBOX
    ax.add_patch(Rectangle((gw, gs), ge-gw, gn-gs,
                            fill=False, edgecolor="blue", lw=1.5,
                            linestyle="--", label="Groningen analysis box"))
    ax.add_patch(Rectangle((rw, rs), re_-rw, rn-rs,
                            fill=False, edgecolor="grey", lw=1.5,
                            linestyle="--", label="Reference (clean)"))

    ax.set_xlim(west, east); ax.set_ylim(south, north)
    ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
    ax.set_title(
        f"TROPOMI L2 CH₄ — {s2_date}\n"
        f"Netherlands / North Sea  (QA ≥ {QA_MIN})",
        fontsize=11,
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    out_png = os.path.join(out_dir, f"groningen_ch4_map_{s2_date}.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"  Map saved → {out_png}")


# ── Stats ─────────────────────────────────────────────────────────────────────

def compute_stats(values: np.ndarray) -> dict:
    if len(values) == 0:
        return {"n": 0, "mean": None, "median": None, "std": None,
                "p25": None, "p75": None, "min": None, "max": None}
    return {
        "n":      int(len(values)),
        "mean":   round(float(np.mean(values)), 2),
        "median": round(float(np.median(values)), 2),
        "std":    round(float(np.std(values)), 2),
        "p25":    round(float(np.percentile(values, 25)), 2),
        "p75":    round(float(np.percentile(values, 75)), 2),
        "min":    round(float(np.min(values)), 2),
        "max":    round(float(np.max(values)), 2),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TROPOMI CH4 co-location — Groningen")
    parser.add_argument("--date", default=S2_ACQUISITION_DATE,
                        help=f"S2 acquisition date (default: {S2_ACQUISITION_DATE})")
    parser.add_argument("--days-window", type=int, default=1,
                        help="Search ± N days around the S2 date (default: 1)")
    parser.add_argument("--search-only", action="store_true",
                        help="List available TROPOMI products without downloading")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip CH4 map generation")
    parser.add_argument("--qa-min", type=float, default=QA_MIN,
                        help=f"QA threshold (default: {QA_MIN}). Use 0 to disable filtering for diagnostics.")
    parser.add_argument("--product-index", type=int, default=None,
                        help="Force-select a specific product by index (0=first listed). "
                             "Default: auto-select nearest date. Use with --search-only first to see indices.")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 70)
    print("  TROPOMI CH4 Co-location — Groningen Gas Field")
    print("=" * 70)
    print(f"  S2 acquisition date : {args.date}")
    print(f"  Search window       : ± {args.days_window} day(s)")
    print(f"  Groningen centre    : {GRONINGEN_LAT}°N, {GRONINGEN_LON}°E")
    print(f"  QA threshold        : ≥ {args.qa_min}")
    print()

    # ── 1. Search catalogue ──
    print("▶  Searching CDSE catalogue for TROPOMI L2 CH4 products …")
    products = search_tropomi_ch4(args.date, days_window=args.days_window)

    if not products:
        print("  No TROPOMI CH4 products found for this date/location.")
        print("  Try --days-window 3 to widen the search.")
        sys.exit(1)

    print(f"  Found {len(products)} product(s):")
    for p in products:
        print(f"    {p['date']} {p['time']} UTC  |  {p['size_mb']:6.0f} MB  |  {p['name'][:60]}")

    if args.search_only:
        print("\n  [--search-only] Exiting before download.")
        sys.exit(0)

    # ── 2. Load credentials ──
    load_dotenv("config/.env")
    user     = os.getenv("COPERNICUS_USER")
    password = os.getenv("COPERNICUS_PASSWORD")
    if not user or not password:
        sys.exit(
            "ERROR: COPERNICUS_USER / COPERNICUS_PASSWORD not found in config/.env"
        )

    print("\n▶  Authenticating with CDSE …")
    token = get_token(user, password)
    print("  Token acquired.")

    # ── 3. Pick best product — auto-scan if needed ──
    target_dt = datetime.strptime(args.date, "%Y-%m-%d")
    def date_dist(p):
        d = datetime.strptime(p["date"], "%Y-%m-%d")
        return abs((d - target_dt).days)

    if args.product_index is not None:
        if args.product_index >= len(products):
            sys.exit(f"ERROR: --product-index {args.product_index} out of range (0–{len(products)-1})")
        candidates = [products[args.product_index]]
    else:
        products.sort(key=date_dist)
        candidates = products  # try all, nearest date first

    nc_path = None
    gron_vals = ref_vals = np.array([])
    gron_total = gron_valid = ref_total = ref_valid = 0
    best = None

    for candidate in candidates:
        delta = date_dist(candidate)
        print(f"\n▶  Trying: {candidate['name'][:60]}")
        print(f"   Date: {candidate['date']} {candidate['time']} UTC  |  "
              f"{candidate['size_mb']} MB  |  {delta} day(s) from S2")

        nc_path_try = download_product(candidate["id"], candidate["name"], token, RESULTS_DIR)

        # Quick check: does this orbit have any QA-valid pixels over Groningen?
        gv, gt, gvld = extract_ch4(nc_path_try, GRONINGEN_BBOX, args.qa_min, verbose=True)
        print(f"   Pixels in box: {gt:,}  |  QA-valid: {gvld:,}  |  Clean: {len(gv):,}")

        if len(gv) >= 10:  # at least 10 real CH4 retrievals over the field
            rv, rt, rvld = extract_ch4(nc_path_try, REFERENCE_BBOX, args.qa_min, verbose=True)
            print(f"   Reference — Pixels in box: {rt:,}  |  QA-valid: {rvld:,}  |  Clean: {len(rv):,}")
            if len(rv) >= 4:
                print(f"   ✓ Sufficient coverage — using this product.")
                nc_path = nc_path_try
                gron_vals, gron_total, gron_valid = gv, gt, gvld
                ref_vals,  ref_total,  ref_valid  = rv, rt, rvld
                best = candidate
                break
            else:
                print(f"   ✗ Reference area too sparse (n={len(rv)}) — trying next orbit.")
        else:
            print(f"   ✗ Insufficient Groningen coverage (n={len(gv)}) — trying next orbit.")

    if best is None:
        print("\n  No product with sufficient clear-sky coverage found in this window.")
        print("  Try --days-window 14 or a different season (June/July tends to be clearer).")
        sys.exit(1)

    # ── 6. Statistics ──
    gron_stats = compute_stats(gron_vals)
    ref_stats  = compute_stats(ref_vals)

    print("\n" + "─" * 70)
    print("  RESULTS")
    print("─" * 70)

    if gron_stats["mean"] and ref_stats["mean"]:
        enhancement_ppb  = round(gron_stats["mean"] - ref_stats["mean"], 2)
        enhancement_pct  = round(enhancement_ppb / ref_stats["mean"] * 100, 1)
        tropomi_sc_ratio = round(gron_stats["mean"] / ref_stats["mean"], 3)

        print(f"  Groningen mean CH4   : {gron_stats['mean']:.1f} ppb  (n={gron_stats['n']})")
        print(f"  Reference mean CH4   : {ref_stats['mean']:.1f} ppb  (n={ref_stats['n']})")
        print(f"  Enhancement          : +{enhancement_ppb:.1f} ppb  ({enhancement_pct:+.1f}%)")
        print(f"  TROPOMI S/C ratio    : {tropomi_sc_ratio:.3f}×")
        print()
        print(f"  Sentinel-2 S/C ratio : 2.950×  (CH4Net v2, same tile date)")
        print()

        # Verdict
        if tropomi_sc_ratio >= 1.05:
            verdict = "VALIDATED ✓"
            interpretation = (
                f"TROPOMI shows {enhancement_ppb:.1f} ppb enhancement over Groningen vs. "
                f"the North Sea reference on {best['date']}. "
                f"Combined with the Sentinel-2 2.95× S/C ratio on the same date, "
                f"this constitutes a co-validated methane detection — "
                f"two independent sensors agree on the same emission event."
            )
        elif tropomi_sc_ratio >= 1.01:
            verdict = "MARGINAL — weak TROPOMI signal"
            interpretation = (
                f"TROPOMI shows only {enhancement_ppb:.1f} ppb enhancement — within normal "
                f"variability. Groningen is a distributed low-flux field; the plume may "
                f"have been too dilute at TROPOMI's 5×3.5 km resolution. The S2 detection "
                f"remains the stronger evidence but is not independently confirmed."
            )
        else:
            verdict = "NOT CONFIRMED — no TROPOMI enhancement"
            interpretation = (
                f"TROPOMI shows no enhancement on {best['date']}. Possible causes: "
                f"(1) clouds/aerosols reduced TROPOMI coverage, (2) the field was not "
                f"actively venting on this overpass, (3) wind dispersal between the "
                f"~10:50 S2 pass and the ~12:00 TROPOMI pass. "
                f"Try --days-window 3 to find a day with both TROPOMI signal and S2 coverage."
            )

        print(f"  VERDICT: {verdict}")
        print()
        print(f"  INTERPRETATION:")
        print(f"  {interpretation}")
        print()

        # Groningen background: global avg CH4 is ~1900 ppb (2024)
        global_bg = 1922.0  # NOAA 2024 global mean
        if gron_stats["mean"] > global_bg - 10:
            print(f"  Context: Global mean CH4 ≈ {global_bg:.0f} ppb (NOAA 2024).")
            print(f"  Groningen reads {gron_stats['mean']:.1f} ppb — ",
                  end="")
            if gron_stats["mean"] > global_bg:
                print(f"{gron_stats['mean'] - global_bg:.1f} ppb above global mean.")
            else:
                print(f"within global background.")

    else:
        print("  WARNING: Not enough valid pixels for statistics.")
        print(f"  Groningen valid pixels: {len(gron_vals)}")
        print(f"  Reference valid pixels: {len(ref_vals)}")
        print("  Try --days-window 3 or check QA coverage over the Netherlands.")

    # ── 7. Save JSON ──
    output = {
        "run_date":         datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "s2_acquisition":   args.date,
        "tropomi_product":  best["name"],
        "tropomi_date":     best["date"],
        "tropomi_time_utc": best["time"],
        "temporal_delta_days": date_dist(best),
        "groningen_bbox":   GRONINGEN_BBOX,
        "reference_bbox":   REFERENCE_BBOX,
        "qa_threshold":     QA_MIN,
        "groningen_stats":  gron_stats,
        "reference_stats":  ref_stats,
        "enhancement_ppb":  round(gron_stats["mean"] - ref_stats["mean"], 2)
                            if gron_stats["mean"] and ref_stats["mean"] else None,
        "tropomi_sc_ratio": round(gron_stats["mean"] / ref_stats["mean"], 3)
                            if gron_stats["mean"] and ref_stats["mean"] else None,
        "s2_sc_ratio":      2.950,
        "verdict":          verdict if gron_stats["mean"] and ref_stats["mean"] else "INSUFFICIENT_DATA",
    }

    json_out = os.path.join(RESULTS_DIR, "groningen_colocation.json")
    with open(json_out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Stats saved → {json_out}")

    # ── 8. Map ──
    if not args.no_plot:
        print("\n▶  Generating CH4 map …")
        plot_ch4_map(nc_path, RESULTS_DIR, args.date)

    print("\n" + "=" * 70)
    print("  Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
