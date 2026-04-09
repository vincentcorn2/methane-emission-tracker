#!/usr/bin/env python3
"""
run_exhaustive_validation.py
────────────────────────────
Exhaustive CH4Net v2 validation across JRC emitter sites + matched blank controls.

For every site this script:
  1. Runs live_pipeline (skip if GeoTIFF already cached from a previous run)
  2. Computes near-plant event inventory at 5 / 10 / 20 km radii
  3. Computes the signal-to-control (S/C) probability ratio at the stack
  4. Sweeps three detection thresholds (0.18 / 0.30 / 0.50)
  5. Builds a concentric-ring probability profile (every 5 km out to 50 km)
  6. Estimates tile-wide false-positive density (events per 1000 km²)
  7. Computes near-plant precision (% of tile events within 10 km of plant)

Controls are purpose-selected rural areas in the same country / terrain class
as each emitter — no known industrial CH4 sources within 30 km.

Usage
-----
    cd /path/to/methane-api
    python run_exhaustive_validation.py              # fresh run
    python run_exhaustive_validation.py --no-cache   # force re-download all

Outputs
-------
    results_validation/<site>/               — GeoTIFF + events JSON per site
    results_validation/validation_full.log   — verbose per-step log
    results_validation/summary.json          — machine-readable results table
"""

import os, sys, json, math, subprocess, time, shutil, argparse, glob, textwrap
from datetime import datetime
from pathlib import Path

# ── Try loading rasterio / scipy (installed via requirements.txt) ──────────────
try:
    import numpy as np
    import rasterio
    from rasterio.transform import rowcol
    from scipy import ndimage
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("WARNING: rasterio / scipy not found — GeoTIFF analysis will be skipped.")
    print("         Run:  pip install rasterio scipy")

# ══════════════════════════════════════════════════════════════════════════════
# SITE DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════
# Each entry: key → dict with lat/lon of stack, fuel, country, terrain note,
# site_type ("emitter" | "control"), and optionally a cache_from path pointing
# to a GeoTIFF produced by a previous run (avoids re-downloading).

BASE_RESULTS = "results_survey"   # previous survey outputs to re-use as cache

SITES = {

    # ── EMITTER SITES: confirmed large-combustion / fugitive CH4 sources ──────

    "groningen": dict(
        lat=53.252, lon=6.682, fuel="gas_field", country="NL",
        note="Positive control — Groningen gas field, TROPOMI-confirmed 2.95× S/C",
        site_type="emitter",
        cache_from=f"{BASE_RESULTS}/groningen",
    ),
    "belchatow": dict(
        lat=51.263, lon=19.332, fuel="coal", country="PL",
        note="Europe's largest coal plant — expected strong signal",
        site_type="emitter",
        cache_from=f"{BASE_RESULTS}/belchatow",
    ),
    "maasvlakte": dict(
        lat=51.951, lon=4.004, fuel="coal", country="NL",
        note="Rotterdam port / coal terminal — cleanest result in survey (5/6 events near-plant)",
        site_type="emitter",
        cache_from=f"{BASE_RESULTS}/maasvlakte",
    ),
    "brindisi": dict(
        lat=40.563, lon=18.032, fuel="coal", country="IT",
        note="S. Italy Mediterranean coast — high event count, strong near-plant conf",
        site_type="emitter",
        cache_from=f"{BASE_RESULTS}/brindisi",
    ),
    "pocerady": dict(
        lat=50.430, lon=13.680, fuel="coal", country="CZ",
        note="Bohemian plains — conf 0.870 within 5 km",
        site_type="emitter",
        cache_from=f"{BASE_RESULTS}/pocerady",
    ),
    "dunkerque": dict(
        lat=51.033, lon=2.377, fuel="mixed", country="FR",
        note="N. France industrial coast — 13 total events, 1 within 5 km, conf 0.774",
        site_type="emitter",
        cache_from=f"{BASE_RESULTS}/dunkerque",
    ),
    "larobla": dict(
        lat=42.784, lon=-5.618, fuel="coal", country="ES",
        note="N. Spain mountain valley — high event density, conf 0.983 within 5 km",
        site_type="emitter",
        cache_from=f"{BASE_RESULTS}/larobla",
    ),

    # ── BORDERLINE / MISS sites — include to understand failure modes ─────────

    "irsching": dict(
        lat=48.767, lon=11.582, fuel="gas", country="DE",
        note="Bavaria — gas CCGT, no near-plant events in survey (expected negative?)",
        site_type="emitter_miss",
        cache_from=f"{BASE_RESULTS}/irsching",
    ),
    "cordemais": dict(
        lat=47.275, lon=-1.874, fuel="coal", country="FR",
        note="Loire estuary — 48 tile events but none within 10 km; query polygon mismatch",
        site_type="emitter_miss",
        cache_from=f"{BASE_RESULTS}/cordemais",
    ),
    "avedore": dict(
        lat=55.600, lon=12.450, fuel="gas", country="DK",
        note="Denmark coastal — 259 tile events, none within 10 km",
        site_type="emitter_miss",
        cache_from=f"{BASE_RESULTS}/avedore",
    ),
    "philippsburg": dict(
        lat=49.252, lon=8.459, fuel="gas", country="DE",
        note="Rhine plain — plant CLOSED 2019; correct prediction = no near-plant events",
        site_type="emitter_closed",
        cache_from=f"{BASE_RESULTS}/philippsburg",
    ),

    # ── BLANK CONTROLS: rural areas, no known industrial CH4 within 30 km ─────
    # Terrain-matched to emitter sites for apples-to-apples comparison.

    "ctrl_nl_flevoland": dict(
        lat=52.530, lon=5.450, fuel=None, country="NL",
        note="Flevoland polder — totally flat reclaimed agricultural land, no industry. "
             "Same S2 orbit / terrain class as Maasvlakte.",
        site_type="control",
    ),
    "ctrl_de_swabian_alb": dict(
        lat=48.400, lon=9.400, fuel=None, country="DE",
        note="Swabian Alb rural plateau — similar terrain to Irsching/Philippsburg, "
             "no emitters within 40 km.",
        site_type="control",
    ),
    "ctrl_pl_mazovia": dict(
        lat=52.200, lon=20.800, fuel=None, country="PL",
        note="Rural Mazovia east of Warsaw — flat Polish plains matching Belchatow terrain, "
             "no major industrial sites.",
        site_type="control",
    ),
    "ctrl_fr_berry": dict(
        lat=47.800, lon=0.500, fuel=None, country="FR",
        note="Maine rural plateau — farmland NW of Le Mans, >40 km from any city or industrial site. "
             "Relocated from original Berry centre (47.1N, 2.2E) which was within 15 km of "
             "Vierzon/Bourges industrial zone, explaining the 688-event FP spike.",
        site_type="control",
    ),
    "ctrl_it_basilicata": dict(
        lat=40.800, lon=15.800, fuel=None, country="IT",
        note="Rural Basilicata highlands — same Mediterranean climate as Brindisi, "
             "no industrial facilities.",
        site_type="control",
    ),
    "ctrl_es_castilla": dict(
        lat=41.000, lon=-2.500, fuel=None, country="ES",
        note="Castilla-La Mancha rural plateau — cereal/wine farmland, >60 km from any city. "
             "Relocated from original centre (41.6N, -4.2E) which was within 15 km of "
             "Valladolid (Renault + chemical plants), explaining the 5,656-event FP catastrophe.",
        site_type="control",
    ),
    "ctrl_cz_bohemia_east": dict(
        lat=50.200, lon=16.200, fuel=None, country="CZ",
        note="Eastern Bohemia agricultural plain — same country / terrain class as Pocerady, "
             "no power plants within 50 km.",
        site_type="control",
    ),
    "ctrl_dk_jutland": dict(
        lat=56.300, lon=9.200, fuel=None, country="DK",
        note="Rural Jutland — flat Danish farmland, same terrain class as Avedore, "
             "no industrial CH4 sources.",
        site_type="control",
    ),
}

# Pipeline parameters (matching the original survey exactly)
DATE_START   = "2024-06-01"
DATE_END     = "2024-08-31"
WEIGHTS      = "weights/ch4net_div8_retrained.pth"
MAX_PRODUCTS = 1
MAX_CLOUD    = 20
THRESHOLD    = 0.18          # primary detection threshold
BOX_DELTA    = 0.20          # ±0.20° bounding box (~22 km half-width)
RESULTS_DIR  = "results_validation"
LOG_FILE     = f"{RESULTS_DIR}/validation_full.log"

# Analysis parameters
RADII_KM     = [5, 10, 20]   # event count radii
THRESHOLDS   = [0.18, 0.30, 0.50]  # threshold sweep for saved GeoTIFFs
RING_STEP_KM = 5             # ring profile step
RING_MAX_KM  = 50            # ring profile outer edge
SC_CTRL_OFFSET_DEG = 0.18    # S/C control crop offset northward (~20 km)
CROP_HALF_PX = 100           # half-width of probability crop (200×200 px = 2 km)

# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════

os.makedirs(RESULTS_DIR, exist_ok=True)
_log_fh = open(LOG_FILE, "a", buffering=1)

def log(msg="", indent=0, also_print=True):
    ts  = datetime.now().strftime("%H:%M:%S")
    pfx = "  " * indent
    line = f"[{ts}] {pfx}{msg}"
    _log_fh.write(line + "\n")
    if also_print:
        print(line)

def section(title):
    bar = "═" * 72
    log()
    log(bar)
    log(f"  {title}")
    log(bar)

# ══════════════════════════════════════════════════════════════════════════════
# GEOMETRY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def make_polygon(lat, lon, delta=BOX_DELTA):
    """0.4°×0.4° WKT bounding box centred on lat/lon."""
    return (f"POLYGON(({lon-delta:.4f} {lat-delta:.4f}, "
            f"{lon+delta:.4f} {lat-delta:.4f}, "
            f"{lon+delta:.4f} {lat+delta:.4f}, "
            f"{lon-delta:.4f} {lat+delta:.4f}, "
            f"{lon-delta:.4f} {lat-delta:.4f}))")

def dist_km(lat1, lon1, lat2, lon2):
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return 6371.0 * 2 * math.asin(math.sqrt(a))

def lonlat_to_pixel(src, lon, lat):
    """
    Convert WGS-84 lon/lat to pixel (row, col) in any GeoTIFF CRS.
    Uses rasterio.warp.transform to reproject if the native CRS is not WGS-84
    (e.g. UTM), so lon/lat are never passed directly to a projected transform.
    """
    from rasterio.warp import transform as warp_transform
    from rasterio.crs import CRS
    wgs84 = CRS.from_epsg(4326)
    if src.crs.to_epsg() == 4326:
        x, y = lon, lat
    else:
        xs, ys = warp_transform(wgs84, src.crs, [lon], [lat])
        x, y = xs[0], ys[0]
    return rowcol(src.transform, x, y)


def tile_area_km2(geotiff_path):
    """Tile area in km² — handles both projected (UTM) and geographic (WGS-84) CRS."""
    if not HAS_RASTERIO:
        return None
    with rasterio.open(geotiff_path) as src:
        b = src.bounds
        if src.crs.is_projected:
            # Bounds are in metres
            width_m  = abs(b.right - b.left)
            height_m = abs(b.top   - b.bottom)
            return (width_m * height_m) / 1e6
        else:
            # Geographic (degrees) → rough km
            dlat = abs(b.top - b.bottom)
            dlon = abs(b.right - b.left)
            mid_lat = (b.top + b.bottom) / 2
            km_lat = dlat * 111.0
            km_lon = dlon * 111.0 * math.cos(math.radians(mid_lat))
            return km_lat * km_lon

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(site_name, lat, lon, out_dir):
    """Run live_pipeline for one site; return (returncode, elapsed_sec)."""
    polygon = make_polygon(lat, lon)
    cmd = [
        sys.executable, "-m", "scripts.live_pipeline",
        "--region",       polygon,
        "--start",        DATE_START,
        "--end",          DATE_END,
        "--weights",      WEIGHTS,
        "--max-products", str(MAX_PRODUCTS),
        "--max-cloud",    str(MAX_CLOUD),
        "--threshold",    str(THRESHOLD),
        "--output",       out_dir,
    ]
    run_log = os.path.join(out_dir, "run.log")
    t0 = time.time()
    with open(run_log, "w") as fh:
        result = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    return result.returncode, time.time() - t0

def find_geotiff(out_dir):
    """Return path to most recently written .tif in out_dir, or None."""
    tifs = glob.glob(os.path.join(out_dir, "*.tif"))
    return max(tifs, key=os.path.getmtime) if tifs else None

def load_events(out_dir):
    """Return list of event dicts from the events JSON, or []."""
    jsons = glob.glob(os.path.join(out_dir, "events_*.json"))
    if not jsons:
        return []
    with open(jsons[0]) as f:
        return json.load(f)

# ══════════════════════════════════════════════════════════════════════════════
# GEOTIFF ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def safe_crop(prob, row, col, half=CROP_HALF_PX):
    H, W = prob.shape
    r0, r1 = row - half, row + half
    c0, c1 = col - half, col + half
    if r0 < 0 or r1 > H or c0 < 0 or c1 > W:
        return None
    return prob[r0:r1, c0:c1]

def sc_ratio_from_geotiff(geotiff_path, plat, plon,
                          ctrl_offset_deg=SC_CTRL_OFFSET_DEG):
    """
    S/C ratio: mean probability in a 200×200 px crop at the plant vs.
    a matched control crop offset ~20 km away in one of four cardinal directions.
    Tries N → S → E → W and uses the first direction whose crop fits inside the tile.
    Returns dict with site_mean, ctrl_mean, sc_ratio, ctrl_direction, or None values.
    """
    if not HAS_RASTERIO:
        return dict(site_mean=None, ctrl_mean=None, sc_ratio=None, error="no_rasterio")

    # Cardinal offsets: (lat_delta, lon_delta, label)
    offsets = [
        ( ctrl_offset_deg,  0.0,              "N"),
        (-ctrl_offset_deg,  0.0,              "S"),
        ( 0.0,              ctrl_offset_deg,  "E"),
        ( 0.0,             -ctrl_offset_deg,  "W"),
    ]

    try:
        with rasterio.open(geotiff_path) as src:
            prob = src.read(1).astype(np.float32)
            s_row, s_col = lonlat_to_pixel(src, plon, plat)
    except Exception as e:
        return dict(site_mean=None, ctrl_mean=None, sc_ratio=None, error=str(e))

    site_crop = safe_crop(prob, s_row, s_col)
    if site_crop is None:
        return dict(site_mean=None, ctrl_mean=None, sc_ratio=None,
                    error="plant_coord_out_of_bounds")

    for dlat, dlon, direction in offsets:
        try:
            with rasterio.open(geotiff_path) as src:
                c_row, c_col = lonlat_to_pixel(src, plon + dlon, plat + dlat)
        except Exception:
            continue
        ctrl_crop = safe_crop(prob, c_row, c_col)
        if ctrl_crop is None:
            continue
        site_mean = float(site_crop.mean())
        ctrl_mean = float(ctrl_crop.mean())
        sc = site_mean / ctrl_mean if ctrl_mean > 1e-9 else float("inf")
        return dict(site_mean=round(site_mean, 6),
                    ctrl_mean=round(ctrl_mean, 6),
                    sc_ratio=round(sc, 4),
                    ctrl_direction=direction)

    return dict(site_mean=None, ctrl_mean=None, sc_ratio=None,
                error="all_directions_out_of_bounds")

def threshold_sweep(geotiff_path, thresholds=THRESHOLDS, min_pixels=115):
    """
    For each threshold, count events (connected components ≥ min_pixels)
    in the full GeoTIFF probability map.
    Returns dict: threshold → {events, total_pixels, pct}.
    """
    if not HAS_RASTERIO:
        return {}
    results = {}
    try:
        with rasterio.open(geotiff_path) as src:
            prob = src.read(1).astype(np.float32)
        for t in thresholds:
            binary = prob >= t
            total_px = int(binary.sum())
            labeled, n = ndimage.label(binary)
            if n > 0:
                sizes = np.bincount(labeled.ravel())[1:]
                events = int((sizes >= min_pixels).sum())
            else:
                events = 0
            results[t] = dict(events=events,
                              total_pixels=total_px,
                              pct=round(100.0 * total_px / binary.size, 4))
    except Exception as e:
        results["error"] = str(e)
    return results

def ring_profile(geotiff_path, plat, plon,
                 step_km=RING_STEP_KM, max_km=RING_MAX_KM):
    """
    Mean probability in concentric rings around the plant stack.

    Works entirely in the GeoTIFF's native projected coordinate system
    (UTM, ~10 m/pixel) — no pyproj required.  Distance is Euclidean in
    UTM metres converted to km, which is accurate to <1% within a single tile.

    Returns list of dicts: {inner_km, outer_km, mean_prob, n_pixels}.
    """
    if not HAS_RASTERIO:
        return []
    rings = []
    try:
        with rasterio.open(geotiff_path) as src:
            prob = src.read(1).astype(np.float32)
            H, W = prob.shape

            # Plant location in pixel coordinates (CRS-aware)
            p_row, p_col = lonlat_to_pixel(src, plon, plat)

            # Pixel size in metres (positive values)
            px_m = abs(src.transform.a)   # x pixel size
            py_m = abs(src.transform.e)   # y pixel size (transform.e is negative)

            # Row / col index grids
            col_idx, row_idx = np.meshgrid(np.arange(W, dtype=np.float32),
                                           np.arange(H, dtype=np.float32))

            # Euclidean distance in km from plant pixel
            dist_grid_km = np.sqrt(
                ((col_idx - p_col) * px_m) ** 2 +
                ((row_idx - p_row) * py_m) ** 2
            ) / 1000.0

            for inner in range(0, max_km, step_km):
                outer = inner + step_km
                mask = (dist_grid_km >= inner) & (dist_grid_km < outer)
                n = int(mask.sum())
                if n == 0:
                    rings.append(dict(inner_km=inner, outer_km=outer,
                                      mean_prob=None, n_pixels=0))
                else:
                    rings.append(dict(inner_km=inner, outer_km=outer,
                                      mean_prob=round(float(prob[mask].mean()), 6),
                                      n_pixels=n))
    except Exception as e:
        rings.append(dict(error=str(e)))
    return rings

def near_plant_precision(events, plat, plon, radius_km=10):
    """
    Precision = fraction of tile-wide events that fall within radius_km of plant.
    Also returns absolute counts.
    """
    if not events:
        return dict(total=0, within=0, precision=None)
    within = sum(1 for e in events
                 if dist_km(plat, plon, e["latitude"], e["longitude"]) <= radius_km)
    return dict(total=len(events), within=within,
                precision=round(within / len(events), 4) if events else None)

# ══════════════════════════════════════════════════════════════════════════════
# PER-SITE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyse_site(site_name, meta, geotiff_path, events):
    """
    Run all analysis on a single site's GeoTIFF + event list.
    Returns a results dict written both to the log and to summary.json.
    """
    plat, plon = meta["lat"], meta["lon"]

    # ── 1. Near-plant event inventory ─────────────────────────────────────────
    log(f"Near-plant event inventory:", indent=2)
    radii_results = {}
    for r in RADII_KM:
        within = [(dist_km(plat, plon, e["latitude"], e["longitude"]), e)
                  for e in events
                  if dist_km(plat, plon, e["latitude"], e["longitude"]) <= r]
        best_conf = max((e["model_confidence"] for _, e in within), default=None)
        radii_results[r] = dict(count=len(within), best_conf=best_conf)
        conf_str = f"{best_conf:.3f}" if best_conf is not None else "—"
        log(f"  ≤{r:2d} km : {len(within):4d} events   best conf = {conf_str}", indent=3)

    # ── 2. Signal/Control ratio ────────────────────────────────────────────────
    sc = sc_ratio_from_geotiff(geotiff_path, plat, plon)
    direction = sc.get("ctrl_direction", "?")
    log(f"S/C probability ratio (plant crop / control crop ~20 km {direction}):", indent=2)
    if sc.get("sc_ratio") is not None:
        sc_str = f"{sc['sc_ratio']:.3f}"
        label = ("✓✓ STRONG" if sc["sc_ratio"] >= 1.5
                 else "✓  elevated" if sc["sc_ratio"] >= 1.1
                 else "~  neutral" if sc["sc_ratio"] >= 0.9
                 else "✗  suppressed")
        log(f"  site={sc['site_mean']:.6f}  ctrl={sc['ctrl_mean']:.6f}  "
            f"ratio={sc_str}  {label}", indent=3)
    else:
        log(f"  S/C unavailable: {sc.get('error','?')}", indent=3)

    # ── 3. Threshold sweep ────────────────────────────────────────────────────
    sweep = threshold_sweep(geotiff_path)
    log(f"Threshold sweep (tile-wide events at each threshold):", indent=2)
    log(f"  {'Threshold':>10}  {'Events':>8}  {'Pixels above':>13}  {'% tile':>7}", indent=3)
    for t in THRESHOLDS:
        if t in sweep:
            s = sweep[t]
            log(f"  {t:>10.2f}  {s['events']:>8d}  {s['total_pixels']:>13,}  "
                f"{s['pct']:>7.3f}%", indent=3)

    # ── 4. Ring probability profile ───────────────────────────────────────────
    rings = ring_profile(geotiff_path, plat, plon)
    log(f"Concentric-ring mean probability (5 km bands, out to 50 km):", indent=2)
    log(f"  {'Ring':>14}  {'Mean prob':>10}  {'Pixels':>9}", indent=3)
    for rng in rings:
        if "error" in rng:
            log(f"  ring error: {rng['error']}", indent=3)
            break
        prob_str = f"{rng['mean_prob']:.6f}" if rng["mean_prob"] is not None else "  (empty)"
        log(f"  {rng['inner_km']:>3d}–{rng['outer_km']:>3d} km  "
            f"{prob_str:>10}  {rng['n_pixels']:>9,}", indent=3)

    # ── 5. Tile-wide false-positive density ───────────────────────────────────
    area = tile_area_km2(geotiff_path)
    density = (len(events) / area * 1000) if (area and events) else None
    log(f"Tile statistics:", indent=2)
    log(f"  Total events in tile:  {len(events)}", indent=3)
    if area:
        log(f"  Tile area (approx):    {area:,.0f} km²", indent=3)
    if density is not None:
        log(f"  Event density:         {density:.2f} events per 1000 km²", indent=3)

    # ── 6. Near-plant precision ───────────────────────────────────────────────
    prec = near_plant_precision(events, plat, plon, radius_km=10)
    if prec["total"] > 0:
        prec_str = f"{prec['precision']*100:.1f}%" if prec["precision"] is not None else "—"
        log(f"  Near-plant precision (events within 10 km / total):  "
            f"{prec['within']}/{prec['total']} = {prec_str}", indent=3)

    # ── Build result dict ─────────────────────────────────────────────────────
    return dict(
        site=site_name,
        site_type=meta["site_type"],
        lat=plat, lon=plon,
        fuel=meta.get("fuel"),
        country=meta["country"],
        note=meta["note"],
        geotiff=os.path.basename(geotiff_path),
        total_events=len(events),
        radii=radii_results,
        sc=sc,
        threshold_sweep=sweep,
        ring_profile=rings,
        tile_area_km2=round(area, 1) if area else None,
        event_density_per1000km2=round(density, 2) if density else None,
        precision_10km=prec,
    )

# ══════════════════════════════════════════════════════════════════════════════
# CACHE LOGIC
# ══════════════════════════════════════════════════════════════════════════════

def copy_cache(src_dir, dst_dir):
    """
    Copy GeoTIFF + events JSON from a previous survey run into the
    validation output directory so we don't re-download.
    Returns True if cache was usable.
    """
    tifs = glob.glob(os.path.join(src_dir, "*.tif"))
    jsons = glob.glob(os.path.join(src_dir, "events_*.json"))
    if not tifs:
        return False
    os.makedirs(dst_dir, exist_ok=True)
    for src_f in tifs + jsons:
        dst_f = os.path.join(dst_dir, os.path.basename(src_f))
        if not os.path.exists(dst_f):
            shutil.copy2(src_f, dst_f)
    return True

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_summary_table(all_results):
    """Print a compact ranked table to log and stdout."""
    section("FINAL SUMMARY TABLE — ranked by S/C ratio")

    # Separate types
    emitters = [r for r in all_results if r["site_type"] == "emitter"]
    misses   = [r for r in all_results if r["site_type"] == "emitter_miss"]
    closed   = [r for r in all_results if r["site_type"] == "emitter_closed"]
    controls = [r for r in all_results if r["site_type"] == "control"]

    def sc_val(r):
        return r["sc"].get("sc_ratio") or -999

    def fmt_row(r):
        sc  = r["sc"].get("sc_ratio")
        sc_str = f"{sc:.3f}" if sc is not None else "   —  "
        ev5  = r["radii"].get(5, {}).get("count", 0)
        ev10 = r["radii"].get(10, {}).get("count", 0)
        bc5  = r["radii"].get(5, {}).get("best_conf")
        bc10 = r["radii"].get(10, {}).get("best_conf")
        bc_str = f"{max(bc5 or 0, bc10 or 0):.3f}" if (bc5 or bc10) else "  —  "
        tot   = r["total_events"]
        prec  = r["precision_10km"].get("precision")
        p_str = f"{prec*100:.0f}%" if prec is not None else "—"
        if sc and sc >= 1.5:         verdict = "✓✓ STRONG"
        elif sc and sc >= 1.1:       verdict = "✓  elevated"
        elif sc and sc >= 0.9:       verdict = "~  neutral"
        elif ev10 > 0:               verdict = "~  events near"
        else:                        verdict = "✗  nothing"
        return (f"  {r['site']:<22} {r['country']:<4} {(r['fuel'] or 'ctrl'):<9} "
                f"{sc_str:>7}  {ev5:>5}  {ev10:>6}  {bc_str:>7}  "
                f"{tot:>7}  {p_str:>6}  {verdict}")

    hdr = (f"  {'Site':<22} {'Ctry':<4} {'Fuel':<9} "
           f"{'S/C':>7}  {'ev<5':>5}  {'ev<10':>6}  {'bestC':>7}  "
           f"{'total':>7}  {'prec':>6}  Verdict")
    bar = "─" * len(hdr)

    for label, group in [("EMITTERS (active)", sorted(emitters, key=sc_val, reverse=True)),
                          ("EMITTERS (misses)", misses),
                          ("EMITTERS (closed)", closed),
                          ("BLANK CONTROLS",    sorted(controls, key=sc_val, reverse=True))]:
        if not group:
            continue
        log(f"\n  {label}")
        log(hdr); log(bar)
        for r in group:
            log(fmt_row(r))

    # Control baseline stats
    ctrl_sc = [r["sc"].get("sc_ratio") for r in controls
               if r["sc"].get("sc_ratio") is not None]
    ctrl_ev = [r["total_events"] for r in controls]
    if ctrl_sc:
        log(f"\n  Control S/C:  mean={sum(ctrl_sc)/len(ctrl_sc):.3f}  "
            f"max={max(ctrl_sc):.3f}  min={min(ctrl_sc):.3f}")
    if ctrl_ev:
        log(f"  Control events/tile:  mean={sum(ctrl_ev)/len(ctrl_ev):.1f}  "
            f"max={max(ctrl_ev)}  min={min(ctrl_ev)}")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-cache", action="store_true",
                        help="Ignore cached GeoTIFFs and re-run the pipeline for every site.")
    parser.add_argument("--sites", nargs="+",
                        help="Run only specific sites (space-separated keys from SITES dict).")
    args = parser.parse_args()

    section(f"CH4Net v2 — Exhaustive Validation  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log(f"Sites:       {len(SITES)} total  "
        f"({sum(1 for s in SITES.values() if s['site_type']=='emitter')} emitters, "
        f"{sum(1 for s in SITES.values() if s['site_type']=='control')} controls, "
        f"{sum(1 for s in SITES.values() if 'miss' in s['site_type'] or 'closed' in s['site_type'])} borderline)")
    log(f"Date range:  {DATE_START} → {DATE_END}")
    log(f"Threshold:   {THRESHOLD}  |  Max products: {MAX_PRODUCTS}  |  Max cloud: {MAX_CLOUD}%")
    log(f"Weights:     {WEIGHTS}")
    log(f"Output dir:  {RESULTS_DIR}/")
    log(f"Log file:    {LOG_FILE}")
    log(f"Rasterio:    {'available' if HAS_RASTERIO else 'NOT INSTALLED — GeoTIFF analysis will be skipped'}")

    sites_to_run = list(SITES.items())
    if args.sites:
        sites_to_run = [(k, v) for k, v in sites_to_run if k in args.sites]
        log(f"Running subset: {[k for k,_ in sites_to_run]}")

    # Load any prior summary so we can resume gracefully
    summary_path = os.path.join(RESULTS_DIR, "summary.json")
    all_results = []
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            all_results = json.load(f)
    done_sites = {r["site"] for r in all_results}

    # ── Per-site loop ──────────────────────────────────────────────────────────
    for site_name, meta in sites_to_run:
        if site_name in done_sites and not args.no_cache:
            log(f"\n[SKIP] {site_name} — already in summary.json (use --no-cache to rerun)")
            continue

        section(f"[{site_name.upper()}]  {meta['site_type'].upper()}  —  {meta['note']}")
        log(f"Coords: ({meta['lat']:.4f}°N, {meta['lon']:.4f}°E)  "
            f"Country: {meta['country']}  Fuel: {meta.get('fuel','—')}")

        out_dir = os.path.join(RESULTS_DIR, site_name)
        os.makedirs(out_dir, exist_ok=True)

        # ── Try cache first ────────────────────────────────────────────────────
        geotiff = find_geotiff(out_dir)
        if geotiff and not args.no_cache:
            log(f"Cache hit — using existing GeoTIFF: {os.path.basename(geotiff)}", indent=1)
        elif meta.get("cache_from") and not args.no_cache:
            if copy_cache(meta["cache_from"], out_dir):
                geotiff = find_geotiff(out_dir)
                log(f"Cache copied from {meta['cache_from']}: {os.path.basename(geotiff)}", indent=1)
            else:
                log(f"No cache available at {meta['cache_from']} — running pipeline", indent=1)

        # ── Run pipeline if still no GeoTIFF ──────────────────────────────────
        if not find_geotiff(out_dir):
            log(f"Running live_pipeline …", indent=1)
            rc, elapsed = run_pipeline(site_name, meta["lat"], meta["lon"], out_dir)
            geotiff = find_geotiff(out_dir)
            if rc != 0 or geotiff is None:
                log(f"Pipeline FAILED (rc={rc}, elapsed={elapsed/60:.1f} min) — see {out_dir}/run.log",
                    indent=1)
                result = dict(site=site_name, site_type=meta["site_type"],
                              lat=meta["lat"], lon=meta["lon"],
                              fuel=meta.get("fuel"), country=meta["country"],
                              note=meta["note"], status="PIPELINE_FAILED",
                              total_events=0, radii={}, sc={},
                              threshold_sweep={}, ring_profile=[],
                              tile_area_km2=None, event_density_per1000km2=None,
                              precision_10km=dict(total=0, within=0, precision=None))
                all_results.append(result)
                with open(summary_path, "w") as f:
                    json.dump(all_results, f, indent=2, default=str)
                continue
            log(f"Pipeline complete ({elapsed/60:.1f} min)  →  {os.path.basename(geotiff)}",
                indent=1)
        else:
            geotiff = find_geotiff(out_dir)

        # ── Load events ────────────────────────────────────────────────────────
        events = load_events(out_dir)
        log(f"GeoTIFF: {os.path.basename(geotiff)}", indent=1)
        log(f"Total events in tile: {len(events)}", indent=1)

        if events == 0 or (not events and meta["site_type"] == "control"):
            log("No events detected (expected for blank control)", indent=2)

        # ── Run all analyses ───────────────────────────────────────────────────
        result = analyse_site(site_name, meta, geotiff, events)
        all_results.append(result)

        # Save after each site (safe to interrupt)
        done_sites.add(site_name)
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        log(f"Result saved to {summary_path}", indent=1)

    # ── Final summary ──────────────────────────────────────────────────────────
    print_summary_table(all_results)

    log()
    log(f"Validation complete at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Full log:  {LOG_FILE}")
    log(f"Results:   {summary_path}")
    _log_fh.close()


if __name__ == "__main__":
    main()
