#!/usr/bin/env python3
"""
run_expanded_sites.py
─────────────────────
Expanded JRC sweep — 20 new emitter sites + 4 terrain-matched controls.

Prioritised by installed capacity so the highest-value plants run first.
All sites are confirmed active in summer 2024 (coordinates from GEO /
latitude.to / Wikipedia cross-checked).

Saves to  results_expanded/<site>/  so it doesn't interfere with
results_validation/ from the first run.

Usage
-----
    cd /path/to/methane-api
    python run_expanded_sites.py           # run everything
    python run_expanded_sites.py --fast    # skip ring profile (~2 min saved per site)

At ~10 min/site you get ≈ 15-18 sites in 3 hours.
Results are cached: safe to interrupt and resume.
"""

import os, sys, json, math, subprocess, time, glob, shutil, argparse
from datetime import datetime
from pathlib import Path

try:
    import numpy as np
    import rasterio
    from rasterio.transform import rowcol
    from rasterio.warp import transform as warp_transform
    from rasterio.crs import CRS
    from scipy import ndimage
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

# ══════════════════════════════════════════════════════════════════════════════
# NEW SITES — ordered by installed capacity (largest first)
# Coordinates from GEO / latitude.to / Wikipedia, cross-checked.
# ══════════════════════════════════════════════════════════════════════════════

SITES = {

    # ── GERMANY — brown coal (lignite), still active summer 2024 ──────────────

    "neurath": dict(
        lat=51.038, lon=6.612, fuel="coal", country="DE", mw=4400,
        note="RWE Neurath — Europe's largest active brown coal plant by capacity. "
             "BoA 2 & 3 units (2×1,100 MW) operating; older units decommissioned Mar 2024.",
        site_type="emitter",
    ),
    "janschwalde": dict(
        lat=51.834, lon=14.456, fuel="coal", country="DE", mw=3000,
        note="Jänschwalde — Lausitz lignite, 3,000 MW. Units A-D active summer 2024; "
             "E & F retired Mar 2024. Largest in eastern Germany.",
        site_type="emitter",
    ),
    "boxberg": dict(
        lat=51.416, lon=14.565, fuel="coal", country="DE", mw=2575,
        note="Boxberg — Lusatian lignite, 2,575 MW. R unit (900 MW BoA) still active. "
             "Same Lausitz cluster as Jänschwalde.",
        site_type="emitter",
    ),
    "schwarze_pumpe": dict(
        lat=51.538, lon=14.354, fuel="coal", country="DE", mw=1600,
        note="Schwarze Pumpe (LEAG) — 1,600 MW, Lusatia. Modernised units, "
             "scheduled close 2038. Between Jänschwalde and Boxberg tiles.",
        site_type="emitter",
    ),
    "lippendorf": dict(
        lat=51.185, lon=12.378, fuel="coal", country="DE", mw=1800,
        note="Lippendorf — 1,800 MW, south of Leipzig. Two BoA units active, "
             "scheduled close 2035. Flat Saxon agricultural terrain.",
        site_type="emitter",
    ),
    "datteln4": dict(
        lat=51.633, lon=7.341, fuel="coal", country="DE", mw=1100,
        note="Datteln 4 (Uniper) — Europe's newest coal plant (commissioned 2020), "
             "1,100 MW hard coal. Dense Ruhr Valley industrial setting.",
        site_type="emitter",
    ),
    "mannheim": dict(
        lat=49.479, lon=8.508, fuel="coal", country="DE", mw=1975,
        note="Großkraftwerk Mannheim — 1,975 MW hard coal, Rhine riverside. "
             "Units 7 & 8 active; one of Germany's last large hard-coal plants.",
        site_type="emitter",
    ),
    "weisweiler": dict(
        lat=50.837, lon=6.322, fuel="coal", country="DE", mw=1800,
        note="Weisweiler (RWE) — 1,800 MW lignite near Aachen. "
             "2 of 4 units still active summer 2024.",
        site_type="emitter",
    ),

    # ── POLAND — hard coal / mixed, all active summer 2024 ───────────────────

    "turow": dict(
        lat=51.003, lon=14.912, fuel="coal", country="PL", mw=2000,
        note="Turów — 2,000 MW lignite, Sudeten foothills near Czech/German border. "
             "In original notebook list but never run. High-relief terrain.",
        site_type="emitter",
    ),
    "kozienice": dict(
        lat=51.667, lon=21.467, fuel="coal", country="PL", mw=2840,
        note="Kozienice — 2,840 MW hard coal, Masovian flatlands south of Warsaw. "
             "Unit 11 (1,075 MW) newest; some older units decommissioned.",
        site_type="emitter",
    ),
    "opole": dict(
        lat=50.751, lon=17.883, fuel="coal", country="PL", mw=1800,
        note="Opole — 1,800 MW (expanded 2018–19), Upper Silesia. "
             "One of Poland's most modern coal plants.",
        site_type="emitter",
    ),
    "rybnik": dict(
        lat=50.135, lon=18.522, fuel="coal", country="PL", mw=1720,
        note="Rybnik — 1,720 MW hard coal, Upper Silesian industrial belt. "
             "Dense industrial terrain; multiple nearby sources.",
        site_type="emitter",
    ),

    # ── ROMANIA — lignite, major emitters ─────────────────────────────────────

    "turceni": dict(
        lat=44.670, lon=23.408, fuel="coal", country="RO", mw=2640,
        note="Turceni — 2,640 MW lignite, Oltenia coal basin. "
             "Open-cast mine adjacent; known large CH4 / CO2 emitter.",
        site_type="emitter",
    ),
    "rovinari": dict(
        lat=44.907, lon=23.138, fuel="coal", country="RO", mw=990,
        note="Rovinari — 990 MW active lignite (down from 1,320 MW), Gorj county. "
             "Open-cast Rovinari mine immediately adjacent — strong fugitive CH4 source.",
        site_type="emitter",
    ),

    # ── GREECE — lignite, Mediterranean terrain ────────────────────────────────

    "agios_dimitrios": dict(
        lat=40.394, lon=21.925, fuel="coal", country="GR", mw=1595,
        note="Agios Dimitrios (PPC) — 1,595 MW lignite, Western Macedonia. "
             "Greece's largest power plant; open-cast Ptolemaida mine adjacent.",
        site_type="emitter",
    ),

    # ── BULGARIA — lignite ────────────────────────────────────────────────────

    "maritsa_east2": dict(
        lat=42.255, lon=26.135, fuel="coal", country="BG", mw=1450,
        note="Maritsa East 2 (AES) — 1,450 MW lignite, Stara Zagora plain. "
             "Maritsa Iztok mining basin adjacent. EU's most carbon-intensive grid.",
        site_type="emitter",
    ),

    # ── CZECH REPUBLIC — additional lignite near Pocerady tile ───────────────

    "ledvice": dict(
        lat=50.577, lon=13.779, fuel="coal", country="CZ", mw=660,
        note="Ledvice (ČEZ) — 660 MW (new BoA unit 4), North Bohemian basin. "
             "Shares terrain class with Počerady; different tile to the east.",
        site_type="emitter",
    ),
    "tusimice": dict(
        lat=50.380, lon=13.340, fuel="coal", country="CZ", mw=800,
        note="Tušimice II (ČEZ) — 800 MW lignite, Ústí nad Labem region. "
             "North Bohemian lignite basin, same cluster as Počerady and Ledvice.",
        site_type="emitter",
    ),

    # ── SPAIN — coal, different terrain from La Robla ─────────────────────────

    "compostilla": dict(
        lat=42.613, lon=-6.565, fuel="coal", country="ES", mw=1570,
        note="Compostilla II (Endesa) — 1,570 MW hard coal, Bierzo basin León. "
             "~20 km from La Robla tile. Mountain valley, same NW Spain terrain. "
             "Active until late 2020; check if visible in Aug 2024 data.",
        site_type="emitter",
    ),

    # ── TERRAIN-MATCHED BLANK CONTROLS for new regions ────────────────────────

    "ctrl_de_lausitz_rural": dict(
        lat=52.100, lon=13.500, fuel=None, country="DE", mw=None,
        note="Rural Brandenburg south of Berlin — flat North European Plain, "
             "same terrain class as Jänschwalde/Boxberg/Schwarze Pumpe, "
             "no industrial CH4 sources within 50 km.",
        site_type="control",
    ),
    "ctrl_ro_wallachia": dict(
        lat=44.300, lon=25.500, fuel=None, country="RO", mw=None,
        note="Rural Wallachian plain east of Bucharest — agricultural flatland, "
             "same climate zone as Turceni/Rovinari, no power plants within 60 km.",
        site_type="control",
    ),
    "ctrl_gr_thessaly": dict(
        lat=39.700, lon=22.300, fuel=None, country="GR", mw=None,
        note="Thessaly plain (Larissa area) — flat Greek agricultural land, "
             "same Mediterranean climate as Agios Dimitrios, no lignite mining.",
        site_type="control",
    ),
    "ctrl_bg_danube": dict(
        lat=43.600, lon=25.000, fuel=None, country="BG", mw=None,
        note="Danubian plain, north-central Bulgaria — flat cereal agriculture, "
             "same country as Maritsa East 2 but far from any industrial site.",
        site_type="control",
    ),

    # ── Groningen — added for multidate validation (strongest detection) ───────
    "groningen": dict(
        lat=53.252, lon=6.682, fuel="gas", country="NL", mw=None,
        note="Groningen gas field — largest European gas field, production halted Jun 2024. "
             "Strongest CH4Net v2 detection (S/C=4.191 on 2024-06-28). Multidate run to "
             "confirm spatial pattern is robust across summer 2024.",
        site_type="emitter",
    ),
}

# ── Parameters ────────────────────────────────────────────────────────────────
DATE_START   = "2024-05-01"
DATE_END     = "2024-09-30"
WEIGHTS      = "weights/ch4net_div8_retrained.pth"
MAX_PRODUCTS = 3
MAX_CLOUD    = 20
THRESHOLD    = 0.18
BOX_DELTA    = 0.20
RESULTS_DIR  = "results_expanded"
LOG_FILE     = f"{RESULTS_DIR}/expanded_run.log"

RADII_KM     = [5, 10, 20]
THRESHOLDS   = [0.18, 0.30, 0.50]
RING_STEP_KM = 5
RING_MAX_KM  = 50
SC_CTRL_OFFSET_DEG = 0.18
CROP_HALF_PX = 100

# ══════════════════════════════════════════════════════════════════════════════
os.makedirs(RESULTS_DIR, exist_ok=True)
_log_fh = open(LOG_FILE, "a", buffering=1)

def log(msg="", indent=0):
    ts   = datetime.now().strftime("%H:%M:%S")
    pfx  = "  " * indent
    line = f"[{ts}] {pfx}{msg}"
    _log_fh.write(line + "\n")
    print(line)

def section(title):
    bar = "═" * 72
    log(); log(bar); log(f"  {title}"); log(bar)

def make_polygon(lat, lon, delta=BOX_DELTA):
    return (f"POLYGON(({lon-delta:.4f} {lat-delta:.4f}, "
            f"{lon+delta:.4f} {lat-delta:.4f}, "
            f"{lon+delta:.4f} {lat+delta:.4f}, "
            f"{lon-delta:.4f} {lat+delta:.4f}, "
            f"{lon-delta:.4f} {lat-delta:.4f}))")

def dist_km(lat1, lon1, lat2, lon2):
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon/2)**2)
    return 6371.0 * 2 * math.asin(math.sqrt(a))

def lonlat_to_pixel(src, lon, lat):
    wgs84 = CRS.from_epsg(4326)
    if src.crs.to_epsg() == 4326:
        x, y = lon, lat
    else:
        xs, ys = warp_transform(wgs84, src.crs, [lon], [lat])
        x, y = xs[0], ys[0]
    return rowcol(src.transform, x, y)

def safe_crop(prob, row, col, half=CROP_HALF_PX):
    H, W = prob.shape
    r0, r1 = row - half, row + half
    c0, c1 = col - half, col + half
    if r0 < 0 or r1 > H or c0 < 0 or c1 > W:
        return None
    return prob[r0:r1, c0:c1]

def tile_area_km2(geotiff_path):
    if not HAS_RASTERIO:
        return None
    with rasterio.open(geotiff_path) as src:
        b = src.bounds
        if src.crs.is_projected:
            return abs(b.right - b.left) * abs(b.top - b.bottom) / 1e6
        else:
            dlat = abs(b.top - b.bottom)
            dlon = abs(b.right - b.left)
            mid  = (b.top + b.bottom) / 2
            return dlat * 111.0 * dlon * 111.0 * math.cos(math.radians(mid))

def run_pipeline(site_name, lat, lon, out_dir):
    polygon = make_polygon(lat, lon)
    cmd = [sys.executable, "-m", "scripts.live_pipeline",
           "--region",       polygon,
           "--start",        DATE_START,
           "--end",          DATE_END,
           "--weights",      WEIGHTS,
           "--max-products", str(MAX_PRODUCTS),
           "--max-cloud",    str(MAX_CLOUD),
           "--threshold",    str(THRESHOLD),
           "--output",       out_dir]
    t0 = time.time()
    with open(os.path.join(out_dir, "run.log"), "w") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
    return rc, time.time() - t0

def find_geotiff(d):
    tifs = glob.glob(os.path.join(d, "*.tif"))
    return max(tifs, key=os.path.getmtime) if tifs else None

def load_events(d):
    jsons = glob.glob(os.path.join(d, "events_*.json"))
    return json.load(open(jsons[0])) if jsons else []

# ── Analysis ──────────────────────────────────────────────────────────────────

def sc_ratio(geotiff_path, plat, plon):
    if not HAS_RASTERIO:
        return {}
    offsets = [(SC_CTRL_OFFSET_DEG, 0.0, "N"), (-SC_CTRL_OFFSET_DEG, 0.0, "S"),
               (0.0, SC_CTRL_OFFSET_DEG, "E"), (0.0, -SC_CTRL_OFFSET_DEG, "W")]
    try:
        with rasterio.open(geotiff_path) as src:
            prob = src.read(1).astype(np.float32)
            s_row, s_col = lonlat_to_pixel(src, plon, plat)
    except Exception as e:
        return {"error": str(e)}
    site_crop = safe_crop(prob, s_row, s_col)
    if site_crop is None:
        return {"error": "plant_out_of_bounds"}
    for dlat, dlon, direction in offsets:
        try:
            with rasterio.open(geotiff_path) as src:
                c_row, c_col = lonlat_to_pixel(src, plon + dlon, plat + dlat)
        except Exception:
            continue
        ctrl_crop = safe_crop(prob, c_row, c_col)
        if ctrl_crop is None:
            continue
        sm = float(site_crop.mean())
        cm = float(ctrl_crop.mean())
        sc = sm / cm if cm > 1e-9 else float("inf")
        return dict(site_mean=round(sm, 6), ctrl_mean=round(cm, 6),
                    sc_ratio=round(sc, 4), ctrl_direction=direction)
    return {"error": "all_directions_oob"}

def threshold_sweep(geotiff_path):
    if not HAS_RASTERIO:
        return {}
    out = {}
    try:
        with rasterio.open(geotiff_path) as src:
            prob = src.read(1).astype(np.float32)
        for t in THRESHOLDS:
            binary = prob >= t
            n_px   = int(binary.sum())
            labeled, n = ndimage.label(binary)
            events = int((np.bincount(labeled.ravel())[1:] >= 115).sum()) if n > 0 else 0
            out[t] = dict(events=events, total_pixels=n_px,
                          pct=round(100.0 * n_px / binary.size, 4))
    except Exception as e:
        out["error"] = str(e)
    return out

def ring_profile(geotiff_path, plat, plon, fast=False):
    """Concentric-ring mean probability. Skip if --fast flag set."""
    if fast or not HAS_RASTERIO:
        return []
    rings = []
    try:
        with rasterio.open(geotiff_path) as src:
            prob = src.read(1).astype(np.float32)
            H, W = prob.shape
            p_row, p_col = lonlat_to_pixel(src, plon, plat)
            px_m = abs(src.transform.a)
            py_m = abs(src.transform.e)
            col_idx, row_idx = np.meshgrid(np.arange(W, dtype=np.float32),
                                           np.arange(H, dtype=np.float32))
            dist_km_grid = np.sqrt(((col_idx - p_col) * px_m)**2 +
                                   ((row_idx - p_row) * py_m)**2) / 1000.0
        for inner in range(0, RING_MAX_KM, RING_STEP_KM):
            outer = inner + RING_STEP_KM
            mask  = (dist_km_grid >= inner) & (dist_km_grid < outer)
            n     = int(mask.sum())
            rings.append(dict(inner_km=inner, outer_km=outer,
                              mean_prob=round(float(prob[mask].mean()), 6) if n else None,
                              n_pixels=n))
    except Exception as e:
        rings.append({"error": str(e)})
    return rings

def analyse(site_name, meta, geotiff_path, events, fast=False):
    plat, plon = meta["lat"], meta["lon"]

    log("Near-plant event inventory:", indent=2)
    radii_res = {}
    for r in RADII_KM:
        within = [(dist_km(plat, plon, e["latitude"], e["longitude"]), e)
                  for e in events
                  if dist_km(plat, plon, e["latitude"], e["longitude"]) <= r]
        bc = max((e["model_confidence"] for _, e in within), default=None)
        radii_res[r] = dict(count=len(within), best_conf=bc)
        log(f"  ≤{r:2d} km : {len(within):4d} events   best conf = "
            f"{bc:.3f}" if bc is not None else "—", indent=3)

    sc = sc_ratio(geotiff_path, plat, plon)
    direction = sc.get("ctrl_direction", "?")
    log(f"S/C ratio (~20 km {direction} control):", indent=2)
    if sc.get("sc_ratio"):
        r = sc["sc_ratio"]
        label = ("✓✓ STRONG" if r >= 1.5 else "✓  elevated" if r >= 1.1
                 else "~  neutral" if r >= 0.9 else "✗  suppressed")
        log(f"  site={sc['site_mean']:.6f}  ctrl={sc['ctrl_mean']:.6f}  "
            f"ratio={r:.3f}  {label}", indent=3)
    else:
        log(f"  unavailable: {sc.get('error', '?')}", indent=3)

    sweep = threshold_sweep(geotiff_path)
    log("Threshold sweep:", indent=2)
    log(f"  {'Thr':>6}  {'Events':>8}  {'% tile':>8}", indent=3)
    for t in THRESHOLDS:
        if t in sweep:
            s = sweep[t]
            log(f"  {t:>6.2f}  {s['events']:>8d}  {s['pct']:>7.3f}%", indent=3)

    rings = ring_profile(geotiff_path, plat, plon, fast=fast)
    if rings:
        log("Ring profile:", indent=2)
        log(f"  {'Ring':>14}  {'Mean prob':>10}  {'Pixels':>9}", indent=3)
        for rng in rings:
            if "error" in rng:
                log(f"  ring error: {rng['error']}", indent=3); break
            ps = f"{rng['mean_prob']:.6f}" if rng["mean_prob"] is not None else "  (empty)"
            log(f"  {rng['inner_km']:>3d}–{rng['outer_km']:>3d} km  "
                f"{ps:>10}  {rng['n_pixels']:>9,}", indent=3)

    area = tile_area_km2(geotiff_path)
    density = (len(events) / area * 1000) if area and events else None
    prec = (sum(1 for e in events
                if dist_km(plat, plon, e["latitude"], e["longitude"]) <= 10) / len(events)
            if events else None)

    log("Tile stats:", indent=2)
    log(f"  Total events: {len(events)}  |  "
        f"Area: {area:,.0f} km²  |  "
        f"Density: {density:.1f}/1000km²  |  "
        f"Precision@10km: {prec*100:.1f}%" if area and density and prec else
        f"  Total events: {len(events)}", indent=3)

    return dict(
        site=site_name, site_type=meta["site_type"],
        lat=plat, lon=plon, fuel=meta.get("fuel"), country=meta["country"],
        mw=meta.get("mw"), note=meta["note"],
        total_events=len(events), radii=radii_res, sc=sc,
        threshold_sweep=sweep, ring_profile=rings,
        tile_area_km2=round(area, 1) if area else None,
        event_density_per1000km2=round(density, 2) if density else None,
        precision_10km=round(prec, 4) if prec else None,
    )

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(results):
    section("SUMMARY — ranked by S/C ratio")
    emitters = sorted([r for r in results if r["site_type"] == "emitter"],
                      key=lambda r: r["sc"].get("sc_ratio") or -1, reverse=True)
    controls = [r for r in results if r["site_type"] == "control"]

    hdr = f"  {'Site':<22} {'MW':>5} {'Ctry':<4} {'S/C':>7}  {'ev<5':>5} {'ev<10':>6} {'tot':>6}  Verdict"
    log(hdr); log("  " + "─"*len(hdr))
    for r in emitters + controls:
        sc = r["sc"].get("sc_ratio")
        sc_s = f"{sc:.3f}" if sc else "  —  "
        ev5  = r["radii"].get(5,  {}).get("count", 0)
        ev10 = r["radii"].get(10, {}).get("count", 0)
        tot  = r["total_events"]
        mw   = r.get("mw") or 0
        if sc and sc >= 1.5:   verdict = "✓✓ STRONG"
        elif sc and sc >= 1.1: verdict = "✓  elevated"
        elif ev10 > 0:         verdict = "~  events near"
        else:                  verdict = "✗  nothing"
        log(f"  {r['site']:<22} {mw:>5} {r['country']:<4} {sc_s:>7}  "
            f"{ev5:>5} {ev10:>6} {tot:>6}  {verdict}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="Skip ring profile computation (~2 min saved per site)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Re-download even if GeoTIFF exists")
    parser.add_argument("--sites", nargs="+",
                        help="Run only specific sites by key")
    args = parser.parse_args()

    section(f"CH4Net Expanded JRC Sweep  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    sites_list = list(SITES.items())
    if args.sites:
        sites_list = [(k, v) for k, v in sites_list if k in args.sites]

    emitters = sum(1 for _, v in sites_list if v["site_type"] == "emitter")
    controls  = sum(1 for _, v in sites_list if v["site_type"] == "control")
    log(f"Sites: {len(sites_list)} ({emitters} emitters, {controls} controls)")
    log(f"Mode:  {'FAST (no ring profile)' if args.fast else 'full analysis'}")
    log(f"Cache: {'disabled' if args.no_cache else 'enabled'}")
    log(f"Output: {RESULTS_DIR}/")

    summary_path = os.path.join(RESULTS_DIR, "summary.json")
    all_results = []
    if os.path.exists(summary_path):
        all_results = json.load(open(summary_path))
    done = {r["site"] for r in all_results}

    for site_name, meta in sites_list:
        if site_name in done and not args.no_cache:
            log(f"\n[SKIP] {site_name} — already done")
            continue

        mw_s = f"{meta['mw']:,} MW" if meta.get("mw") else "control"
        section(f"[{site_name.upper()}]  {meta['country']}  {mw_s}  —  {meta['note'][:60]}")
        log(f"Coords: ({meta['lat']:.4f}°N, {meta['lon']:.4f}°E)")

        out_dir = os.path.join(RESULTS_DIR, site_name)
        os.makedirs(out_dir, exist_ok=True)

        geotiff = find_geotiff(out_dir)
        if geotiff and not args.no_cache:
            log(f"Cache hit: {os.path.basename(geotiff)}", indent=1)
        else:
            log("Running live_pipeline …", indent=1)
            rc, elapsed = run_pipeline(site_name, meta["lat"], meta["lon"], out_dir)
            geotiff = find_geotiff(out_dir)
            if rc != 0 or not geotiff:
                log(f"FAILED (rc={rc}, {elapsed/60:.1f} min) — see {out_dir}/run.log", indent=1)
                all_results.append(dict(
                    site=site_name, site_type=meta["site_type"],
                    lat=meta["lat"], lon=meta["lon"],
                    fuel=meta.get("fuel"), country=meta["country"],
                    mw=meta.get("mw"), note=meta["note"], status="PIPELINE_FAILED",
                    total_events=0, radii={}, sc={}, threshold_sweep={},
                    ring_profile=[], tile_area_km2=None,
                    event_density_per1000km2=None, precision_10km=None))
                with open(summary_path, "w") as f:
                    json.dump(all_results, f, indent=2, default=str)
                continue
            log(f"Done ({elapsed/60:.1f} min) → {os.path.basename(geotiff)}", indent=1)

        events = load_events(out_dir)
        log(f"GeoTIFF: {os.path.basename(geotiff)}  |  Events in tile: {len(events)}", indent=1)

        result = analyse(site_name, meta, geotiff, events, fast=args.fast)
        done.add(site_name)
        all_results.append(result)
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        log(f"Saved → {summary_path}", indent=1)

    print_summary(all_results)
    log(f"\nExpanded sweep complete at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Results: {summary_path}")
    _log_fh.close()


if __name__ == "__main__":
    main()
