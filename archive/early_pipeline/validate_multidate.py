"""
validate_multidate.py
=====================
Multi-date validation of CH4Net v2 detections across all confirmed and candidate
European emitter sites.

For each site this script:
  1. Finds all L1C .npy tiles already cached in data/npy_cache/ for that tile ID.
  2. Optionally downloads up to N_DATES additional summer-2024 acquisitions
     (skipped if --no-download or enough dates are already cached).
  3. Runs CH4Net v2 baseline inference on every cached acquisition date.
     Inference is skipped if the output TIF already exists (idempotent).
  4. Computes S/C + CFAR for each date and prints a site × date summary table.
  5. Saves results to results_analysis/multidate_validation.json.

Why: Neurath and Niederaußem returned S/C=1.000 on 2024-09-20 (likely cloud/haze).
Testing additional acquisition dates will confirm whether the non-detection was
scene-specific or a genuine model failure on these tiles.

Usage:
    conda activate methane
    caffeinate -i python validate_multidate.py                    # download + eval all
    python validate_multidate.py --no-download                    # eval cached only
    python validate_multidate.py --sites neurath niederaussem     # specific sites
    python validate_multidate.py --sites weisweiler --n-dates 6   # more dates

Output:
    results_bitemporal/<site>/original_<tile_date>.tif   — one TIF per date
    results_analysis/multidate_validation.json           — full results
"""

import os
import sys
import json
import getpass
import logging
import argparse
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

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
        logging.FileHandler("results_analysis/multidate_validation.log"),
    ],
)
log = logging.getLogger(__name__)

# ── Check dependencies ─────────────────────────────────────────────────────────
MISSING = []
try:
    import torch
except ImportError:
    MISSING.append("torch")
try:
    import rasterio
    from rasterio.transform import rowcol
    HAS_RASTERIO = True
except ImportError:
    MISSING.append("rasterio")
    HAS_RASTERIO = False

if MISSING:
    print(f"Missing packages: {', '.join(MISSING)}")
    print(f"Install with:  conda install {' '.join(MISSING)}")
    sys.exit(1)

from src.detection.ch4net_model import CH4NetDetector, Unet
from src.ingestion.preprocessing import (
    tile_scene,
    stitch_predictions,
    GeoMetadata,
    save_prediction_geotiff,
    safe_to_npy,
)
from src.ingestion.copernicus_client import CopernicusClient

# ── Config ────────────────────────────────────────────────────────────────────
NPY_CACHE    = Path("data/npy_cache")
DOWNLOAD_DIR = Path("data/downloads/multidate")
OUT_DIR      = Path("results_bitemporal")
WEIGHTS      = "weights/european_model_v8.pth"
THRESHOLD    = 0.18
TILE_SIZE    = 100

# S/C + CFAR parameters — must stay in sync with apply_bitemporal_diff.py
SC_CROP_PX    = 100
SC_OFFSET_DEG = 0.20
CFAR_K        = 3.0
CLASSIC_THRESH = 1.15

# Download window: June–September 2024 (peak emission season)
DATE_START = "2024-06-01T00:00:00.000Z"
DATE_END   = "2024-09-30T23:59:59.000Z"
MAX_CLOUD  = 15.0  # %; relax to 30 if fewer than N_DATES found

# Default max acquisition dates to download per tile
DEFAULT_N_DATES = 4

B11_IDX = 10
B12_IDX = 11

# ── Site registry ─────────────────────────────────────────────────────────────
# Each entry mirrors apply_bitemporal_diff.py SITES exactly.
# skip_bitemporal is noted here only for display purposes (we only run baseline).
SITES = {
    # Confirmed emitters — expect consistent detection across dates
    "weisweiler":   dict(lat=50.837, lon=6.322,  tile_id="T31UGS",
                         skip_bitemporal=True,
                         label="✓ confirmed",
                         note="1060 MW lignite, Rhineland — S/C=2.091 on two dates"),
    "belchatow":    dict(lat=51.266, lon=19.315, tile_id="T34UCB",
                         skip_bitemporal=True,
                         label="✓ confirmed",
                         note="858 MW lignite, Poland — S/C=27.303 on 2024-08-24"),
    "lippendorf":   dict(lat=51.178, lon=12.378, tile_id="T33UUS",
                         skip_bitemporal=True,
                         label="✓ confirmed",
                         note="891 MW lignite x2, Saxony — S/C=155.362 on 2024-09-22"),

    # Marginal — test consistency
    "boxberg":      dict(lat=51.416, lon=14.565, tile_id="T33UVT",
                         skip_bitemporal=True,
                         label="~ marginal",
                         note="S/C=1.517 on single date; Lausitz bg=1.397"),

    # Non-detections to re-test on better dates
    "neurath":      dict(lat=51.038, lon=6.616,  tile_id="T32ULB",
                         skip_bitemporal=True,
                         label="? retest",
                         note="1060 MW lignite, Rhineland — S/C=1.000 on 2024-09-20 (bad scene)"),
    "niederaussem": dict(lat=50.971, lon=6.667,  tile_id="T32ULB",
                         skip_bitemporal=True,
                         label="? retest",
                         note="924 MW lignite, Rhineland — S/C=1.000 on same bad scene"),

    # FP controls — expect S/C < 1.15 (or suppressed by terrain)
    "groningen":    dict(lat=53.252, lon=6.682,  tile_id="T31UGV",
                         skip_bitemporal=False,
                         label="✗ control",
                         note="Terrain artefact; TROPOMI non-detection"),
    "rybnik":       dict(lat=50.135, lon=18.522, tile_id="T34UCA",
                         skip_bitemporal=True,
                         label="✗ control",
                         note="Coal mine — ring profile increases outward"),
    "maasvlakte":   dict(lat=51.944, lon=4.067,  tile_id="T31UET",
                         skip_bitemporal=True,
                         label="✗ control",
                         note="1070 MW hard coal — Rotterdam port; S/C=0.210 on v8"),

    # ── New sites: top-10 EU CO2 emitters (Brandenburg/Saxony/Poland cluster) ──
    "jaenschwalde":   dict(lat=51.838, lon=14.456, tile_id=None,
                           tile_candidates=["T33UUT", "T33UUS", "T33UVT"],
                           skip_bitemporal=True,
                           label="? new",
                           note="3000 MW lignite, LEAG Brandenburg — EU #2 CO2 ~22 Mt/yr"),
    "schwarze_pumpe": dict(lat=51.536, lon=14.353, tile_id=None,
                           tile_candidates=["T33UUT", "T33UUS"],
                           skip_bitemporal=True,
                           label="? new",
                           note="1600 MW lignite, LEAG Brandenburg — ~13 Mt CO2/yr"),
    "turow":          dict(lat=50.946, lon=14.915, tile_id=None,
                           tile_candidates=["T33UUT", "T33UVT"],
                           skip_bitemporal=True,
                           label="? new",
                           note="1938 MW lignite, PGE SW Poland — ~10 Mt CO2/yr"),
}


# ── Helpers: download ──────────────────────────────────────────────────────────

def site_bbox_wkt(lat, lon, margin_deg=0.3):
    lo, hi = lat - margin_deg, lat + margin_deg
    lo2, hi2 = lon - margin_deg, lon + margin_deg
    return (f"POLYGON(({lo2} {lo},{hi2} {lo},{hi2} {hi},{lo2} {hi},{lo2} {lo}))")


def list_cached_l1c(tile_id: str) -> list[Path]:
    """Return all L1C (non-reference, non-bitemporal) .npy files for this tile."""
    return sorted(
        p for p in NPY_CACHE.glob(f"*{tile_id}*.npy")
        if "MSIL1C" in p.name
        and "_ref_" not in p.name
        and "_bitemporal" not in p.name
    )


def search_tile_products(client, lat, lon, tile_id, max_cloud=MAX_CLOUD):
    """Search Copernicus for all L1C products for tile_id in DATE_START–DATE_END."""
    wkt = site_bbox_wkt(lat, lon)
    try:
        products = client.search_products(
            wkt_polygon=wkt,
            start_date=DATE_START,
            end_date=DATE_END,
            collection="SENTINEL-2",
            max_cloud_cover=max_cloud,
        )
    except Exception as e:
        log.error("  Search failed: %s", e)
        return []

    l1c = [p for p in products if p.tile_id == tile_id and "MSIL1C" in p.name]
    l1c.sort(key=lambda p: (p.cloud_cover or 99, p.acquisition_date))
    log.info("  Found %d L1C products for %s (cloud ≤ %.0f%%)", len(l1c), tile_id, max_cloud)
    return l1c


def cached_dates(tile_id: str) -> set[str]:
    """Return set of acquisition date strings (YYYYMMDD) already in cache."""
    dates = set()
    for p in list_cached_l1c(tile_id):
        # Filename convention: S2X_MSIL1C_YYYYMMDDTHHMMSS_..._T<tile>_...npy
        parts = p.stem.split("_")
        for part in parts:
            if len(part) == 15 and part[8] == "T" and part[:8].isdigit():
                dates.add(part[:8])
                break
    return dates


def download_tile(product, client, tile_id, extract_root):
    """Download + convert one product. Returns Path to .npy or None on error."""
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    log.info("    Downloading %s ...", product.name[:80])
    zip_path = client.download_product(product, str(DOWNLOAD_DIR))
    if zip_path is None:
        log.error("    Download returned None")
        return None

    extract_dir = tempfile.mkdtemp(prefix="s2_multi_", dir=str(extract_root))
    try:
        npy_path, _ = safe_to_npy(
            zip_path=zip_path,
            output_dir=str(NPY_CACHE),
            tile_id=tile_id,
            acquisition_date=product.acquisition_date,
            satellite=product.satellite,
            extract_dir=extract_dir,
        )
        log.info("    Saved: %s", Path(npy_path).name)
        return Path(npy_path)
    except Exception as e:
        log.error("    Conversion failed: %s", e)
        return None
    finally:
        shutil.rmtree(extract_dir, ignore_errors=True)


# ── Helpers: inference ─────────────────────────────────────────────────────────

def run_inference(arr: np.ndarray, detector: CH4NetDetector,
                  geo_meta: GeoMetadata, out_tif: Path) -> None:
    """Run tiled CH4Net inference on arr and save output TIF."""
    H, W, _ = arr.shape
    log.info("    Tiling %d×%d scene → %d×%d patches...", H, W, TILE_SIZE, TILE_SIZE)
    tiles = tile_scene(arr, tile_size=TILE_SIZE, overlap=0)
    log.info("    Running batched inference on %d tiles...", len(tiles))
    preds = detector.detect_batch([t.data for t in tiles], batch_size=32)
    log.info("    Stitching predictions...")
    prob = stitch_predictions(tiles, preds, H, W)
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    save_prediction_geotiff(prob, geo_meta, str(out_tif))


def find_geo_meta(npy_path: Path) -> GeoMetadata | None:
    """Load JSON geo sidecar for a .npy file, returning a GeoMetadata object."""
    # Primary: exact stem + _geo.json
    candidate = npy_path.parent / (npy_path.stem + "_geo.json")
    if candidate.exists():
        return GeoMetadata.load(str(candidate))
    # Fallback: glob for any _geo.json sharing the product name prefix
    prefix = "_".join(npy_path.stem.split("_")[:6])
    for p in npy_path.parent.glob(f"{prefix}*_geo.json"):
        return GeoMetadata.load(str(p))
    return None


# ── Helpers: S/C + CFAR ───────────────────────────────────────────────────────

def safe_crop(arr: np.ndarray, row: int, col: int, half: int = SC_CROP_PX // 2):
    r0, r1 = row - half, row + half
    c0, c1 = col - half, col + half
    if r0 < 0 or c0 < 0 or r1 > arr.shape[0] or c1 > arr.shape[1]:
        return None
    return arr[r0:r1, c0:c1]


def lonlat_to_pixel(tif_path: Path, lon: float, lat: float):
    with rasterio.open(tif_path) as src:
        xs, ys = rasterio.warp.transform("EPSG:4326", src.crs, [lon], [lat])
        row, col = rowcol(src.transform, xs[0], ys[0])
    return int(row), int(col)


def compute_sc_ratio(tif_path: Path, lat: float, lon: float) -> dict:
    offsets = [
        ( SC_OFFSET_DEG, 0.0,           "N"),
        (-SC_OFFSET_DEG, 0.0,           "S"),
        (0.0,            SC_OFFSET_DEG, "E"),
        (0.0,           -SC_OFFSET_DEG, "W"),
    ]
    try:
        with rasterio.open(tif_path) as src:
            prob = src.read(1).astype(np.float32)
        s_row, s_col = lonlat_to_pixel(tif_path, lon, lat)
    except Exception as e:
        return {"error": str(e)}

    site_crop = safe_crop(prob, s_row, s_col)
    if site_crop is None:
        return {"error": "site_out_of_bounds"}

    sm = float(site_crop.mean())
    ctrl_means = []
    first_result = None

    for dlat, dlon, direction in offsets:
        try:
            c_row, c_col = lonlat_to_pixel(tif_path, lon + dlon, lat + dlat)
        except Exception:
            continue
        ctrl_crop = safe_crop(prob, c_row, c_col)
        if ctrl_crop is None:
            continue
        cm = float(ctrl_crop.mean())
        ctrl_means.append(cm)
        if first_result is None:
            sc = sm / cm if cm > 1e-9 else float("inf")
            first_result = dict(site_mean=round(sm, 6), ctrl_mean=round(cm, 6),
                                sc_ratio=round(sc, 4), ctrl_direction=direction)

    if first_result is None:
        return {"error": "all_directions_oob"}

    mu_ctrl    = float(np.mean(ctrl_means))
    sigma_ctrl = float(np.std(ctrl_means, ddof=0)) if len(ctrl_means) >= 2 else 0.0
    sc_cfar    = sm / mu_ctrl if mu_ctrl > 1e-9 else float("inf")
    cv_ctrl    = sigma_ctrl / mu_ctrl if mu_ctrl > 1e-9 else 0.0
    cfar_thresh_ratio = CLASSIC_THRESH + CFAR_K * cv_ctrl
    cfar_detect = bool(sc_cfar > cfar_thresh_ratio)

    return {
        **first_result,
        "sc_cfar":           round(sc_cfar, 4),
        "ctrl_mu":           round(mu_ctrl, 6),
        "ctrl_sigma":        round(sigma_ctrl, 6),
        "cv_ctrl":           round(cv_ctrl, 4),
        "cfar_thresh_ratio": round(cfar_thresh_ratio, 4),
        "cfar_detect":       cfar_detect,
        "cfar_margin":       round(sc_cfar - cfar_thresh_ratio, 4),
        "ctrl_n":            len(ctrl_means),
        "ctrl_all_means":    [round(v, 6) for v in ctrl_means],
    }


# ── Phase 1: optional download ─────────────────────────────────────────────────

def download_phase(sites_to_run: dict, n_dates: int):
    """Download up to n_dates summer-2024 L1C tiles per tile_id (deduped)."""
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

    # Deduplicate: one download per tile_id even if multiple sites share it
    tile_to_sites = {}
    for name, meta in sites_to_run.items():
        tid = meta["tile_id"]
        tile_to_sites.setdefault(tid, []).append((name, meta))

    extract_root = Path(tempfile.gettempdir())

    for tile_id, site_pairs in tile_to_sites.items():
        name0, meta0 = site_pairs[0]
        cached = list_cached_l1c(tile_id)
        already = len(cached)
        log.info("=" * 65)
        log.info("Tile: %s  (%s)  —  %d already cached",
                 tile_id, ", ".join(n for n, _ in site_pairs), already)

        if already >= n_dates:
            log.info("  %d dates cached — target %d met, skipping download", already, n_dates)
            continue

        need = n_dates - already
        log.info("  Need %d more date(s). Searching catalog...", need)

        products = search_tile_products(client, meta0["lat"], meta0["lon"], tile_id)
        if not products and MAX_CLOUD < 30:
            log.info("  Relaxing cloud filter to 30%%...")
            products = search_tile_products(client, meta0["lat"], meta0["lon"], tile_id, max_cloud=30.0)

        have_dates = cached_dates(tile_id)
        new_products = [
            p for p in products
            if p.acquisition_date[:8] not in have_dates
        ][:need]

        if not new_products:
            log.info("  No new products to download for %s", tile_id)
            continue

        for p in new_products:
            log.info("  → %s  cloud=%.1f%%  date=%s",
                     p.name[:70], p.cloud_cover or 0, p.acquisition_date[:10])
            download_tile(p, client, tile_id, extract_root)


# ── Phase 2+3: inference + S/C per site × date ────────────────────────────────

def eval_phase(sites_to_run: dict) -> dict:
    """Run inference on all cached dates per site. Return nested results dict."""
    log.info("\nLoading CH4Net v2 weights from %s ...", WEIGHTS)
    detector = CH4NetDetector(WEIGHTS)
    log.info("  Model loaded OK")

    all_results = {}

    for site_name, meta in sites_to_run.items():
        tile_id = meta["tile_id"]
        lat, lon = meta["lat"], meta["lon"]

        log.info("\n" + "=" * 65)
        log.info("Site: %-15s  tile: %s  (%s)", site_name.upper(), tile_id, meta["label"])
        log.info("Note: %s", meta["note"])

        npy_files = list_cached_l1c(tile_id)
        if not npy_files:
            log.warning("  No cached L1C .npy files for tile %s — skipping", tile_id)
            all_results[site_name] = {"status": "no_data", "tile_id": tile_id}
            continue

        log.info("  Found %d cached date(s) for tile %s", len(npy_files), tile_id)

        site_dir = OUT_DIR / site_name
        site_dir.mkdir(parents=True, exist_ok=True)

        date_results = {}

        for npy_path in sorted(npy_files):
            # Extract acquisition date from filename
            parts = npy_path.stem.split("_")
            acq_date = next(
                (p[:8] for p in parts if len(p) == 15 and p[8] == "T" and p[:8].isdigit()),
                npy_path.stem[-8:],
            )

            out_tif = site_dir / f"original_{npy_path.stem}.tif"

            log.info("  Date %s:", acq_date)

            # Run inference if TIF not already cached
            if out_tif.exists():
                log.info("    [cached TIF] %s", out_tif.name)
            else:
                geo_meta = find_geo_meta(npy_path)
                if geo_meta is None:
                    log.warning("    No geo metadata for %s — skipping", npy_path.name)
                    date_results[acq_date] = {"error": "no_geo_meta"}
                    continue

                log.info("    Loading array (may take ~30s)...")
                arr = np.load(npy_path)
                log.info("    Shape: %s  Running inference...", arr.shape)
                run_inference(arr, detector, geo_meta, out_tif)
                del arr

            # Compute S/C + CFAR
            sc = compute_sc_ratio(out_tif, lat, lon)
            if "error" in sc:
                log.warning("    S/C error: %s", sc["error"])
                date_results[acq_date] = {"error": sc["error"]}
                continue

            r    = sc["sc_ratio"]
            cfar = sc["cfar_detect"]
            cv   = sc["cv_ctrl"]
            thr  = sc["cfar_thresh_ratio"]
            detect_str = "DETECT" if (r > CLASSIC_THRESH) else "no"

            log.info("    S/C=%.3f  CFAR=%s  (thresh=%.3f, CV=%.3f)  site_mean=%.6f",
                     r, "DETECT" if cfar else "no", thr, cv, sc["site_mean"])

            date_results[acq_date] = {
                "tif":               str(out_tif),
                "sc_ratio":          r,
                "sc_cfar":           sc["sc_cfar"],
                "site_mean":         sc["site_mean"],
                "ctrl_mean":         sc["ctrl_mean"],
                "ctrl_mu":           sc["ctrl_mu"],
                "cv_ctrl":           cv,
                "cfar_thresh_ratio": thr,
                "cfar_detect":       cfar,
                "cfar_margin":       sc["cfar_margin"],
                "classic_detect":    bool(r > CLASSIC_THRESH),
            }

        all_results[site_name] = {
            "status":        "ok",
            "tile_id":       tile_id,
            "label":         meta["label"],
            "skip_bitemporal": meta.get("skip_bitemporal", False),
            "dates":         date_results,
        }

    return all_results


# ── Phase 4: summary table ─────────────────────────────────────────────────────

def print_summary(all_results: dict):
    """Print site × date S/C table grouped by site."""
    print("\n")
    print("=" * 95)
    print("  MULTI-DATE VALIDATION — CH4Net v2  (baseline S/C, summer 2024)")
    print("=" * 95)
    print(f"  {'Site':<15} {'Label':<14} {'Date':>10}  {'S/C':>8}  "
          f"{'CFAR':>6}  {'thresh':>7}  {'CV':>6}  {'Result'}")
    print("  " + "-" * 92)

    for site_name, res in all_results.items():
        if res.get("status") != "ok":
            print(f"  {site_name:<15} SKIPPED ({res.get('status','?')})")
            continue

        label  = res.get("label", "")
        dates  = res.get("dates", {})
        if not dates:
            print(f"  {site_name:<15} {label:<14} — no dates evaluated")
            continue

        for i, (date, dr) in enumerate(sorted(dates.items())):
            prefix = site_name if i == 0 else ""
            lab    = label     if i == 0 else ""

            if "error" in dr:
                print(f"  {prefix:<15} {lab:<14} {date:>10}  {'—':>8}  {'—':>6}  {'—':>7}  {'—':>6}  ERROR: {dr['error']}")
                continue

            r    = dr["sc_ratio"]
            cfar = dr["cfar_detect"]
            thr  = dr["cfar_thresh_ratio"]
            cv   = dr["cv_ctrl"]

            result_str = (
                "✓ EMITTER DETECTED" if r > CLASSIC_THRESH
                else "✗ non-detection"
            )
            if cfar:
                result_str += " [CFAR]"

            print(f"  {prefix:<15} {lab:<14} {date:>10}  {r:>8.3f}  "
                  f"{'yes' if cfar else 'no':>6}  {thr:>7.3f}  {cv:>6.3f}  {result_str}")

        # Summary row: detection rate
        valid = [d for d in dates.values() if "sc_ratio" in d]
        n_det = sum(1 for d in valid if d["sc_ratio"] > CLASSIC_THRESH)
        if valid:
            sc_vals = [d["sc_ratio"] for d in valid]
            print(f"  {'':15} {'':14} {'SUMMARY':>10}  "
                  f"{sum(sc_vals)/len(sc_vals):>8.3f}  "
                  f"{'':>6}  {'':>7}  {'':>6}  "
                  f"mean S/C, {n_det}/{len(valid)} dates detected")
        print("  " + "-" * 92)

    print("=" * 95)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multi-date CH4Net validation")
    parser.add_argument("--sites", nargs="+", default=None,
                        help="Site names to evaluate (default: all)")
    parser.add_argument("--no-download", action="store_true",
                        help="Skip Copernicus download; evaluate cached tiles only")
    parser.add_argument("--n-dates", type=int, default=DEFAULT_N_DATES,
                        help=f"Target number of acquisition dates per tile (default: {DEFAULT_N_DATES})")
    args = parser.parse_args()

    # Filter sites
    if args.sites:
        missing = [s for s in args.sites if s not in SITES]
        if missing:
            print(f"Unknown site(s): {missing}")
            print(f"Available: {list(SITES.keys())}")
            sys.exit(1)
        sites_to_run = {k: v for k, v in SITES.items() if k in args.sites}
    else:
        sites_to_run = SITES

    print("=" * 65)
    print("  Multi-date CH4Net Validation — European Emitter Sites")
    print(f"  Sites:   {', '.join(sites_to_run.keys())}")
    print(f"  Window:  Jun 2024 – Sep 2024")
    print(f"  N dates: {args.n_dates} per tile")
    print(f"  Download: {'disabled' if args.no_download else 'enabled'}")
    print("=" * 65)

    # Phase 1: download
    if not args.no_download:
        download_phase(sites_to_run, args.n_dates)
    else:
        print("\n[download skipped — using cached tiles only]\n")

    # Phase 2+3: inference + S/C
    print("\n[Evaluating cached tiles...]\n")
    all_results = eval_phase(sites_to_run)

    # Phase 4: table
    print_summary(all_results)

    # Save JSON
    out_json = Path("results_analysis/multidate_validation.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("\nResults saved → %s", out_json)


if __name__ == "__main__":
    main()
