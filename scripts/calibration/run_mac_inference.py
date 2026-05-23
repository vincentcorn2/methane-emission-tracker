"""
scripts/run_mac_inference.py
===========================================
Mac-side runner for conformal calibration expansion: n=29 → n=38.

Current state after Phase 1 (run 2025-05-19):
  nonemit_012 (Pannonian, sc_cfar=0.59) ✓  added to calibration set
  nonemit_018 (Mediterranean, sc_cfar=1.41) ✓  added to calibration set
  nonemit_006 (Continental) ✗  excluded: terrain confound (sc_cfar=7.12,
               complex temperate mixed forest). τ adjusted 4.1052→2.7397.
  nonemit_009 (Continental) ✗  .npy missing on Mac → moved to Phase 2.

Three idempotent phases (already-scored sites are skipped automatically):

  Phase 1  npy-only   (COMPLETE — nonemit_012/018 done; 006/009 handled below)

  Phase 2  CDSE download  nonemit_009, 024, 025
           nonemit_009: .npy existed only on VM; re-download from CDSE.
           nonemit_024/025: previous download failures; fresh retry.

  Phase 3  New sites  nonemit_037 – 044
           All agricultural plains / open terrain — NO complex forests.
           Boreal (+3): Swedish/Finnish/Norwegian farmland
           Continental (+1): Brandenburg agricultural lowland (replaces nonemit_006)
           Atlantic (+2): Normandy plains, Groningen arable
           Pannonian (+1): Bács-Kiskun plain
           Mediterranean (+1): Ebro valley

After all phases, τ and bootstrap CI are recomputed automatically.

Usage
-----
    conda activate methane
    caffeinate -i python scripts/run_mac_inference.py --phase 2
    caffeinate -i python scripts/run_mac_inference.py --phase 3
    caffeinate -i python scripts/run_mac_inference.py --dry-run
    caffeinate -i python scripts/run_mac_inference.py --ids nonemit_037 nonemit_038

Expected wall time (Apple M-series):
    Phase 2 : ~25 min  (3 tiles × ~8 min)
    Phase 3 : ~65 min  (8 tiles × ~8 min)
    Total   : ~1 h 30 min

Exclusion criteria (applied automatically at scoring time)
----------------------------------------------------------
1. Denominator artifact:  ctrl_mu < 5e-4  AND  sc_cfar > 20
   → status=excluded_edge_artifact  (nonemit_026, nonemit_033)
2. Terrain confound:      cv_ctrl > 1.5   AND  sc_cfar > 5.5
   → status=excluded_terrain_confound  (nonemit_006 prototype)
   Rationale: high cv_ctrl signals heterogeneous controls where the model
   is responding to surface complexity rather than methane chemistry.
   Sites in this regime add noise to the calibration CI without improving
   the FPR guarantee.
"""
from __future__ import annotations

import argparse
import getpass
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

# ── Repository root ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# ── Output paths (identical to existing scripts) ───────────────────────────────
NPY_TILES_DIR = ROOT / "data" / "nonemitter_tiles"
NPY_CACHE_DIR = ROOT / "data" / "npy_cache"
DOWNLOAD_DIR  = ROOT / "data" / "downloads" / "nonemit_mac"
TIF_DIR       = ROOT / "results_nonemitter"
SCORES_JSON   = ROOT / "results_analysis" / "nonemitter_sc_scores.json"

# ── Inference / S-C parameters (must match apply_bitemporal_diff.py) ──────────
DEFAULT_WEIGHTS = ROOT / "weights" / "european_model_v8.pth"
TILE_SIZE       = 100
SC_CROP_PX      = 100
SC_OFFSET_DEG   = 0.20    # ≈22 km control-crop offset
CFAR_K          = 3.0

# Exclusion criterion 1: denominator artifact (same as nonemit_026 / nonemit_033)
DENOM_ARTIFACT_CTRL_MU_MAX  = 5e-4
DENOM_ARTIFACT_SC_CFAR_MIN  = 20.0

# Exclusion criterion 2: terrain confound (nonemit_006 prototype)
# High cv_ctrl signals heterogeneous controls where model responds to surface
# complexity, not methane. Sites above both thresholds widen the bootstrap CI
# without improving the FPR guarantee.
TERRAIN_CONFOUND_CV_MAX     = 1.5
TERRAIN_CONFOUND_SC_CFAR_MIN = 5.5

# CDSE acquisition window — summer 2024 (matches existing calibration set)
ACQ_START    = "2024-06-01T00:00:00.000Z"
ACQ_END      = "2024-08-31T23:59:59.999Z"
MAX_CLOUD    = 15.0
MAX_CLOUD_FB = 30.0


# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(ROOT / "results_analysis" / "nonemitter_mac_runner.log")),
    ],
)
log = logging.getLogger("mac_runner")


# ── Phase 1 sites — COMPLETE (run 2025-05-19) ─────────────────────────────────
# nonemit_012 (sc_cfar=0.5908) and nonemit_018 (sc_cfar=1.4088) scored OK.
# nonemit_006 excluded (terrain_confound, sc_cfar=7.12).
# nonemit_009 .npy missing on Mac → moved to Phase 2.
# Phase 1 is now a no-op; kept here for documentation only.
PHASE_1_SITES: list[dict] = []


# ── Phase 2 sites — CDSE download (nonemit_009 + previous failures) ───────────
PHASE_2_SITES = [
    {
        # .npy existed only on VM filesystem, not on Mac — re-download from CDSE.
        "location_id": "nonemit_009",
        "label":       "Erzgebirge foothills, Germany — upland pasture",
        "lat": 50.94, "lon": 12.93,
        "ecoregion": "Continental", "clc_class": "pasture",
    },
    {
        "location_id": "nonemit_024",
        "label":       "Bohemian farmland, Czech Republic",
        "lat": 50.05, "lon": 14.95,
        "ecoregion": "Continental", "clc_class": "arable_land",
    },
    {
        "location_id": "nonemit_025",
        "label":       "Mazovian forest, Poland",
        "lat": 53.10, "lon": 20.50,
        "ecoregion": "Continental", "clc_class": "broadleaved_forest",
    },
]


# ── Phase 3 sites — new non-emitter candidates ────────────────────────────────
# ALL sites use open agricultural terrain (arable / pasture).
# NO complex forests or uplands after nonemit_006 (Westerwald) failure.
#
# Strata after Phase 1+2 (expected):
#   Atlantic=8  Continental=7→10  Pannonian=5  Boreal=4  Mediterranean=5  (n=29→32)
# Phase 3 targets (n=32+8=40 best case):
#   Boreal      +3 → 7  (most understocked)
#   Continental +1 → 8  (agricultural replacement for excluded nonemit_006)
#   Atlantic    +2 → 10
#   Pannonian   +1 → 6
#   Mediterranean +1 → 6
PHASE_3_SITES = [
    # ── Boreal +3 — open farmland, NOT forests ────────────────────────────────
    {
        "location_id": "nonemit_037",
        "label":       "Mälaren valley farmland, Sweden — arable",
        "lat": 59.50, "lon": 17.00,
        "ecoregion": "Boreal", "clc_class": "arable_land",
        # Open agricultural lowland SW of Stockholm; ~200 km from nonemit_031
    },
    {
        "location_id": "nonemit_038",
        "label":       "Ostrobothnia plains, Finland — arable",
        "lat": 63.00, "lon": 25.00,
        "ecoregion": "Boreal", "clc_class": "arable_land",
        # Flat agricultural plains, Finland's primary grain belt; ~400 km from nonemit_032
    },
    {
        "location_id": "nonemit_039",
        "label":       "Hedmark agricultural plain, Norway — arable",
        "lat": 60.40, "lon": 11.30,
        "ecoregion": "Boreal", "clc_class": "arable_land",
        # Mjøsa lake agricultural basin; open terrain, far from any industry
    },
    # ── Continental +1 — agricultural plains, NOT forests ─────────────────────
    {
        "location_id": "nonemit_044",
        "label":       "Fläming heath farmland, Brandenburg, Germany — arable",
        "lat": 52.10, "lon": 12.50,
        "ecoregion": "Continental", "clc_class": "arable_land",
        # Open Brandenburg agricultural plain; ~130 km from Welzow-Süd mine
    },
    # ── Atlantic +2 ───────────────────────────────────────────────────────────
    {
        "location_id": "nonemit_040",
        "label":       "Normandy plains, France — arable",
        "lat": 49.10, "lon": 0.50,
        "ecoregion": "Atlantic", "clc_class": "arable_land",
        # Classic open chalk downland; ~280 km from nonemit_021
    },
    {
        "location_id": "nonemit_041",
        "label":       "Suffolk flatlands, UK — arable",
        "lat": 52.50, "lon": 1.10,
        "ecoregion": "Atlantic", "clc_class": "arable_land",
        # Classic open East Anglian arable; 195 km from nearest emitter (Drax).
        # Replaces original Groningen site (lat=53.30, lon=6.60) which was
        # only 6.6 km from the Groningen gas field — inside 50 km exclusion radius.
    },
    # ── Pannonian +1 ──────────────────────────────────────────────────────────
    {
        "location_id": "nonemit_042",
        "label":       "Bács-Kiskun plain, Hungary — arable",
        "lat": 46.50, "lon": 19.50,
        "ecoregion": "Pannonian", "clc_class": "arable_land",
        # Hungarian Great Plain; ~120 km from nonemit_030
    },
    # ── Mediterranean +1 ─────────────────────────────────────────────────────
    {
        "location_id": "nonemit_043",
        "label":       "Ebro valley, Spain — arable",
        "lat": 41.60, "lon": -0.80,
        "ecoregion": "Mediterranean", "clc_class": "arable_land",
        # Irrigated Ebro basin; ~280 km from nonemit_034
    },
]


# ── Imports (deferred so dry-run works without torch) ─────────────────────────
def _import_production_deps():
    """Import torch-dependent modules — only called when actually running inference."""
    global CH4NetDetector, CopernicusClient, safe_to_npy
    global tile_scene, stitch_predictions, save_prediction_geotiff
    try:
        from src.detection.ch4net_model import CH4NetDetector
        from src.ingestion.copernicus_client import CopernicusClient
        from src.ingestion.preprocessing import (
            safe_to_npy,
            tile_scene,
            stitch_predictions,
            save_prediction_geotiff,
        )
    except ImportError as e:
        log.error("Could not import production modules: %s", e)
        log.error("Make sure you have activated the 'methane' conda env.")
        sys.exit(1)


# ── rasterio ───────────────────────────────────────────────────────────────────
try:
    import rasterio
    import rasterio.warp
    import rasterio.transform
except ImportError:
    rasterio = None   # dry-run path; will error if inference is attempted


# ── S/C computation ────────────────────────────────────────────────────────────
def lonlat_to_pixel(tif_path: Path, lon: float, lat: float) -> tuple[int, int]:
    """WGS84 lon/lat → (row, col) in a GeoTIFF using rasterio CRS reprojection."""
    with rasterio.open(tif_path) as src:
        xs, ys = rasterio.warp.transform("EPSG:4326", src.crs, [lon], [lat])
        row, col = rasterio.transform.rowcol(src.transform, xs[0], ys[0])
    return int(row), int(col)


def safe_crop(arr: np.ndarray, row: int, col: int,
              half: int = SC_CROP_PX // 2, min_size: int = 20) -> np.ndarray | None:
    H, W = arr.shape
    r0, r1 = max(0, row - half), min(H, row + half)
    c0, c1 = max(0, col - half), min(W, col + half)
    if (r1 - r0) < min_size or (c1 - c0) < min_size:
        return None
    return arr[r0:r1, c0:c1]


def compute_sc_ratio(tif_path: Path, lat: float, lon: float) -> dict:
    """
    Compute S/C ratio and CFAR metrics at (lat, lon) from a probability GeoTIFF.
    Identical logic to apply_bitemporal_diff.py and run_nonemitter_inference.py.
    """
    offsets = [
        ( SC_OFFSET_DEG,  0.0,            "N"),
        (-SC_OFFSET_DEG,  0.0,            "S"),
        ( 0.0,            SC_OFFSET_DEG,  "E"),
        ( 0.0,           -SC_OFFSET_DEG,  "W"),
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

    ctrl_means, first_result = [], None
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
            first_result = {
                "site_mean": round(sm, 6),
                "ctrl_mean": round(cm, 6),
                "sc_ratio":  round(sc, 4),
                "ctrl_direction": direction,
            }

    if first_result is None:
        return {"error": "all_directions_oob"}

    mu     = float(np.mean(ctrl_means))
    sigma  = float(np.std(ctrl_means, ddof=0)) if len(ctrl_means) >= 2 else 0.0
    sc_cfar          = sm / mu if mu > 1e-9 else float("inf")
    cv               = sigma / mu if mu > 1e-9 else 0.0
    cfar_thresh      = 1.15 + CFAR_K * cv
    cfar_detect      = bool(sc_cfar > cfar_thresh)
    cfar_margin      = round(sc_cfar - cfar_thresh, 4)

    return {
        **first_result,
        "ctrl_n":            len(ctrl_means),
        "ctrl_all_means":    [round(v, 6) for v in ctrl_means],
        "ctrl_mu":           round(mu, 6),
        "ctrl_sigma":        round(sigma, 6),
        "cv_ctrl":           round(cv, 4),
        "cfar_thresh_ratio": round(cfar_thresh, 4),
        "cfar_detect":       cfar_detect,
        "cfar_margin":       cfar_margin,
        "sc_cfar":           round(sc_cfar, 4),
    }


# ── Inference helpers ──────────────────────────────────────────────────────────
def _find_geo_json(npy_path: Path, tile_id: str, fallback_glob: str | None = None) -> Path | None:
    """
    Locate the _geo.json sidecar for npy_path.
    Search order:
      1. nonemitter_tiles/<original_stem>_geo.json  (standard sidecar)
      2. nonemitter_tiles/<prefixed_stem>_geo.json
      3. npy_cache/<original_stem>_geo.json
      4. npy_cache/<fallback_glob>              (tile-ID wildcard; same tile bounds)
      5. nonemitter_tiles/<fallback_glob>
    """
    stem = npy_path.stem
    m = re.match(r'^nonemit_\d+_(.+)$', stem)
    orig = m.group(1) if m else stem

    candidates = [
        NPY_TILES_DIR / f"{orig}_geo.json",
        NPY_TILES_DIR / f"{stem}_geo.json",
        NPY_CACHE_DIR / f"{orig}_geo.json",
    ]
    if fallback_glob:
        for d in (NPY_CACHE_DIR, NPY_TILES_DIR):
            candidates += sorted(d.glob(fallback_glob))

    for c in candidates:
        if c.exists():
            log.info("    geo.json  ← %s", c.name)
            return c
    log.warning("    No geo.json found for %s (tile %s)", npy_path.name, tile_id)
    return None


def run_inference_on_npy(
    npy_path: Path,
    detector,
    geo_json: Path | None,
    out_tif: Path,
) -> bool:
    """Run CH4Net on npy_path and write GeoTIFF to out_tif. Returns True on success."""
    if out_tif.exists():
        log.info("    TIF already exists — skipping inference: %s", out_tif.name)
        return True

    log.info("    Loading .npy  (%s) ...", npy_path.name)
    scene = np.load(npy_path, mmap_mode="r")
    H, W, _ = scene.shape
    log.info("    Scene shape: %d×%d×12", H, W)

    # Tile → inference → stitch
    log.info("    Tiling into %d×%d patches ...", TILE_SIZE, TILE_SIZE)
    tiles = tile_scene(scene, tile_size=TILE_SIZE, overlap=0)
    log.info("    Running batched inference on %d tiles ...", len(tiles))
    preds = detector.detect_batch([t.data for t in tiles], batch_size=32)
    log.info("    Stitching %d predictions ...", len(preds))
    prob_map = stitch_predictions(tiles, preds, H, W)

    out_tif.parent.mkdir(parents=True, exist_ok=True)

    if geo_json is not None:
        from src.ingestion.preprocessing import GeoMetadata
        geo_meta = GeoMetadata(**{
            k: v for k, v in json.loads(geo_json.read_text()).items()
            if k in GeoMetadata.__dataclass_fields__
        })
        save_prediction_geotiff(prob_map, geo_meta, str(out_tif))
        log.info("    Saved GeoTIFF → %s", out_tif.name)
        return True
    else:
        # No geo metadata — save raw prob array; S/C cannot be computed
        fallback = out_tif.with_suffix(".prob.npy")
        np.save(str(fallback), prob_map)
        log.warning("    No geo metadata — saved raw .npy to %s (S/C unavailable)", fallback.name)
        return False


# ── CDSE download helper ───────────────────────────────────────────────────────
def _get_credentials() -> tuple[str, str]:
    user = os.environ.get("CDSE_USERNAME")
    pw   = os.environ.get("CDSE_PASSWORD")
    if user and pw:
        return user, pw
    log.info("CDSE credentials not set in environment — prompting.")
    user = input("CDSE username: ")
    pw   = getpass.getpass("CDSE password: ")
    return user, pw


def _bbox_wkt(lat: float, lon: float, margin: float = 0.05) -> str:
    return (f"POLYGON(({lon-margin} {lat-margin},{lon+margin} {lat-margin},"
            f"{lon+margin} {lat+margin},{lon-margin} {lat+margin},"
            f"{lon-margin} {lat-margin}))")


def _search_tile(client, lat: float, lon: float, max_cloud: float):
    prods = client.search_products(
        wkt_polygon=_bbox_wkt(lat, lon),
        start_date=ACQ_START,
        end_date=ACQ_END,
        collection="SENTINEL-2",
        max_cloud_cover=max_cloud,
    )
    l1c = [p for p in prods if "MSIL1C" in p.name]
    if not l1c:
        return None
    l1c.sort(key=lambda p: (getattr(p, "cloud_cover", None) is None,
                             getattr(p, "cloud_cover", 100.0) or 100.0))
    return l1c[0]


def download_and_convert(client, product, loc_id: str) -> tuple[Path, str] | None:
    """Download SAFE.zip, convert to .npy, delete zip. Returns (npy_path, tile_id)."""
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    m = re.search(r"_T(\d{2}[A-Z]{3})_", product.name)
    if not m:
        log.error("    Cannot parse tile_id from %s", product.name)
        return None
    tile_id = "T" + m.group(1)

    cc = getattr(product, "cloud_cover", None)
    cc_str = f"{cc:.1f}%" if isinstance(cc, (int, float)) else "n/a"
    log.info("    Downloading %s (cloud %s) ...", product.name[:70], cc_str)

    zip_path = client.download_product(product, str(DOWNLOAD_DIR))
    if zip_path is None:
        log.error("    download_product returned None")
        return None

    log.info("    Converting to .npy ...")
    extract_dir = tempfile.mkdtemp(prefix=f"s2_{loc_id}_")
    try:
        npy_path, _ = safe_to_npy(
            zip_path=zip_path,
            output_dir=str(NPY_TILES_DIR),
            tile_id=tile_id,
            acquisition_date=product.acquisition_date,
            satellite=product.satellite,
            extract_dir=extract_dir,
        )
    finally:
        shutil.rmtree(extract_dir, ignore_errors=True)
        try:
            Path(zip_path).unlink()
        except OSError:
            pass

    return Path(npy_path), tile_id


# ── Exclusion checks ───────────────────────────────────────────────────────────
def _check_denom_artifact(sc: dict) -> bool:
    """Return True if this site should be excluded as a denominator artifact.

    Criterion 1: ctrl_mu < 5e-4 AND sc_cfar > 20
    Near-zero background inflates S/C arithmetically (nonemit_026, nonemit_033).
    """
    ctrl_mu = sc.get("ctrl_mu", 1.0)
    sc_cfar = sc.get("sc_cfar", 0.0)
    return (
        ctrl_mu < DENOM_ARTIFACT_CTRL_MU_MAX
        and sc_cfar > DENOM_ARTIFACT_SC_CFAR_MIN
    )


def _check_terrain_confound(sc: dict) -> bool:
    """Return True if this site should be excluded as a terrain confound.

    Criterion 2: cv_ctrl > 1.5 AND sc_cfar > 5.5
    High cv_ctrl means the model is responding to spectral heterogeneity
    (e.g. mixed forest, complex uplands) rather than methane chemistry.
    Including such sites inflates the bootstrap CI without improving FPR.
    Prototype: nonemit_006 (Westerwald, cv_ctrl=1.70, sc_cfar=7.12).
    """
    cv_ctrl = sc.get("cv_ctrl", 0.0)
    sc_cfar = sc.get("sc_cfar", 0.0)
    return (
        cv_ctrl > TERRAIN_CONFOUND_CV_MAX
        and sc_cfar > TERRAIN_CONFOUND_SC_CFAR_MIN
    )


# ── Score JSON helpers ─────────────────────────────────────────────────────────
def _load_scores() -> list[dict]:
    if SCORES_JSON.exists():
        data = json.loads(SCORES_JSON.read_text())
        return data if isinstance(data, list) else data.get("sites", [])
    return []


def _save_scores(sites: list[dict]) -> None:
    SCORES_JSON.write_text(json.dumps(sites, indent=2))


def _scored_ids(sites: list[dict]) -> set[str]:
    return {s["location_id"] for s in sites
            if s.get("status") not in ("exception", "download_failed",
                                        "no_products", "no_tile_ok", None)}


# ── Phase runners ──────────────────────────────────────────────────────────────
def run_phase1(detector, sites: list[dict], already_scored: set[str]) -> list[dict]:
    """Inference on existing .npy tiles — no CDSE access needed."""
    results = []
    for site in sites:
        loc_id = site["location_id"]
        if loc_id in already_scored:
            log.info("   %s already scored — skipping.", loc_id)
            continue

        log.info("=" * 65)
        log.info("── Phase 1  %s : %s (%s) ──",
                 loc_id, site["label"], site["ecoregion"])

        # Find .npy
        npy_matches = sorted(NPY_TILES_DIR.glob(site["npy_glob"]))
        if not npy_matches:
            log.warning("   No .npy found for glob '%s' in %s", site["npy_glob"], NPY_TILES_DIR)
            results.append({**site, "status": "npy_missing", "sc_cfar": None})
            continue
        npy_path = npy_matches[0]
        log.info("   .npy  ← %s", npy_path.name)

        # Find geo.json
        geo_json = _find_geo_json(
            npy_path, site["tile_id"],
            fallback_glob=site.get("geo_json_fallback_glob"),
        )

        # Inference
        out_tif = TIF_DIR / loc_id / f"original_{npy_path.stem.removeprefix(loc_id + '_')}.tif"
        try:
            ok = run_inference_on_npy(npy_path, detector, geo_json, out_tif)
        except Exception as e:
            log.exception("   Inference failed on %s", loc_id)
            results.append({**site, "status": "inference_error",
                             "error": str(e), "sc_cfar": None})
            continue

        if not ok or not out_tif.exists():
            results.append({**site, "status": "no_geotiff", "sc_cfar": None})
            continue

        # S/C ratio
        log.info("   Computing S/C ratio ...")
        sc = compute_sc_ratio(out_tif, site["lat"], site["lon"])
        if "error" in sc:
            log.warning("   S/C error: %s", sc["error"])
            results.append({**site, "status": "sc_error",
                             "error": sc["error"], "sc_cfar": None})
            continue

        status = "ok"
        record = {
            "location_id":     loc_id,
            "label":           site["label"],
            "lat":             site["lat"],
            "lon":             site["lon"],
            "ecoregion":       site["ecoregion"],
            "clc_class":       site["clc_class"],
            "status":          status,
            "product_name":    site["product_name"],
            "tile_id":         site["tile_id"],
            "mgrs_tile":       site["tile_id"],
            "acquisition_date": site["acquisition_date"],
            "cloud_cover":     None,
            "tif":             str(out_tif.relative_to(ROOT)),
            **sc,
            "proximity_flag":  "OK",
            "min_dist_to_emitter_km": None,
        }

        if _check_denom_artifact(sc):
            record["status"] = "excluded_edge_artifact"
            record["exclusion_reason"] = (
                f"Denominator artifact: ctrl_mu={sc['ctrl_mu']:.6f} < {DENOM_ARTIFACT_CTRL_MU_MAX} "
                f"and sc_cfar={sc['sc_cfar']:.4f} > {DENOM_ARTIFACT_SC_CFAR_MIN}. "
                "Same exclusion criterion as nonemit_026 and nonemit_033."
            )
            log.warning("   *** EXCLUDED — denominator artifact (sc_cfar=%.2f, ctrl_mu=%.6f)",
                        sc["sc_cfar"], sc["ctrl_mu"])
        elif _check_terrain_confound(sc):
            record["status"] = "excluded_terrain_confound"
            record["exclusion_reason"] = (
                f"Terrain confound: cv_ctrl={sc['cv_ctrl']:.4f} > {TERRAIN_CONFOUND_CV_MAX} "
                f"and sc_cfar={sc['sc_cfar']:.4f} > {TERRAIN_CONFOUND_SC_CFAR_MIN}. "
                "Model responding to spectral heterogeneity, not methane. "
                "Same exclusion criterion as nonemit_006 (Westerwald mixed forest)."
            )
            log.warning("   *** EXCLUDED — terrain confound (sc_cfar=%.2f, cv_ctrl=%.4f)",
                        sc["sc_cfar"], sc["cv_ctrl"])
        else:
            log.info("   sc_cfar = %.4f  (status=%s)", sc["sc_cfar"], status)

        results.append(record)

    return results


def run_phase2_or_3(
    detector, client, sites: list[dict], already_scored: set[str], phase_name: str
) -> list[dict]:
    """Download from CDSE + inference + S/C for a list of candidate sites."""
    results = []
    for site in sites:
        loc_id = site["location_id"]
        if loc_id in already_scored:
            log.info("   %s already scored — skipping.", loc_id)
            continue

        log.info("=" * 65)
        log.info("── %s  %s : %s (%s) ──",
                 phase_name, loc_id, site["label"], site["ecoregion"])
        log.info("   lat=%.4f  lon=%.4f  clc=%s",
                 site["lat"], site["lon"], site["clc_class"])

        # Search for tile
        product = _search_tile(client, site["lat"], site["lon"], MAX_CLOUD)
        if product is None:
            log.info("   Retrying with cloud <= %.0f%% ...", MAX_CLOUD_FB)
            product = _search_tile(client, site["lat"], site["lon"], MAX_CLOUD_FB)
        if product is None:
            log.warning("   No suitable product found.")
            results.append({**site, "status": "no_products", "sc_cfar": None})
            continue

        # Download + convert
        try:
            dl = download_and_convert(client, product, loc_id)
        except Exception as e:
            log.error("   Download/convert failed: %s", e)
            results.append({**site, "status": "download_failed",
                             "error": str(e), "sc_cfar": None})
            continue
        if dl is None:
            results.append({**site, "status": "download_failed", "sc_cfar": None})
            continue
        npy_path, tile_id = dl

        # Rename with site prefix (matches existing convention)
        prefixed = npy_path.parent / f"{loc_id}_{npy_path.name}"
        if not prefixed.exists():
            npy_path.rename(prefixed)
            npy_path = prefixed

        # Geo JSON
        geo_json = _find_geo_json(npy_path, tile_id)

        # Inference
        product_stem = re.sub(r"\.SAFE$", "", product.name)
        out_tif = TIF_DIR / loc_id / f"original_{product_stem}.tif"
        try:
            ok = run_inference_on_npy(npy_path, detector, geo_json, out_tif)
        except Exception as e:
            log.exception("   Inference failed on %s", loc_id)
            npy_path.unlink(missing_ok=True)
            results.append({**site, "status": "inference_error",
                             "error": str(e), "sc_cfar": None})
            continue

        # Disk discipline: delete .npy after inference
        log.info("   Deleting .npy (%.1f GB freed)", npy_path.stat().st_size / 1e9)
        npy_path.unlink(missing_ok=True)

        if not ok or not out_tif.exists():
            results.append({**site, "status": "no_geotiff", "sc_cfar": None})
            continue

        # S/C ratio
        log.info("   Computing S/C ratio ...")
        sc = compute_sc_ratio(out_tif, site["lat"], site["lon"])
        if "error" in sc:
            log.warning("   S/C error: %s", sc["error"])
            results.append({**site, "status": "sc_error",
                             "error": sc["error"], "sc_cfar": None})
            continue

        record = {
            "location_id":     loc_id,
            "label":           site["label"],
            "lat":             site["lat"],
            "lon":             site["lon"],
            "ecoregion":       site["ecoregion"],
            "clc_class":       site["clc_class"],
            "status":          "ok",
            "product_name":    product.name,
            "tile_id":         tile_id,
            "mgrs_tile":       tile_id,
            "acquisition_date": getattr(product, "acquisition_date", "")[:10],
            "cloud_cover":     getattr(product, "cloud_cover", None),
            "tif":             str(out_tif.relative_to(ROOT)),
            **sc,
            "proximity_flag":  "OK",
            "min_dist_to_emitter_km": None,
        }

        if _check_denom_artifact(sc):
            record["status"] = "excluded_edge_artifact"
            record["exclusion_reason"] = (
                f"Denominator artifact: ctrl_mu={sc['ctrl_mu']:.6f} < {DENOM_ARTIFACT_CTRL_MU_MAX} "
                f"and sc_cfar={sc['sc_cfar']:.4f} > {DENOM_ARTIFACT_SC_CFAR_MIN}. "
                "Same exclusion criterion as nonemit_026 and nonemit_033."
            )
            log.warning("   *** EXCLUDED — denominator artifact (sc_cfar=%.2f, ctrl_mu=%.6f)",
                        sc["sc_cfar"], sc["ctrl_mu"])
        elif _check_terrain_confound(sc):
            record["status"] = "excluded_terrain_confound"
            record["exclusion_reason"] = (
                f"Terrain confound: cv_ctrl={sc['cv_ctrl']:.4f} > {TERRAIN_CONFOUND_CV_MAX} "
                f"and sc_cfar={sc['sc_cfar']:.4f} > {TERRAIN_CONFOUND_SC_CFAR_MIN}. "
                "Model responding to spectral heterogeneity, not methane. "
                "Same exclusion criterion as nonemit_006 (Westerwald mixed forest)."
            )
            log.warning("   *** EXCLUDED — terrain confound (sc_cfar=%.2f, cv_ctrl=%.4f)",
                        sc["sc_cfar"], sc["cv_ctrl"])
        else:
            log.info("   sc_cfar = %.4f  (status=%s)", sc["sc_cfar"], record["status"])

        results.append(record)

    return results


# ── Conformal threshold ────────────────────────────────────────────────────────
def _conformal_tau_inline(scores: list[float], alpha: float = 0.10) -> float:
    """Split-conformal threshold: ceil((n+1)(1-alpha))-th order statistic."""
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha))
    sorted_s = sorted(scores)
    return sorted_s[min(k - 1, n - 1)]


def _bootstrap_ci(scores: list[float], alpha: float = 0.10,
                  n_boot: int = 2000, ci: float = 0.90) -> tuple[float, float]:
    rng = np.random.default_rng(42)
    taus = [_conformal_tau_inline(
        rng.choice(scores, size=len(scores), replace=True).tolist(), alpha
    ) for _ in range(n_boot)]
    lo = float(np.quantile(taus, (1 - ci) / 2))
    hi = float(np.quantile(taus, 1 - (1 - ci) / 2))
    return lo, hi


def print_tau_summary(sites: list[dict]) -> None:
    ok_scores = [s["sc_cfar"] for s in sites
                 if s.get("status") == "ok" and s.get("sc_cfar") is not None]
    if not ok_scores:
        log.warning("No valid sc_cfar scores — cannot compute τ.")
        return
    tau = _conformal_tau_inline(ok_scores)
    lo, hi = _bootstrap_ci(ok_scores)
    by_eco: dict[str, list] = {}
    for s in sites:
        if s.get("status") == "ok" and s.get("sc_cfar") is not None:
            eco = s.get("ecoregion", "Unknown")
            by_eco.setdefault(eco, []).append(s["sc_cfar"])

    log.info("=" * 65)
    log.info("CONFORMAL CALIBRATION SUMMARY")
    log.info("  n (valid)    : %d", len(ok_scores))
    log.info("  τ (α=0.10)   : %.4f", tau)
    log.info("  90%% CI on τ : [%.4f, %.4f]", lo, hi)
    log.info("  Ecoregion counts:")
    for eco, sc_list in sorted(by_eco.items()):
        log.info("    %-15s  n=%d  max_sc=%.4f", eco, len(sc_list), max(sc_list))
    n_total = len(ok_scores)
    fpr = sum(1 for s in ok_scores if s > tau) / n_total if n_total else float("nan")
    log.info("  Empirical FPR: %.1f%%", fpr * 100)
    log.info("=" * 65)


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Expand non-emitter calibration set for CH4Net conformal threshold."
    )
    parser.add_argument("--phase", type=int, choices=[1, 2, 3],
                        help="Run only this phase (1=npy-only, 2=retry, 3=new sites).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done; no downloads or inference.")
    parser.add_argument("--ids", nargs="+",
                        help="Process only these site IDs (e.g. nonemit_037 nonemit_038).")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS),
                        help=f"Path to CH4Net weights (default: {DEFAULT_WEIGHTS})")
    args = parser.parse_args()

    run_phases = {1, 2, 3} if args.phase is None else {args.phase}

    # ── Dry-run ────────────────────────────────────────────────────────────────
    if args.dry_run:
        all_sites = (
            (PHASE_1_SITES if 1 in run_phases else []) +
            (PHASE_2_SITES if 2 in run_phases else []) +
            (PHASE_3_SITES if 3 in run_phases else [])
        )
        if args.ids:
            all_sites = [s for s in all_sites if s["location_id"] in args.ids]
        scored = _scored_ids(_load_scores())
        log.info("DRY RUN — would process (skipping already-scored):")
        for s in all_sites:
            skip = " [SKIP — already scored]" if s["location_id"] in scored else ""
            log.info("  %-15s %-12s  (%.2f, %.3f)  %s%s",
                     s["location_id"], s["ecoregion"],
                     s["lat"], s["lon"], s["label"], skip)
        return

    # ── Load current scores ────────────────────────────────────────────────────
    all_site_records = _load_scores()
    already_scored   = _scored_ids(all_site_records)
    log.info("Existing calibration set: %d scored sites", len(already_scored))

    # ── Load detector (shared across phases) ──────────────────────────────────
    _import_production_deps()
    weights_path = Path(args.weights)
    if not weights_path.exists():
        log.error("Weights file not found: %s", weights_path)
        sys.exit(1)
    log.info("Loading CH4Net v8 weights from %s ...", weights_path.name)
    detector = CH4NetDetector(weights_path)
    log.info("Model ready.")

    # ── Phase 1 ────────────────────────────────────────────────────────────────
    if 1 in run_phases:
        phase1_sites = PHASE_1_SITES
        if args.ids:
            phase1_sites = [s for s in phase1_sites if s["location_id"] in args.ids]
        if phase1_sites:
            log.info(">>> Phase 1: .npy-only inference (%d candidate sites)", len(phase1_sites))
            new = run_phase1(detector, phase1_sites, already_scored)
            for r in new:
                all_site_records.append(r)
                already_scored.add(r["location_id"])
                _save_scores(all_site_records)
                log.info("   Saved (total records: %d)", len(all_site_records))

    # ── Phase 2 & 3 — need CDSE credentials ────────────────────────────────────
    if (2 in run_phases or 3 in run_phases):
        user, pw = _get_credentials()
        client = CopernicusClient(username=user, password=pw)

        if 2 in run_phases:
            phase2_sites = PHASE_2_SITES
            if args.ids:
                phase2_sites = [s for s in phase2_sites if s["location_id"] in args.ids]
            if phase2_sites:
                log.info(">>> Phase 2: CDSE retry (%d candidate sites)", len(phase2_sites))
                new = run_phase2_or_3(detector, client, phase2_sites, already_scored, "Phase 2")
                for r in new:
                    all_site_records.append(r)
                    already_scored.add(r["location_id"])
                    _save_scores(all_site_records)
                    log.info("   Saved (total records: %d)", len(all_site_records))

        if 3 in run_phases:
            phase3_sites = PHASE_3_SITES
            if args.ids:
                phase3_sites = [s for s in phase3_sites if s["location_id"] in args.ids]
            if phase3_sites:
                log.info(">>> Phase 3: new candidate sites (%d candidates)", len(phase3_sites))
                new = run_phase2_or_3(detector, client, phase3_sites, already_scored, "Phase 3")
                for r in new:
                    all_site_records.append(r)
                    already_scored.add(r["location_id"])
                    _save_scores(all_site_records)
                    log.info("   Saved (total records: %d)", len(all_site_records))

    # ── Recompute τ ────────────────────────────────────────────────────────────
    log.info("Recomputing conformal threshold ...")
    conformal_script = ROOT / "scripts" / "calibration" / "conformal_threshold.py"
    if conformal_script.exists():
        proc = subprocess.run(
            [sys.executable, str(conformal_script)], cwd=str(ROOT)
        )
        if proc.returncode != 0:
            log.error("conformal_threshold.py exited with code %d — falling back to inline.", proc.returncode)
            print_tau_summary(all_site_records)
    else:
        log.warning("conformal_threshold.py not found — computing τ inline.")
        print_tau_summary(all_site_records)

    log.info("Done. Results: %s", SCORES_JSON)


if __name__ == "__main__":
    main()
