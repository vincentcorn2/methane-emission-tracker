"""
scripts/sherwin_baseline.py
============================
WS5 Pillar 3 — Sherwin controlled-release baseline scoring.

Runs the CH4Net v8 → S/C → CEMF → IME pipeline on Sentinel-2 overpasses
from the Sherwin/El Abbadi et al. (2023) controlled-release study and scores
the output against the known release rates.

Dataset (Zenodo)
----------------
El Abbadi & Sherwin (2023). CRF22 Controlled-Release Study.
DOI: 10.5281/zenodo.10149991

NOTE: The Zenodo archive contains operator reports and meter data, but NOT
the raw spectral imagery (explicitly withheld by the study authors). The
S2 imagery for the valid overpasses must be downloaded separately from the
Copernicus Data Space (free, no auth for L1C products) using the overpass
datetimes and site coordinates from the Zenodo CSV.

Sentinel-2 arm caveats
-----------------------
Of 23 Sentinel-2 overpasses during the study period, only 6 had actual
controlled releases (the rest were blanks due to a scheduling miscommunication,
per Supplement S2.7). This script processes only the 6 valid overpass rows.

Expected file layout
--------------------
ZENODO_DIR/
  Satellite_overpasses_with_release_rates_20230404.csv   ← key manifest
  01_clean_reports/ ...

S2_TILES_DIR/
  <scene_id>/                 ← unpacked S2 L1C tile (SAFE directory)
    GRANULE/
      L1C_.../
        IMG_DATA/
          *_B01.jp2  ...  *_B12.jp2

Outputs
-------
results_analysis/sherwin_scores.json   — per-scene scoring results
results_analysis/sherwin_rmse.json     — aggregated RMSE / bias / detection
results_analysis/sherwin_rmse_table.md — human-readable table for the report

Usage
-----
    # Step 0 — dry-run: parse the Zenodo CSV, show which scenes are needed
    python scripts/sherwin_baseline.py --zenodo-dir ~/sherwin_zenodo --dry-run

    # Step 1 — download the 6 S2 L1C tiles (requires sentinelsat + credentials)
    python scripts/sherwin_baseline.py --zenodo-dir ~/sherwin_zenodo \\
        --download --copernicus-user <user> --copernicus-pass <pass>

    # Step 2 — run inference on downloaded tiles
    python scripts/sherwin_baseline.py --zenodo-dir ~/sherwin_zenodo \\
        --s2-tiles-dir ~/sherwin_s2_tiles

    # Skip inference if .tif files already exist
    python scripts/sherwin_baseline.py --zenodo-dir ~/sherwin_zenodo \\
        --s2-tiles-dir ~/sherwin_s2_tiles --no-inference
"""

import argparse
import csv
import json
import logging
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Configure logging ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent
RESULTS_DIR  = ROOT / "results_analysis"
TIF_OUT_DIR  = ROOT / "results_sherwin"
SCORES_OUT   = RESULTS_DIR / "sherwin_scores.json"
RMSE_OUT     = RESULTS_DIR / "sherwin_rmse.json"
TABLE_OUT    = RESULTS_DIR / "sherwin_rmse_table.md"

# Zenodo manifest filename (the main per-overpass CSV)
ZENODO_CSV = "Satellite_overpasses_with_release_rates_20230404.csv"

# Wind data subdirectory inside the Zenodo archive
WIND_DATA_SUBDIR = "03_wind_data"

# S/C detection thresholds
SC_TAU_10  = 4.1052   # conformal, α=0.10
SC_TAU_20  = 2.5653   # conformal, α=0.20
SC_PROD    = 1.15     # legacy production threshold

# Wind regime bins (m/s)
WIND_LOW_MAX  = 3.0
WIND_HIGH_MIN = 7.0

# Minimum valid-pixel fraction to attempt inference
MIN_VALID_FRAC = 0.50


# ── Utility ───────────────────────────────────────────────────────────────────
def _classify_wind(speed_ms: Optional[float]) -> str:
    if speed_ms is None:
        return "unknown"
    if speed_ms < WIND_LOW_MAX:
        return "low"
    if speed_ms >= WIND_HIGH_MIN:
        return "high"
    return "moderate"


def _safe_float(val) -> Optional[float]:
    try:
        f = float(val)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


# ── Wind data loader ───────────────────────────────────────────────────────────
def load_wind_for_date(zenodo_dir: Path, date_str: str,
                       timestamp_utc: Optional[str] = None,
                       window_minutes: int = 60) -> dict:
    """
    Load per-day wind CSV from 03_wind_data/<MM_DD>.csv and return
    the mean wind speed and direction over a window centred on the
    overpass timestamp (or the full day if no timestamp given).

    Wind CSVs are named by month_day (e.g., 10_10.csv for Oct 10).
    Expected columns: a datetime/time column + wind_speed + wind_dir.
    Returns dict with keys: wind_speed, wind_dir_deg (both may be None
    if file not found or columns unrecognised).
    """
    import re
    from datetime import datetime, timedelta

    empty = {"wind_speed": None, "wind_dir_deg": None}

    # Parse date to get MM_DD filename
    date_parsed = None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            date_parsed = datetime.strptime(date_str.strip(), fmt)
            break
        except ValueError:
            continue
    if date_parsed is None:
        log.debug("Could not parse date '%s' for wind lookup", date_str)
        return empty

    fname = f"{date_parsed.month:02d}_{date_parsed.day:02d}.csv"
    wind_path = zenodo_dir / WIND_DATA_SUBDIR / fname
    if not wind_path.exists():
        log.debug("Wind file not found: %s", wind_path)
        return empty

    with open(wind_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []

        # Auto-detect columns
        h_low = {c.lower().strip(): c for c in header}
        time_col  = next((h_low[k] for k in h_low
                         if any(x in k for x in ("time", "datetime", "timestamp"))), None)
        spd_col   = next((h_low[k] for k in h_low
                         if any(x in k for x in ("speed", "ws", "wind_s"))), None)
        dir_col   = next((h_low[k] for k in h_low
                         if any(x in k for x in ("dir", "wd", "direction"))), None)

        if spd_col is None:
            log.debug("No wind speed column in %s (cols: %s)", fname, header)
            return empty

        speeds, dirs = [], []
        for row in reader:
            # Time filtering if timestamp provided
            if timestamp_utc and time_col:
                try:
                    row_t_str = row[time_col].strip()
                    # Try to parse as HH:MM or HH:MM:SS
                    if re.match(r'^\d{1,2}:\d{2}', row_t_str):
                        row_t = datetime.strptime(
                            f"{date_parsed.date()} {row_t_str[:8]}", "%Y-%m-%d %H:%M:%S"
                        )
                    else:
                        row_t = datetime.fromisoformat(row_t_str[:19])
                    ovp_t = datetime.fromisoformat(timestamp_utc[:19])
                    if abs((row_t - ovp_t).total_seconds()) > window_minutes * 60:
                        continue
                except Exception:
                    pass  # fallback: include all rows

            spd = _safe_float(row.get(spd_col, ""))
            if spd is not None:
                speeds.append(spd)
            if dir_col:
                d = _safe_float(row.get(dir_col, ""))
                if d is not None:
                    dirs.append(d)

    if not speeds:
        return empty

    mean_speed = float(np.mean(speeds))
    # Circular mean for wind direction
    if dirs:
        sin_mean = np.mean(np.sin(np.radians(dirs)))
        cos_mean = np.mean(np.cos(np.radians(dirs)))
        mean_dir = float(np.degrees(np.arctan2(sin_mean, cos_mean)) % 360)
    else:
        mean_dir = None

    log.debug("Wind %s: n=%d  speed=%.1f m/s  dir=%.0f°",
              fname, len(speeds), mean_speed, mean_dir or -1)
    return {"wind_speed": round(mean_speed, 2), "wind_dir_deg": round(mean_dir, 1) if mean_dir else None}


# ── Parse Zenodo manifest ──────────────────────────────────────────────────────
def load_s2_overpass_rows(zenodo_dir: Path,
                           site_lat: float, site_lon: float) -> list[dict]:
    """
    Parse Satellite_overpasses_with_release_rates_20230404.csv.

    Actual columns in the El Abbadi / Sherwin (2023) Zenodo CSV:
      Team             — operator name (GHGSat, Kayrros, etc.)
      Satellite        — platform (Sentinel-2, WorldView-3, etc.)
      Date             — overpass date (YYYY-MM-DD or similar)
      Timestamp (UTC)  — overpass time
      DateTime (UTC)   — combined
      start_release / end_release — release window timestamps
      gas_kgh_mean     — total gas release rate (kg/hr) from meter
      gas_kgh_sigma    — uncertainty
      ch4_fraction_km  — CH4 mole fraction
      ch4_kgh_mean     — CH4 release rate (kg/hr) — this is Q_known
      ch4_kgh_sigma    — Q_known uncertainty

    Wind comes from separate 03_wind_data/<MM_DD>.csv files.
    Lat/lon is fixed (single release site); passed as arguments.

    Deduplicates by date so each S2 overpass appears only once
    (multiple teams may appear in the same overpass row).
    """
    csv_path = zenodo_dir / ZENODO_CSV
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Zenodo manifest not found: {csv_path}\n"
            f"Download the Zenodo archive (DOI 10.5281/zenodo.10149991) and "
            f"set --zenodo-dir to the extracted root."
        )

    def _pick_col(header: list[str], candidates: list[str]) -> Optional[str]:
        h_lower = {c.lower().strip(): c for c in header}
        for cand in candidates:
            if cand.lower() in h_lower:
                return h_lower[cand.lower()]
        return None

    # Actual column name candidates based on confirmed CSV schema
    _COL_CANDIDATES = {
        "satellite": ["satellite", "platform", "sensor"],
        "date":      ["date", "overpass_date", "acquisition_date"],
        "timestamp": ["datetime (utc)", "timestamp (utc)", "datetime_utc",
                      "timestamp_utc", "datetime", "timestamp"],
        "Q":         ["ch4_kgh_mean", "ch4_kg_h_mean", "ch4_kgh",
                      "gas_kgh_mean", "release_rate_kg_h", "Q_kg_h"],
        "Q_sigma":   ["ch4_kgh_sigma", "gas_kgh_sigma", "Q_sigma"],
        "t_start":   ["start_release", "release_start", "start"],
        "t_end":     ["end_release",   "release_end",   "end"],
    }

    seen_dates: dict[str, dict] = {}  # date → record (dedup by date)

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        log.info("CSV columns: %s", header)

        cols = {k: _pick_col(header, v) for k, v in _COL_CANDIDATES.items()}
        log.info("Column map: %s", {k: v for k, v in cols.items() if v})

        for row in reader:
            # Filter to Sentinel-2
            sat_col = cols.get("satellite")
            if sat_col:
                sat_name = row.get(sat_col, "").strip().lower()
                if "sentinel" not in sat_name and sat_name not in ("s2", "sentinel2"):
                    continue
            # else: no satellite column, process all

            # Parse Q — prefer ch4_kgh_mean, fall back to gas_kgh_mean
            q_col = cols.get("Q")
            Q = _safe_float(row.get(q_col, "")) if q_col else None
            if Q is None or Q <= 0:
                continue  # blank overpass or zero release

            Q_sigma = None
            if cols.get("Q_sigma"):
                Q_sigma = _safe_float(row.get(cols["Q_sigma"], ""))

            date_str = row.get(cols["date"], "").strip() if cols.get("date") else ""
            ts_str   = row.get(cols["timestamp"], "").strip() if cols.get("timestamp") else date_str

            # Deduplicate: keep the first valid row per date
            dedup_key = date_str or ts_str[:10]
            if dedup_key in seen_dates:
                continue

            seen_dates[dedup_key] = {
                "satellite":    row.get(sat_col, "Sentinel-2").strip() if sat_col else "Sentinel-2",
                "date":         date_str,
                "datetime":     ts_str,
                "Q_known_kg_h": round(Q, 3),
                "Q_sigma_kg_h": round(Q_sigma, 3) if Q_sigma else None,
                "lat":          site_lat,
                "lon":          site_lon,
                "scene_id":     "",   # no scene ID in this CSV; set from date
                "_raw":         dict(row),
            }

    rows = list(seen_dates.values())

    # Load wind from 03_wind_data/ for each overpass
    wind_dir_path = zenodo_dir / WIND_DATA_SUBDIR
    has_wind = wind_dir_path.exists()
    if not has_wind:
        log.warning("Wind data directory not found: %s — wind will be None", wind_dir_path)

    for rec in rows:
        if has_wind:
            w = load_wind_for_date(zenodo_dir, rec["date"], rec["datetime"])
        else:
            w = {"wind_speed": None, "wind_dir_deg": None}
        rec["wind_speed"]   = w["wind_speed"]
        rec["wind_dir_deg"] = w["wind_dir_deg"]
        rec["wind_regime"]  = _classify_wind(w["wind_speed"])
        # Use date as scene_id
        rec["scene_id"] = rec["scene_id"] or rec["date"].replace("/", "-").replace(" ", "_")

    log.info("Found %d unique Sentinel-2 overpass dates with valid Q", len(rows))
    if len(rows) > 10:
        log.warning(
            "Expected ~6 valid S2 overpass rows (per paper Supplement S2.7). "
            "Got %d — check satellite column filtering.", len(rows)
        )
    return rows


# ── S2 tile finding ────────────────────────────────────────────────────────────
def find_s2_safe_dir(scene_id: str, s2_tiles_dir: Path) -> Optional[Path]:
    """
    Locate the SAFE directory for a given scene_id under s2_tiles_dir.
    Handles both:
      - <scene_id>.SAFE/
      - <scene_id>/  (unpacked without .SAFE extension)
      - Fuzzy match on the first 40 chars of scene_id (product ID stem)
    """
    if not s2_tiles_dir or not s2_tiles_dir.exists():
        return None
    stem = scene_id[:40] if len(scene_id) > 40 else scene_id
    for candidate in s2_tiles_dir.iterdir():
        if stem in candidate.name:
            return candidate
    return None


def s2_safe_to_npy(safe_dir: Path, lat: float, lon: float,
                   npy_out: Path, crop_km: float = 20.0) -> Path:
    """
    Convert a Sentinel-2 SAFE directory to a cropped NPY array centred on
    (lat, lon). Crops a square of crop_km × crop_km.

    Requires: rasterio, pyproj, numpy, glymur (for .jp2) or GDAL.
    Returns path to the written NPY file.
    """
    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.crs import CRS
    from pyproj import Transformer

    S2_BANDS = [
        "B01", "B02", "B03", "B04", "B05", "B06",
        "B07", "B08", "B8A", "B09", "B11", "B12",
    ]

    # Find band files
    band_paths = {}
    for jp2 in safe_dir.rglob("*.jp2"):
        for b in S2_BANDS:
            if jp2.stem.endswith(f"_{b}"):
                band_paths[b] = jp2
                break

    missing = [b for b in S2_BANDS if b not in band_paths]
    if missing:
        raise FileNotFoundError(f"Missing S2 bands in {safe_dir}: {missing}")

    # Use B02 (10m) as reference grid
    ref_path = band_paths["B02"]
    arrays = {}
    with rasterio.open(ref_path) as ref:
        epsg = ref.crs.to_epsg()
        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}",
                                           always_xy=True)
        cx, cy = transformer.transform(lon, lat)
        half_m = crop_km * 500  # half crop in metres
        crop_bounds = (cx - half_m, cy - half_m, cx + half_m, cy + half_m)
        ref_w = from_bounds(*crop_bounds, transform=ref.transform)
        H = int(ref_w.height)
        W = int(ref_w.width)

    # Read each band, resample to 10m grid
    cube = np.zeros((H, W, 12), dtype=np.uint8)
    for i, b in enumerate(S2_BANDS):
        with rasterio.open(band_paths[b]) as src:
            w = from_bounds(*crop_bounds, transform=src.transform)
            data = src.read(
                1,
                window=w,
                out_shape=(H, W),
                resampling=rasterio.enums.Resampling.bilinear,
            )
            # Normalise to uint8 (S2 L1C is 12-bit; scale to 0–255)
            data = np.clip(data / 16, 0, 255).astype(np.uint8)
            cube[:, :, i] = data

    npy_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_out, cube)
    log.info("  Saved NPY %s  shape=%s", npy_out.name, cube.shape)
    return npy_out


# ── Coverage check ─────────────────────────────────────────────────────────────
def check_site_coverage(npy_path: Path, lat: float, lon: float,
                        tif_path: Optional[Path] = None) -> float:
    """
    Return fraction of non-zero pixels in the 100×100 crop centred on
    (lat, lon) within the NPY array.
    """
    import rasterio
    from pyproj import Transformer

    arr = np.load(npy_path)
    H, W = arr.shape[:2]
    half = 50

    if tif_path and tif_path.exists():
        with rasterio.open(tif_path) as src:
            epsg = src.crs.to_epsg()
            tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
            xs, ys = tr.transform([lon], [lat])
            row, col = rasterio.transform.rowcol(src.transform, xs[0], ys[0])
    else:
        # Fallback: assume centre of array
        row, col = H // 2, W // 2

    r0 = max(0, row - half)
    r1 = min(H, row + half)
    c0 = max(0, col - half)
    c1 = min(W, col + half)
    patch = arr[r0:r1, c0:c1, :]
    if patch.size == 0:
        return 0.0
    return float(patch.any(axis=-1).mean())


# ── Inference & scoring ────────────────────────────────────────────────────────
def _load_detector():
    from apply_bitemporal_diff import CH4NetDetector, WEIGHTS, THRESHOLD
    log.info("Loading CH4Net v8 weights from %s", WEIGHTS)
    return CH4NetDetector(WEIGHTS, THRESHOLD)


def _compute_sc(tif_path: Path, lat: float, lon: float) -> dict:
    from apply_bitemporal_diff import compute_sc_ratio
    return compute_sc_ratio(tif_path, lat, lon)


def _run_cemf_ime(tif_path: Path, lat: float, lon: float,
                  wind_speed: float, wind_dir_deg: float,
                  acquisition_ts: str) -> dict:
    try:
        import rasterio as rio
        from src.quantification.cemf import run_cemf
        from src.quantification.ime import CEMFIntegratedMassEnhancement

        with rio.open(tif_path) as src:
            prob = src.read(1).astype(np.float32)
            transform = src.transform
            crs = src.crs

        cemf_result = run_cemf(
            prob_map=prob, transform=transform, crs=crs,
            site_lat=lat, site_lon=lon,
        )
        ime = CEMFIntegratedMassEnhancement()
        Q_result = ime.estimate_from_cemf(
            cemf_result=cemf_result,
            wind_speed_ms=wind_speed,
            wind_dir_deg=wind_dir_deg,
            acquisition_timestamp=acquisition_ts,
        )
        return {
            "Q_hat_kg_h":          round(Q_result.Q_kg_h, 3),
            "Q_hat_lo95":          round(Q_result.Q_lo95, 3),
            "Q_hat_hi95":          round(Q_result.Q_hi95, 3),
            "Q_hat_lo80":          round(Q_result.Q_lo80, 3),
            "Q_hat_hi80":          round(Q_result.Q_hi80, 3),
            "cemf_enhancement_kg": round(cemf_result.total_enhancement_kg, 6),
            "cemf_plume_pixels":   cemf_result.plume_pixels,
            "ime_wind_speed":      round(wind_speed, 2),
            "ime_wind_dir":        round(wind_dir_deg, 1),
            "quantification_ok":   True,
        }
    except Exception as e:
        log.warning("CEMF/IME failed: %s", e)
        return {"quantification_ok": False, "quantification_error": str(e)}


def score_scene(record: dict, s2_tiles_dir: Optional[Path],
                npy_cache: Path, tif_out_dir: Path,
                detector, no_inference: bool) -> dict:
    """
    Full pipeline for a single overpass row.
    Returns a result dict to be appended to sherwin_scores.json.
    """
    scene_id  = record["scene_id"] or record["datetime"].replace(":", "-").replace(" ", "_")
    lat       = record["lat"]
    lon       = record["lon"]
    Q_known   = record["Q_known_kg_h"]
    wind_spd  = record["wind_speed"]
    wind_dir  = record["wind_dir_deg"]
    dt_str    = record["datetime"]

    result = {
        "scene_id":      scene_id,
        "datetime":      dt_str,
        "lat":           lat,
        "lon":           lon,
        "Q_known_kg_h":  Q_known,
        "wind_speed":    wind_spd,
        "wind_dir_deg":  wind_dir,
        "wind_regime":   record["wind_regime"],
        "status":        "pending",
    }

    # ── locate or build NPY ──────────────────────────────────────────────
    npy_path = npy_cache / f"{scene_id}.npy"
    tif_path = tif_out_dir / f"{scene_id}_prob.tif"

    if not npy_path.exists():
        # Try to build NPY from SAFE dir
        if s2_tiles_dir:
            safe_dir = find_s2_safe_dir(scene_id, s2_tiles_dir)
            if safe_dir:
                log.info("  Converting SAFE → NPY for %s", scene_id)
                try:
                    s2_safe_to_npy(safe_dir, lat, lon, npy_path)
                except Exception as e:
                    log.error("  SAFE→NPY failed: %s", e)
                    result["status"] = "error"
                    result["error"] = f"safe_to_npy: {e}"
                    return result
            else:
                log.warning("  SAFE dir not found for %s in %s", scene_id, s2_tiles_dir)
        if not npy_path.exists():
            result["status"] = "missing_npy"
            result["note"] = (
                f"No NPY at {npy_path} and no SAFE directory found. "
                "Download the S2 L1C tile from Copernicus Data Space: "
                f"search for overpass near {lat:.3f},{lon:.3f} on {dt_str[:10]}"
            )
            return result

    # ── coverage check ──────────────────────────────────────────────────
    vf = check_site_coverage(npy_path, lat, lon)
    result["valid_fraction"] = round(vf, 4)
    if vf < MIN_VALID_FRAC:
        result["status"] = "no_coverage"
        result["coverage_note"] = (
            f"Site crop only {vf*100:.0f}% valid — partial-swath tile"
        )
        return result

    # ── inference ───────────────────────────────────────────────────────
    if not no_inference:
        if not tif_path.exists():
            if detector is None:
                log.warning("  Detector not loaded; skipping inference for %s", scene_id)
                result["status"] = "no_inference"
                return result
            try:
                from apply_bitemporal_diff import run_inference
                arr = np.load(npy_path)
                run_inference(arr, detector, {}, tif_path)
                del arr
            except Exception as e:
                log.error("  Inference failed for %s: %s", scene_id, e)
                result["status"] = "inference_error"
                result["error"] = str(e)
                return result

    if not tif_path.exists():
        result["status"] = "no_tif"
        return result

    # ── S/C ratio ───────────────────────────────────────────────────────
    try:
        sc = _compute_sc(tif_path, lat, lon)
        result.update({k: round(v, 6) if isinstance(v, float) else v
                       for k, v in sc.items()})
        sc_ratio = sc.get("sc_ratio", 0.0)
    except Exception as e:
        log.warning("  SC failed: %s", e)
        sc_ratio = None
        result["sc_error"] = str(e)

    # ── detection flags ─────────────────────────────────────────────────
    result["detected_tau10"] = bool(sc_ratio is not None and sc_ratio >= SC_TAU_10)
    result["detected_tau20"] = bool(sc_ratio is not None and sc_ratio >= SC_TAU_20)
    result["detected_prod"]  = bool(sc_ratio is not None and sc_ratio >= SC_PROD)

    # ── CEMF/IME ────────────────────────────────────────────────────────
    wind_for_ime = wind_spd if wind_spd is not None else 3.5
    wind_dir_for_ime = wind_dir if wind_dir is not None else 270.0

    if result["detected_prod"]:
        qr = _run_cemf_ime(tif_path, lat, lon,
                           wind_for_ime, wind_dir_for_ime, dt_str)
        result.update(qr)

        # CI coverage flags
        if qr.get("quantification_ok") and Q_known is not None:
            result["in_ci80"] = (qr["Q_hat_lo80"] <= Q_known <= qr["Q_hat_hi80"])
            result["in_ci95"] = (qr["Q_hat_lo95"] <= Q_known <= qr["Q_hat_hi95"])
            result["abs_error_kg_h"] = round(abs(qr["Q_hat_kg_h"] - Q_known), 3)
            result["rel_error"]      = round((qr["Q_hat_kg_h"] - Q_known) / Q_known, 4)

    result["status"] = "ok"
    return result


# ── Aggregation ────────────────────────────────────────────────────────────────
def compute_rmse_table(scores: list[dict]) -> dict:
    def _stats(subset):
        ok = [s for s in subset
              if s.get("status") == "ok"
              and s.get("quantification_ok")
              and s.get("Q_known_kg_h") is not None
              and s.get("Q_hat_kg_h") is not None]
        if not ok:
            return None
        errs   = [s["Q_hat_kg_h"] - s["Q_known_kg_h"] for s in ok]
        abs_e  = [abs(e) for e in errs]
        n_det  = sum(1 for s in subset if s.get("detected_tau10"))
        n_all  = len(subset)
        n_ci80 = sum(1 for s in ok if s.get("in_ci80"))
        n_ci95 = sum(1 for s in ok if s.get("in_ci95"))
        return {
            "n_scenes":        n_all,
            "n_quantified":    len(ok),
            "n_detected_tau10": n_det,
            "detection_rate_tau10": round(n_det / n_all, 3) if n_all else None,
            "rmse_kg_h":       round(math.sqrt(sum(e**2 for e in errs) / len(errs)), 3),
            "bias_kg_h":       round(sum(errs) / len(errs), 3),
            "mae_kg_h":        round(sum(abs_e) / len(abs_e), 3),
            "ci80_coverage":   round(n_ci80 / len(ok), 3) if ok else None,
            "ci95_coverage":   round(n_ci95 / len(ok), 3) if ok else None,
        }

    out = {"overall": _stats(scores)}

    for regime in ("low", "moderate", "high"):
        sub = [s for s in scores if s.get("wind_regime") == regime]
        out[f"wind_{regime}"] = _stats(sub) if sub else None

    return out


def render_markdown_table(rmse: dict) -> str:
    lines = [
        "## Sherwin Controlled-Release Benchmarking — CH4Net v8\n",
        "| Stratum | n | n_detected | Detect Rate | RMSE (kg/hr) | Bias (kg/hr) | MAE (kg/hr) | 80% CI cov. | 95% CI cov. |",
        "|---------|---|-----------|-------------|--------------|--------------|-------------|-------------|-------------|",
    ]
    order = ["overall", "wind_low", "wind_moderate", "wind_high"]
    labels = {
        "overall":       "All scenes",
        "wind_low":      "Wind < 3 m/s",
        "wind_moderate": "Wind 3–7 m/s",
        "wind_high":     "Wind > 7 m/s",
    }
    for key in order:
        s = rmse.get(key)
        if not s:
            continue
        row = (
            f"| {labels[key]} "
            f"| {s['n_scenes']} "
            f"| {s['n_detected_tau10']} "
            f"| {s['detection_rate_tau10']:.0%} "
            f"| {s['rmse_kg_h']} "
            f"| {s['bias_kg_h']:+.1f} "
            f"| {s['mae_kg_h']} "
            f"| {s['ci80_coverage']:.0%} "
            f"| {s['ci95_coverage']:.0%} |"
        )
        lines.append(row)
    lines.append(
        "\n_Detection threshold: τ = 4.1052 (α = 0.10 conformal). "
        "CI from CEMF+IME heteroscedastic envelope._\n"
    )
    return "\n".join(lines)


# ── Copernicus download helper ─────────────────────────────────────────────────
def download_s2_tiles(overpass_rows: list[dict], out_dir: Path,
                      user: str, password: str) -> None:
    """
    Download S2 L1C tiles for each overpass using sentinelsat.
    Requires: pip install sentinelsat

    Each tile is searched by:
      - platformname = Sentinel-2
      - producttype  = S2MSI1C
      - date window  = ±1 day around overpass datetime
      - footprint    = small box around (lat, lon)
    """
    try:
        from sentinelsat import SentinelAPI, geojson_to_wkt
        from datetime import datetime, timedelta
        from shapely.geometry import Point
    except ImportError:
        log.error("sentinelsat not installed. Run: pip install sentinelsat shapely")
        sys.exit(1)

    api = SentinelAPI(user, password,
                      "https://apihub.copernicus.eu/apihub")
    out_dir.mkdir(parents=True, exist_ok=True)

    for rec in overpass_rows:
        lat, lon = rec["lat"], rec["lon"]
        dt_raw   = rec["datetime"]
        try:
            dt = datetime.fromisoformat(dt_raw[:19])
        except ValueError:
            log.warning("Cannot parse datetime '%s' for %s", dt_raw, rec["scene_id"])
            continue

        footprint = geojson_to_wkt(
            Point(lon, lat).buffer(0.05).__geo_interface__
        )
        date_from = (dt - timedelta(days=1)).strftime("%Y%m%d")
        date_to   = (dt + timedelta(days=1)).strftime("%Y%m%d")

        log.info("Searching S2 tile: %s (%s)", rec["scene_id"] or dt_raw, date_from)
        products = api.query(
            footprint,
            date=(date_from, date_to),
            platformname="Sentinel-2",
            producttype="S2MSI1C",
            cloudcoverpercentage=(0, 50),
        )
        if not products:
            log.warning("  No products found for %s", dt_raw)
            continue
        # Download first result
        pid = next(iter(products))
        log.info("  Downloading %s", products[pid]["title"])
        api.download(pid, directory_path=str(out_dir))

    log.info("Download complete. Tiles in %s", out_dir)


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sherwin controlled-release baseline scoring for CH4Net v8"
    )
    parser.add_argument("--zenodo-dir", required=True, type=Path,
                        help="Path to the extracted Zenodo archive root "
                             "(must contain Satellite_overpasses_with_release_rates_*.csv)")
    parser.add_argument("--s2-tiles-dir", type=Path, default=None,
                        help="Directory containing downloaded S2 SAFE dirs or extracted tiles")
    parser.add_argument("--npy-cache", type=Path, default=None,
                        help="Override default NPY cache directory "
                             "(default: data/npy_cache_sherwin/)")
    parser.add_argument("--site-lat", type=float, default=35.299,
                        help="Release site latitude (default: 35.299 — North Coles Levee Oil Field, Kern County CA)")
    parser.add_argument("--site-lon", type=float, default=-119.285,
                        help="Release site longitude (default: -119.285 — North Coles Levee Oil Field, Kern County CA)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse manifest and show required scenes without running inference")
    parser.add_argument("--no-inference", action="store_true",
                        help="Skip inference; use existing .tif files only")
    parser.add_argument("--download", action="store_true",
                        help="Auto-download S2 tiles from Copernicus (requires --copernicus-user/pass)")
    parser.add_argument("--copernicus-user", default=None)
    parser.add_argument("--copernicus-pass", default=None)
    args = parser.parse_args()

    npy_cache   = args.npy_cache or ROOT / "data" / "npy_cache_sherwin"
    tif_out_dir = TIF_OUT_DIR
    tif_out_dir.mkdir(parents=True, exist_ok=True)
    npy_cache.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load manifest ───────────────────────────────────────────────────
    overpass_rows = load_s2_overpass_rows(args.zenodo_dir, args.site_lat, args.site_lon)
    log.info("\n=== Sherwin S2 overpass rows with valid Q ===")
    for i, r in enumerate(overpass_rows):
        log.info("  [%d] %s  lat=%.3f lon=%.3f  Q=%.1f kg/hr  wind=%.1f m/s %s",
                 i+1, r["datetime"], r["lat"] or 0, r["lon"] or 0,
                 r["Q_known_kg_h"], r["wind_speed"] or 0, r["wind_regime"])

    if args.dry_run:
        log.info("\nDRY RUN — no inference or downloads performed.")
        log.info("To proceed:")
        log.info("  1. Download the S2 tiles for the %d overpasses above from:", len(overpass_rows))
        log.info("     https://dataspace.copernicus.eu  (search by date + coordinates)")
        log.info("  2. Re-run with --s2-tiles-dir /path/to/tiles")
        return

    # ── Optional auto-download ──────────────────────────────────────────
    if args.download:
        if not args.copernicus_user or not args.copernicus_pass:
            log.error("--download requires --copernicus-user and --copernicus-pass")
            sys.exit(1)
        download_s2_tiles(overpass_rows, args.s2_tiles_dir or ROOT / "data" / "sherwin_s2_tiles",
                          args.copernicus_user, args.copernicus_pass)
        if not args.s2_tiles_dir:
            args.s2_tiles_dir = ROOT / "data" / "sherwin_s2_tiles"

    # ── Load detector ───────────────────────────────────────────────────
    detector = None
    if not args.no_inference:
        try:
            detector = _load_detector()
        except Exception as e:
            log.warning("Could not load detector: %s — using --no-inference mode", e)

    # ── Score each scene ────────────────────────────────────────────────
    all_scores = []
    for rec in overpass_rows:
        scene_id = rec["scene_id"] or rec["datetime"][:10]
        log.info("\n── Processing %s ──────────────────────────", scene_id)
        result = score_scene(
            rec, args.s2_tiles_dir, npy_cache, tif_out_dir,
            detector, args.no_inference,
        )
        all_scores.append(result)
        log.info("  status=%s  sc_ratio=%s  Q_hat=%s  Q_known=%s",
                 result.get("status"),
                 result.get("sc_ratio"),
                 result.get("Q_hat_kg_h"),
                 result.get("Q_known_kg_h"))

    # ── Write outputs ───────────────────────────────────────────────────
    with open(SCORES_OUT, "w") as f:
        json.dump(all_scores, f, indent=2)
    log.info("\nScores written to %s", SCORES_OUT)

    rmse = compute_rmse_table(all_scores)
    with open(RMSE_OUT, "w") as f:
        json.dump(rmse, f, indent=2)

    md = render_markdown_table(rmse)
    with open(TABLE_OUT, "w") as f:
        f.write(md)

    log.info("RMSE table written to %s", TABLE_OUT)
    print("\n" + md)

    # ── Summary ─────────────────────────────────────────────────────────
    n_ok        = sum(1 for s in all_scores if s.get("status") == "ok")
    n_no_cov    = sum(1 for s in all_scores if s.get("status") == "no_coverage")
    n_miss      = sum(1 for s in all_scores if s.get("status") == "missing_npy")
    print(f"\n=== Run summary ===")
    print(f"  Scenes total    : {len(all_scores)}")
    print(f"  Scored OK       : {n_ok}")
    print(f"  No coverage     : {n_no_cov}")
    print(f"  Missing tiles   : {n_miss}")
    if n_miss:
        print(f"\n  Missing tile datetimes:")
        for s in all_scores:
            if s.get("status") == "missing_npy":
                print(f"    {s['datetime']}  lat={s['lat']}  lon={s['lon']}")
        print(f"\n  Download these from: https://dataspace.copernicus.eu")
        print(f"  Then re-run with --s2-tiles-dir /path/to/downloaded_tiles")


if __name__ == "__main__":
    main()
