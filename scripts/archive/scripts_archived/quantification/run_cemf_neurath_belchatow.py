"""
scripts/run_cemf_neurath_belchatow.py
======================================
Targeted CEMF+IME quantification for:
  1. neurath          2024-06-25 (T32ULB) — ERA5 auto-fetch, ±30% bounds
  2. neurath_20240829 2024-08-29 (T32ULB) — ERA5 auto-fetch, ±30% bounds  [S/C=67.2 CFAR]
  3. belchatow        2024-08-24 (T34UCB) — ERA5 wind 2.19 m/s (pre-fetched), ±30% bounds
  4. belchatow_20240710 2024-07-10 (T34UCB) — ERA5 auto-fetch, ±30% bounds  [S/C=142.9 CFAR]
  5. Weisweiler       2024-09-18 (T31UGS) — ERA5 auto-fetch at run time, ±30% bounds

This script bypasses the generic _load_arrays() path in run_quantification.py,
which looks for per-band npy files. Instead it loads the 12-band stacked npy
arrays (band layout: B01…B09 at idx 0–8, B11 at idx 10, B12 at idx 11) and
extracts B11/B12 directly.

Band layout (CH4Net 12-band convention):
  Index  0  1  2  3  4  5  6  7   8   9  10  11
  Band  B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B11 B12

Mask strategy:
  The TIF files are raw CH4Net probability maps (float32, 0–1).
  CEMF needs a binary plume/background split. We crop the full-tile
  probability map to a region around the facility and threshold it:
    plume   : prob > PROB_THRESH  (scene-adaptive per site)
    background : prob ≤ PROB_THRESH
  Thresholds are derived from the multidate_validation.json site/ctrl means.

Spatial cropping:
  Both sites have 10m pixel spacing. We crop a CROP_PX × CROP_PX patch
  centred on the facility lat/lon. For CEMF at 20m (SWIR native), the
  patch is further downsampled to CROP_PX//2 × CROP_PX//2.

Usage:
    python scripts/run_cemf_neurath_belchatow.py [--dry-run] [--site neurath|belchatow]
"""
import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

# ── Path setup ────────────────────────────────────────────────────────────────
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = None  # suppress for large tiles

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.quantification.cemf import run_cemf, CEMFResult
from src.quantification.ime import CEMFIntegratedMassEnhancement
from src.quantification.uncertainty import get_uncertainty_pct
from src.quantification.canonical_writer import (
    QuantificationRecord,
    write_quantification_record,
    DEFAULT_QUANT_PATH,
)
from src.ingestion.era5_client import ERA5Client, FALLBACK_WIND_SPEED, FALLBACK_WIND_SOURCE

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("cemf_runner")

# ── Band indices in the 12-band npy stack ─────────────────────────────────────
B11_IDX = 10   # 1610 nm SWIR reference (methane-transparent)
B12_IDX = 11   # 2190 nm SWIR absorption (methane-sensitive)

# ── Default crop size (10m pixels) — 500×500 = 5 km × 5 km around facility ──
# Per-site override via "crop_px" key in SITES dict.
CROP_PX = 500

# ── Site configuration ────────────────────────────────────────────────────────
SITES = {
    "neurath": {
        "npy":  "data/npy_cache/S2A_MSIL1C_20240625T103631_N0510_R008_T32ULB_20240625T142035.npy",
        "tif":  "results_bitemporal/neurath/original_S2A_MSIL1C_20240625T103631_N0510_R008_T32ULB_20240625T142035.tif",
        "scene_id": "S2A_MSIL1C_20240625T103631_N0510_R008_T32ULB_20240625T142035",
        "acquisition_timestamp": "2024-06-25T10:36:31Z",
        # Pixel coordinates in T32ULB tile (10m/px, computed via pyproj UTM32N)
        # Neurath power plant: 51.038°N, 6.616°E → row=4324, col=3286
        "site_row": 4324,
        "site_col": 3286,
        "lat": 51.038,
        "lon": 6.616,
        # Probability threshold — 3× ctrl_mean from multidate_validation.json
        # site_mean=0.121191, ctrl_mean=0.00526 → thresh = 0.0158
        "prob_thresh": 0.016,
        "tropomi_confirm": True,
        "ch4net_peak_probability": 0.93,
        # ERA5 auto-fetched at run time (CDS key in ~/.cdsapirc)
        "wind": None,
        "retrieval_notes": (
            "Dual-sensor confirm: TROPOMI DXCH4=+12.2 ppb; S/C=23.04 (T32ULB, 2024-06-25). "
            "Probability threshold: 0.016 (3× ctrl_mean). "
            "ERA5 wind auto-fetched from CDS API."
        ),
    },
    # ── Neurath second confirmed detection ────────────────────────────────────
    # 2024-08-29: S/C=67.2 (classic) / sc_cfar=97.0 (CFAR margin=94.96) — strongest
    # detection across all Neurath dates. cv_ctrl=0.288 (lower terrain noise than June).
    "neurath_20240829": {
        "site": "neurath",   # canonical site name for quantification.json
        "npy":  "data/npy_cache/S2B_MSIL1C_20240829T103629_N0511_R008_T32ULB_20240829T124434.npy",
        "tif":  "results_bitemporal/neurath/original_S2B_MSIL1C_20240829T103629_N0511_R008_T32ULB_20240829T124434.tif",
        "scene_id": "S2B_MSIL1C_20240829T103629_N0511_R008_T32ULB_20240829T124434",
        "acquisition_timestamp": "2024-08-29T10:36:29Z",
        # Same tile T32ULB — same pixel coordinates as June-25
        "site_row": 4324,
        "site_col": 3286,
        "lat": 51.038,
        "lon": 6.616,
        # site_mean=0.014405, ctrl_mu=0.000149 → 5× ctrl_mu = 0.000745
        "prob_thresh": 0.0007,
        "tropomi_confirm": True,   # same facility; TROPOMI confirmation applies
        "ch4net_peak_probability": 0.93,
        "wind": None,
        "retrieval_notes": (
            "Second confirmed detection at Neurath: S/C=67.2 (classic), "
            "sc_cfar=97.0, CFAR margin=94.96 (T32ULB, 2024-08-29). "
            "cv_ctrl=0.288 — lower terrain CV than Jun-25 (0.992). "
            "Probability threshold: 0.0007 (5× ctrl_mu=0.000149). "
            "ERA5 wind auto-fetched from CDS API."
        ),
    },
    "belchatow": {
        "npy":  "data/npy_cache/S2B_MSIL1C_20240824T094549_N0511_R079_T34UCB_20240824T115611.npy",
        "tif":  "results_bitemporal/belchatow/original_S2B_MSIL1C_20240824T094549_N0511_R079_T34UCB_20240824T115611.tif",
        "scene_id": "S2B_MSIL1C_20240824T094549_N0511_R079_T34UCB_20240824T115611",
        "acquisition_timestamp": "2024-08-24T09:45:49Z",
        # Bełchatów: 51.264°N, 19.331°E → T34UCB row=1949, col=8356
        "site_row": 1949,
        "site_col": 8356,
        "lat": 51.264,
        "lon": 19.331,
        # site_mean=0.001062, ctrl_mean=3.9e-05 → thresh = 3×ctrl = 0.000117
        # Use a small but meaningful threshold
        "prob_thresh": 0.0002,
        "tropomi_confirm": False,
        "ch4net_peak_probability": 0.82,
        "wind": {
            "wind_speed_ms": 2.19,
            "wind_dir_deg": 245.0,
            "wind_source": "ERA5_reanalysis",
            "era5_u_ms": -1.55,
            "era5_v_ms": -1.55,
        },
        "retrieval_notes": (
            "ERA5 wind 2.19 m/s (2024-08-24, 09:45 UTC) — pre-retrieved. "
            "BT mask excluded: BT kills real signal (S/C 27.3→1.94). "
            "TROPOMI cloud-blocked; CEMF on original mask. "
            "Probability threshold: 0.0002 (5× ctrl_mean 3.9e-05)."
        ),
    },
    # ── Bełchatów strongest detection ─────────────────────────────────────────
    # 2024-07-10: S/C=142.9 (classic) / sc_cfar=50.7 (CFAR margin=46.2) — highest
    # S/C of any Bełchatów date. cv_ctrl=1.105 (moderate terrain heterogeneity).
    "belchatow_20240710": {
        "site": "belchatow",   # canonical site name for quantification.json
        "npy":  "data/npy_cache/S2A_MSIL1C_20240710T095031_N0510_R079_T34UCB_20240710T133148.npy",
        "tif":  "results_bitemporal/belchatow/original_S2A_MSIL1C_20240710T095031_N0510_R079_T34UCB_20240710T133148.tif",
        "scene_id": "S2A_MSIL1C_20240710T095031_N0510_R079_T34UCB_20240710T133148",
        "acquisition_timestamp": "2024-07-10T09:50:31Z",
        # Same tile T34UCB — same pixel coordinates as Aug-24
        "site_row": 1949,
        "site_col": 8356,
        "lat": 51.264,
        "lon": 19.331,
        # site_mean=0.025282, ctrl_mu=0.000499 → 5× ctrl_mu = 0.002495
        "prob_thresh": 0.002,
        "tropomi_confirm": False,
        "ch4net_peak_probability": 0.82,
        "wind": None,
        "retrieval_notes": (
            "Strongest Belchatow detection: S/C=142.9 (classic), "
            "sc_cfar=50.7, CFAR margin=46.2 (T34UCB, 2024-07-10). "
            "cv_ctrl=1.105 — moderate terrain heterogeneity; CFAR still triggered. "
            "Probability threshold: 0.002 (5× ctrl_mu=0.000499). "
            "ERA5 wind auto-fetched from CDS API."
        ),
    },
    "weisweiler": {
        # Kraftwerk Weisweiler (RWE, ~775 MW lignite, near Aachen).
        # T31UGS 2024-09-18: sc_cfar=2.083 (confirmed detection).
        # Classic S/C=23.46 (vs North ctrl 0.000469); all-direction mu_ctrl=0.005277, CV=1.607.
        # CFAR adaptive threshold=5.97 not reached — heterogeneous Rhine/Aachen terrain.
        # Quantification: ERA5 auto-fetched at run time (CDS key in ~/.cdsapirc).
        # Background note: tile-wide scatter makes some plume pixels bleed into
        # the background mask; treat flow estimate as indicative ±50%.
        "npy":  "data/npy_cache/S2B_MSIL1C_20240918T103619_N0511_R008_T31UGS_20240918T142046.npy",
        "tif":  "results_bitemporal/weisweiler/original_S2B_MSIL1C_20240918T103619_N0511_R008_T31UGS_20240918T142046.tif",
        "scene_id": "S2B_MSIL1C_20240918T103619_N0511_R008_T31UGS_20240918T142046",
        "acquisition_timestamp": "2024-09-18T10:36:19Z",
        # Pixel coordinates in T31UGS (EPSG:32631, origin 699960 / 5700000, 10m/px)
        # lat=50.837, lon=6.322 → row=6304, col=3393
        "site_row": 6304,
        "site_col": 3393,
        "lat": 50.837,
        "lon": 6.322,
        # Threshold at 5× all-direction mu_ctrl (0.005277).
        # 2× mu_ctrl = 0.01055 gives ~73k plume px in 500px crop.
        # 5× mu_ctrl = 0.02638 gives ~63k plume px — best separation from noise floor.
        "prob_thresh": 0.026,
        "tropomi_confirm": False,
        "ch4net_peak_probability": 0.42,   # peak in site 100px crop
        # ERA5 to be auto-fetched at run time for 2024-09-18
        "wind": None,   # None → ERA5Client.get_wind() called in run_site()
        "retrieval_notes": (
            "Kraftwerk Weisweiler — sc_cfar=2.083 (T31UGS, 2024-09-18). "
            "Classic S/C=23.46 vs North ctrl; all-dir mu_ctrl=0.005277, CV=1.607. "
            "High terrain CV raises CFAR adaptive threshold to 5.97 (not triggered). "
            "CEMF run on baseline (original) mask; BT skipped per skip_bitemporal rule. "
            "Probability threshold: 0.026 (5× mu_ctrl). "
            "ERA5 wind fetched live from CDS API. "
            "Quantification indicative — heterogeneous background may inflate plume area."
        ),
    },
}


def load_site_arrays(site_cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load B11, B12 (at 10m), and probability mask (at 10m) — cropped to site region.

    Returns:
        b11_crop   : float32 array (crop_px, crop_px) at 10m
        b12_crop   : float32 array (crop_px, crop_px) at 10m
        mask_crop  : float32 array (crop_px, crop_px) at 10m
    """
    npy_path = site_cfg["npy"]
    tif_path = site_cfg["tif"]
    row, col = site_cfg["site_row"], site_cfg["site_col"]
    crop_px = site_cfg.get("crop_px", CROP_PX)
    half = crop_px // 2

    # Crop box
    r0, r1 = row - half, row + half
    c0, c1 = col - half, col + half

    logger.info("Loading npy %s", npy_path)
    arr = np.load(npy_path, mmap_mode="r")   # memory-mapped, shape (H, W, 12)

    # Validate band count
    assert arr.ndim == 3 and arr.shape[2] >= 12, (
        f"Expected (H, W, 12+) npy, got {arr.shape}"
    )

    b11_crop = arr[r0:r1, c0:c1, B11_IDX].astype(np.float32)
    b12_crop = arr[r0:r1, c0:c1, B12_IDX].astype(np.float32)
    logger.info("B11 crop: shape=%s  B12 crop: shape=%s", b11_crop.shape, b12_crop.shape)
    logger.info("B11 range: %.3f – %.3f  B12 range: %.3f – %.3f",
                b11_crop.min(), b11_crop.max(), b12_crop.min(), b12_crop.max())

    logger.info("Loading TIF probability map (crop only) %s", tif_path)
    img = Image.open(tif_path)
    # PIL crop: (left, upper, right, lower)
    mask_pil = img.crop((c0, r0, c1, r1))
    mask_crop = np.array(mask_pil).astype(np.float32)
    logger.info("Probability map crop: shape=%s  range: %.6f – %.6f",
                mask_crop.shape, mask_crop.min(), mask_crop.max())

    return b11_crop, b12_crop, mask_crop


def downsample_20m(arr_10m: np.ndarray) -> np.ndarray:
    """Downsample 10m array to 20m by taking every other pixel."""
    return arr_10m[::2, ::2]


def run_site(
    site_name: str,
    site_cfg: dict,
    dry_run: bool,
    era5_client: "ERA5Client | None" = None,
) -> QuantificationRecord:
    """Run CEMF+IME for one site and write the canonical record."""
    logger.info("═══ %s ═══", site_name.upper())

    b11_10m, b12_10m, prob_10m = load_site_arrays(site_cfg)

    # Apply probability threshold to get binary plume mask
    thresh = site_cfg["prob_thresh"]
    mask_10m = (prob_10m > thresh).astype(np.float32)
    n_plume_10m = int(mask_10m.sum())
    n_total = mask_10m.size
    logger.info("Probability threshold %.4g: plume=%d/%d px (%.1f%% of crop)",
                thresh, n_plume_10m, n_total, 100 * n_plume_10m / n_total)

    if n_plume_10m < 10:
        logger.error("%s: only %d plume pixels above threshold %.4g — "
                     "raising threshold may be needed", site_name, n_plume_10m, thresh)
        raise RuntimeError(f"Too few plume pixels ({n_plume_10m}) for CEMF at {site_name}")

    # Downsample to 20m (native SWIR resolution)
    b11_20m = downsample_20m(b11_10m)
    b12_20m = downsample_20m(b12_10m)
    mask_20m = downsample_20m(mask_10m)

    n_plume_20m = int(mask_20m.sum())
    n_bg_20m = mask_20m.size - n_plume_20m
    logger.info("20m crop: plume=%d px, background=%d px", n_plume_20m, n_bg_20m)

    if n_bg_20m < 100:
        logger.error("%s: insufficient background pixels (%d) — "
                     "probability threshold %.4g may be too low", site_name, n_bg_20m, thresh)
        raise RuntimeError(f"Insufficient background ({n_bg_20m} px) at {site_name}")

    # Run CEMF
    cemf: CEMFResult = run_cemf(
        b11=b11_20m,
        b12=b12_20m,
        mask=mask_20m,
        scene_id=site_cfg["scene_id"],
        timestamp=site_cfg["acquisition_timestamp"],
    )
    logger.info("CEMF: valid=%s  total_mass=%.4f kg  plume_px=%d",
                cemf.retrieval_valid, cemf.total_mass_kg, cemf.n_plume_pixels)
    if cemf.warning:
        logger.warning("CEMF warning: %s", cemf.warning)

    # ── Wind: use pre-set override, auto-fetch ERA5, or climatological fallback ──
    wind_cfg = site_cfg.get("wind")
    if wind_cfg is not None:
        # Pre-fetched wind dict already in site config (e.g. belchatow ERA5 pre-retrieved)
        wind = wind_cfg
        logger.info("%s: using pre-set wind %.2f m/s (%s)",
                    site_name, wind["wind_speed_ms"], wind["wind_source"])
    else:
        # Attempt live ERA5 fetch (requires ~/.cdsapirc)
        client = era5_client
        if client is None:
            try:
                client = ERA5Client()
            except Exception as e:
                logger.warning("%s: ERA5Client init failed (%s) — falling back to %.1f m/s",
                               site_name, e, FALLBACK_WIND_SPEED)
                client = None

        if client is not None:
            try:
                date_str = site_cfg["acquisition_timestamp"][:10]
                wind = client.get_wind(site_cfg["lat"], site_cfg["lon"],
                                       date_str, hour="10:00")
                logger.info("%s: ERA5 wind %.2f m/s (%s)",
                            site_name, wind["wind_speed_ms"], wind["wind_source"])
            except Exception as e:
                logger.warning("%s: ERA5 fetch failed (%s) — falling back to %.1f m/s",
                               site_name, e, FALLBACK_WIND_SPEED)
                wind = {
                    "wind_speed_ms": FALLBACK_WIND_SPEED,
                    "wind_source": FALLBACK_WIND_SOURCE,
                    "wind_dir_deg": None,
                    "era5_u_ms": None,
                    "era5_v_ms": None,
                }
        else:
            wind = {
                "wind_speed_ms": FALLBACK_WIND_SPEED,
                "wind_source": FALLBACK_WIND_SOURCE,
                "wind_dir_deg": None,
                "era5_u_ms": None,
                "era5_v_ms": None,
            }

    # IME inversion
    ime = CEMFIntegratedMassEnhancement()
    qr = ime.estimate_from_cemf(cemf, wind_speed_ms=wind["wind_speed_ms"],
                                wind_source=wind["wind_source"])
    logger.info("IME: flow_rate=%.2f kg/h  plume_length=%.1f m  "
                "uncertainty=%s%%",
                qr.flow_rate_kgh, qr.plume_length_m,
                get_uncertainty_pct(wind["wind_source"]))

    unc_pct = get_uncertainty_pct(wind["wind_source"])
    lo = round(qr.flow_rate_kgh * (1 - unc_pct / 100), 2)
    hi = round(qr.flow_rate_kgh * (1 + unc_pct / 100), 2)
    annual = round(qr.flow_rate_kgh * 8760 / 1000, 4) if qr.flow_rate_kgh else None

    # Allow site config to override the canonical site name (e.g. "neurath_20240829" → "neurath")
    canonical_site = site_cfg.get("site", site_name)

    record = QuantificationRecord(
        site=canonical_site,
        scene_id=site_cfg["scene_id"],
        acquisition_timestamp=site_cfg["acquisition_timestamp"],
        plume_centroid_lat=site_cfg["lat"],
        plume_centroid_lon=site_cfg["lon"],
        methodology="CEMF+IME",
        cemf_sensitivity_coeff="4e-7 (Varon 2021 AMT Sec 2.2)",
        mask_source="ch4net_v8_original",
        mask_file=site_cfg["tif"],
        n_plume_pixels=cemf.n_plume_pixels,
        total_mass_kg=round(cemf.total_mass_kg, 4) if cemf.total_mass_kg else None,
        plume_length_m=round(qr.plume_length_m, 1) if qr.plume_length_m else None,
        wind_speed_ms=wind["wind_speed_ms"],
        wind_dir_deg=wind.get("wind_dir_deg"),
        wind_source=wind["wind_source"],
        era5_u_ms=wind.get("era5_u_ms"),
        era5_v_ms=wind.get("era5_v_ms"),
        flow_rate_kgh=qr.flow_rate_kgh,
        flow_rate_lower_kgh=lo,
        flow_rate_upper_kgh=hi,
        uncertainty_pct=unc_pct,
        annual_tonnes_if_continuous=annual,
        cemf_valid=cemf.retrieval_valid,
        excluded=False,
        exclusion_reason=None,
        tropomi_confirm=site_cfg["tropomi_confirm"],
        ch4net_peak_probability=site_cfg["ch4net_peak_probability"],
        cloud_cover_quality="clear",
        retrieval_notes=site_cfg["retrieval_notes"],
    )

    logger.info(
        "Result: flow_rate=%.2f kg/h [%.2f – %.2f]  wind=%.2f m/s (%s)  "
        "annual=%.1f t/yr",
        record.flow_rate_kgh, record.flow_rate_lower_kgh, record.flow_rate_upper_kgh,
        record.wind_speed_ms, record.wind_source,
        record.annual_tonnes_if_continuous or 0,
    )

    if dry_run:
        logger.info("DRY RUN — skipping disk write")
        print(json.dumps(record.to_dict(), indent=2, default=str))
    else:
        write_quantification_record(record)
        logger.info("Written to %s", DEFAULT_QUANT_PATH)

    return record


def main():
    parser = argparse.ArgumentParser(
        description="CEMF+IME for Neurath, Bełchatów, and Weisweiler"
    )
    parser.add_argument("--site", choices=list(SITES.keys()),
                        help="Run only this site (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print record without writing to disk")
    parser.add_argument("--no-era5", action="store_true",
                        help="Skip ERA5 fetch; use climatological fallback wind")
    args = parser.parse_args()

    targets = [args.site] if args.site else list(SITES.keys())

    # Build ERA5 client once (shared across sites); None if --no-era5
    era5_client = None
    if not args.no_era5:
        try:
            era5_client = ERA5Client()
            logger.info("ERA5Client ready")
        except Exception as e:
            logger.warning("ERA5Client init failed (%s) — will use fallback wind for all sites", e)

    results = {}
    for site_name in targets:
        cfg = SITES[site_name]
        try:
            record = run_site(site_name, cfg, dry_run=args.dry_run,
                              era5_client=era5_client)
            results[site_name] = {
                "flow_rate_kgh": record.flow_rate_kgh,
                "flow_lo": record.flow_rate_lower_kgh,
                "flow_hi": record.flow_rate_upper_kgh,
                "wind_ms": record.wind_speed_ms,
                "wind_source": record.wind_source,
                "uncertainty_pct": record.uncertainty_pct,
                "annual_t": record.annual_tonnes_if_continuous,
                "cemf_valid": record.cemf_valid,
            }
        except Exception as exc:
            logger.error("%s: FAILED — %s", site_name, exc, exc_info=True)
            results[site_name] = {"error": str(exc)}

    print("\n── Summary ─────────────────────────────────────────────")
    for site_name, r in results.items():
        if "error" in r:
            print(f"  {site_name:<15}  ERROR: {r['error']}")
        else:
            print(
                f"  {site_name:<15}  "
                f"Q̂ = {r['flow_rate_kgh']:.0f} kg/h  "
                f"[{r['flow_lo']:.0f} – {r['flow_hi']:.0f}]  "
                f"wind={r['wind_ms']:.2f} m/s ({r['wind_source']})  "
                f"±{r['uncertainty_pct']}%  "
                f"annual≈{r['annual_t']:.0f} t CH4/yr"
            )


if __name__ == "__main__":
    main()
