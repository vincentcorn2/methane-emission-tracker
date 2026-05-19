"""
scripts/reingest_gap8.py
=========================
Gap 8 cleanup: write canonical quantification records for the two confirmed
detections that were lost in the site-keyed writer cycle.

  - Neurath 2024-06-25   (S2A T32ULB)  TROPOMI-confirmed, ~338 kg/h
  - Bełchatów 2024-08-24 (S2B T34UCB)  cv_ctrl=1.265, ~426 kg/h

Both NPY files and TIF probability maps already exist.
ERA5 wind for Aug-24 was pre-retrieved (2.19 m/s, 245°).
ERA5 wind for Jun-25 uses fallback 3.5 m/s in sandbox (will be governance-flagged).

Usage:
    conda activate methane
    python scripts/reingest_gap8.py [--dry-run]
"""

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np

try:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
except ImportError:
    Image = None

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from src.quantification.cemf import run_cemf, downsample_mask
from src.quantification.ime import CEMFIntegratedMassEnhancement
from src.quantification.uncertainty import get_uncertainty_pct
from src.quantification.canonical_writer import (
    QuantificationRecord, write_quantification_record, DEFAULT_QUANT_PATH
)
from src.ingestion.era5_client import ERA5Client, FALLBACK_WIND_SPEED, FALLBACK_WIND_SOURCE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("reingest_gap8")

B11_IDX, B12_IDX = 10, 11
CROP_PX = 500

SITES = {
    "neurath_20240625": {
        "site":             "neurath",
        "scene_id":         "S2A_MSIL1C_20240625T103631_N0510_R008_T32ULB_20240625T142035",
        "acquisition_timestamp": "2024-06-25T10:36:31Z",
        "npy":  "data/npy_cache/S2A_MSIL1C_20240625T103631_N0510_R008_T32ULB_20240625T142035.npy",
        "tif":  "results_bitemporal/neurath/original_S2A_MSIL1C_20240625T103631_N0510_R008_T32ULB_20240625T142035.tif",
        "site_row": 4324, "site_col": 3286,
        "lat": 51.038, "lon": 6.616,
        "prob_thresh": 0.016,           # 3× ctrl_mean (from run_cemf_neurath_belchatow.py)
        "mask_source": "ch4net_v8_original",
        "mask_file":   "results_bitemporal/neurath/original_S2A_MSIL1C_20240625T103631_N0510_R008_T32ULB_20240625T142035.tif",
        "wind":        None,            # ERA5 unavailable in sandbox → fallback
        "tropomi_confirm": True,
        "ch4net_peak_probability": 0.93,
        "cloud_cover_quality": "clear",
        "retrieval_notes": (
            "Gap 8 re-ingest (2026-04-17). TROPOMI-confirmed: DXCH4=+12.2 ppb "
            "(Jun-25 orbit, qa=1.00). S/C=23.04, cv_ctrl=0.992. "
            "Probability threshold: 0.016 (3× ctrl_mean). "
            "ERA5 wind unavailable in sandbox — using 3.5 m/s fallback. "
            "Record governance-flagged: WIND_FALLBACK; uncertainty inflated to ±50%."
        ),
    },
    "belchatow_20240824": {
        "site":             "belchatow",
        "scene_id":         "S2B_MSIL1C_20240824T094549_N0511_R079_T34UCB_20240824T115611",
        "acquisition_timestamp": "2024-08-24T09:45:49Z",
        "npy":  "data/npy_cache/S2B_MSIL1C_20240824T094549_N0511_R079_T34UCB_20240824T115611.npy",
        "tif":  "results_bitemporal/belchatow/original_S2B_MSIL1C_20240824T094549_N0511_R079_T34UCB_20240824T115611.tif",
        "site_row": 1949, "site_col": 8356,
        "lat": 51.264, "lon": 19.331,
        "prob_thresh": 0.0002,          # 5× ctrl_mean (from run_cemf_neurath_belchatow.py)
        "mask_source": "ch4net_v8_original",
        "mask_file":   "results_bitemporal/belchatow/original_S2B_MSIL1C_20240824T094549_N0511_R079_T34UCB_20240824T115611.tif",
        "wind": {                       # pre-retrieved ERA5 (from original run)
            "wind_speed_ms": 2.19,
            "wind_dir_deg":  245.0,
            "wind_source":   "ERA5_reanalysis",
            "era5_u_ms":    -1.55,
            "era5_v_ms":    -1.55,
        },
        "tropomi_confirm": False,
        "ch4net_peak_probability": 0.82,
        "cloud_cover_quality": "clear",
        "retrieval_notes": (
            "Gap 8 re-ingest (2026-04-17). S/C=27.30, cv_ctrl=1.265. "
            "CFAR not triggered (cv_ctrl=1.265, high terrain variability). "
            "BT mask excluded (S/C 27.3→1.94 after BT differencing). "
            "TROPOMI cloud-blocked on acquisition date. "
            "ERA5 wind pre-retrieved: 2.19 m/s, dir=245°."
        ),
    },
}


def load_crop(cfg):
    """Load B11, B12, and probability map crops for a site."""
    row, col = cfg["site_row"], cfg["site_col"]
    half = CROP_PX // 2
    r0, r1, c0, c1 = row - half, row + half, col - half, col + half

    arr = np.load(cfg["npy"], mmap_mode="r")
    b11 = arr[r0:r1, c0:c1, B11_IDX].astype(np.float32) / 255.0
    b12 = arr[r0:r1, c0:c1, B12_IDX].astype(np.float32) / 255.0

    if Image:
        img = Image.open(cfg["tif"])
        prob = np.array(img.crop((c0, r0, c1, r1))).astype(np.float32)
    else:
        import rasterio
        from rasterio.windows import Window
        with rasterio.open(cfg["tif"]) as src:
            prob = src.read(1, window=Window(c0, r0, CROP_PX, CROP_PX)).astype(np.float32)

    return b11, b12, prob


def run_one(name, cfg, dry_run):
    log.info("── %s ──", name.upper())
    b11_10m, b12_10m, prob_10m = load_crop(cfg)

    thresh = cfg["prob_thresh"]
    mask_10m = (prob_10m > thresh).astype(np.float32)
    log.info("  Plume pixels at thresh=%.4g: %d / %d",
             thresh, int(mask_10m.sum()), mask_10m.size)

    # Downsample to 20m
    b11_20m = b11_10m[::2, ::2]
    b12_20m = b12_10m[::2, ::2]
    mask_20m = mask_10m[::2, ::2]

    # Wind
    wind_cfg = cfg.get("wind")
    if wind_cfg:
        wind_ms    = wind_cfg["wind_speed_ms"]
        wind_dir   = wind_cfg.get("wind_dir_deg", 0.0)
        wind_source = wind_cfg["wind_source"]
        era5_u     = wind_cfg.get("era5_u_ms")
        era5_v     = wind_cfg.get("era5_v_ms")
    else:
        log.info("  Fetching ERA5 wind...")
        try:
            client = ERA5Client()
            res    = client.get_wind(cfg["lat"], cfg["lon"], cfg["acquisition_timestamp"])
            wind_ms = float(res["wind_speed_ms"])
            wind_dir = res.get("wind_dir_deg") or 0.0
            wind_source = res.get("wind_source", "ERA5_reanalysis")
            era5_u  = res.get("era5_u_ms")
            era5_v  = res.get("era5_v_ms")
        except Exception as e:
            log.warning("  ERA5 failed (%s) → fallback %.1f m/s", e, FALLBACK_WIND_SPEED)
            wind_ms     = FALLBACK_WIND_SPEED
            wind_source = FALLBACK_WIND_SOURCE
            wind_dir    = None
            era5_u = era5_v = None

    log.info("  Wind: %.3f m/s  dir=%.1f°  source=%s",
             wind_ms, wind_dir or 0.0, wind_source)

    # CEMF
    from src.quantification.cemf import run_cemf as _run_cemf
    cemf_result = _run_cemf(b11_20m, b12_20m, mask_20m,
                            cfg["scene_id"], cfg["acquisition_timestamp"])
    log.info("  CEMF: valid=%s  mass=%.4f kg  pixels=%d",
             cemf_result.retrieval_valid, cemf_result.total_mass_kg,
             cemf_result.n_plume_pixels)

    # IME
    ime = CEMFIntegratedMassEnhancement()
    q_result = ime.estimate_from_cemf(cemf_result,
                                       wind_speed_ms=wind_ms,
                                       wind_source=wind_source)
    log.info("  Flow: %.1f kg/h  [%.1f, %.1f]  uncertainty=%d%%",
             q_result.flow_rate_kgh, q_result.flow_rate_lower_kgh,
             q_result.flow_rate_upper_kgh,
             get_uncertainty_pct(wind_source))

    if dry_run:
        log.info("  [dry-run] would write record — skipping")
        return

    # Build canonical record
    rec = QuantificationRecord(
        site             = cfg["site"],
        scene_id         = cfg["scene_id"],
        acquisition_timestamp = cfg["acquisition_timestamp"],
        plume_centroid_lat = cfg["lat"],
        plume_centroid_lon = cfg["lon"],
        methodology      = "CEMF+IME",
        cemf_sensitivity_coeff = "4e-7 (Varon 2021 AMT Sec 2.2)",
        mask_source      = cfg["mask_source"],
        mask_file        = cfg["mask_file"],
        n_plume_pixels   = cemf_result.n_plume_pixels,
        total_mass_kg    = cemf_result.total_mass_kg,
        plume_length_m   = q_result.plume_length_m,
        wind_speed_ms    = wind_ms,
        wind_dir_deg     = wind_dir,
        wind_source      = wind_source,
        era5_u_ms        = era5_u,
        era5_v_ms        = era5_v,
        flow_rate_kgh    = q_result.flow_rate_kgh,
        flow_rate_lower_kgh = q_result.flow_rate_lower_kgh,
        flow_rate_upper_kgh = q_result.flow_rate_upper_kgh,
        uncertainty_pct  = get_uncertainty_pct(wind_source),
        cemf_valid       = cemf_result.retrieval_valid and q_result.flow_rate_kgh > 0,
        excluded         = False,
        tropomi_confirm  = cfg["tropomi_confirm"],
        ch4net_peak_probability = cfg["ch4net_peak_probability"],
        cloud_cover_quality = cfg["cloud_cover_quality"],
        retrieval_notes  = cfg["retrieval_notes"],
    )

    write_quantification_record(rec)
    log.info("  Record written to %s", DEFAULT_QUANT_PATH)
    return rec


def main():
    parser = argparse.ArgumentParser(description="Gap 8: re-ingest missing canonical records")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--site", default=None,
                        help="neurath_20240625 or belchatow_20240824")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  Gap 8 Re-ingest — Neurath Jun-25 + Bełchatów Aug-24")
    if args.dry_run:
        print("  MODE: DRY RUN")
    print("=" * 60)

    targets = SITES
    if args.site:
        targets = {k: v for k, v in SITES.items() if k == args.site}

    results = {}
    for name, cfg in targets.items():
        try:
            rec = run_one(name, cfg, args.dry_run)
            results[name] = "ok" if rec or args.dry_run else "dry_run"
        except Exception as e:
            log.exception("Failed: %s: %s", name, e)
            results[name] = f"error: {e}"

    print("\nResults:")
    for name, status in results.items():
        print(f"  {'✓' if 'ok' in status or 'dry' in status else '✗'} {name}: {status}")

    # Verify final state
    if not args.dry_run:
        records = json.load(open(DEFAULT_QUANT_PATH))
        print(f"\nquantification.json now has {len(records)} records:")
        for r in records:
            flow = r.get('flow_rate_kgh')
            valid = r.get('cemf_valid')
            flags = r.get('input_degradation', {}).get('flags', [])
            flag_str = f"  ⚑ {flags}" if flags else ""
            print(f"  {r.get('site'):<15} "
                  f"{'flow='+str(flow)+' kg/h' if flow else 'no flow':<22} "
                  f"valid={valid}{flag_str}")


if __name__ == "__main__":
    main()
