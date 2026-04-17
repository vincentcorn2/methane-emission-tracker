"""
scripts/run_quantification.py
=============================
Canonical batch entry point for CEMF+IME+ERA5 quantification.

Calls src.quantification.runner.run_quantification() for each configured site
and writes the canonical QuantificationRecord to results_analysis/quantification.json
via canonical_writer.write_quantification_record().

This script is the ONLY permitted batch path for producing quantification.json.
The notebook (European_CH4_Pipeline.ipynb) calls runner.run_quantification()
directly cell-by-cell; this script runs the same function in a loop.

Usage:
  python scripts/run_quantification.py [--sites neurath belchatow] [--dry-run]

Flags:
  --sites      space-separated list of site slugs to process (default: all configured)
  --dry-run    skip writing to disk; prints the QuantificationRecord instead
  --no-era5    use climatological fallback wind (3.5 m/s) instead of fetching ERA5
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# Ensure project root is on path when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.quantification.runner import SiteCfg, run_quantification
from src.ingestion.era5_client import ERA5Client, FALLBACK_WIND_SPEED, FALLBACK_WIND_SOURCE

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Site configuration registry ───────────────────────────────────────────────
# Each entry provides the minimum metadata needed to locate the mask + band
# arrays from disk.  Band arrays (b11, b12, mask_original) are loaded lazily
# below via _load_arrays().  Add new sites here; runner.py handles the rest.

SITE_REGISTRY: dict[str, dict] = {
    "maasvlakte": {
        "scene_id": "S2A_T31UFT_20240604",
        "acquisition_timestamp": "2024-06-04T10:50:00Z",
        "lat": 51.954, "lon": 4.008,
        "tif_original": "results_validation/maasvlakte/detection_T31UFT_2024-06-04.tif",
        "tropomi_confirm": True,
        "ch4net_peak_probability": 0.87,
    },
    "belchatow": {
        "scene_id": "S2B_T34UCB_20240824",
        "acquisition_timestamp": "2024-08-24T09:40:00Z",
        "lat": 51.264, "lon": 19.331,
        "tif_original": "results_validation/belchatow/detection_T34UCB_2024-08-24.tif",
        "tif_bitemporal": "results_bitemporal/belchatow/bitemporal_S2B_MSIL1C_20240824_T34UCB.tif",
        "tropomi_confirm": False,
        "ch4net_peak_probability": 0.82,
        # ERA5 pre-retrieved: 2.19 m/s (2024-08-24, 09:40 UTC)
        "wind_override": {
            "wind_speed_ms": 2.19,
            "wind_dir_deg": 245.0,
            "wind_source": "ERA5_reanalysis",
            "era5_u_ms": -1.55,
            "era5_v_ms": -1.55,
        },
    },
    "neurath": {
        "scene_id": "S2A_T31UGS_20240625",
        "acquisition_timestamp": "2024-06-25T10:36:31Z",
        "lat": 51.038, "lon": 6.616,
        "tif_original": "results_validation/neurath/detection_T31UGS_2024-06-25.tif",
        "tropomi_confirm": True,
        "ch4net_peak_probability": 0.93,
        "retrieval_notes": "Dual-sensor confirm: TROPOMI DXCH4=+12.2 ppb; S/C=45.1",
        # ERA5 to be fetched live for 2024-06-25 12:00 UTC at (51.038, 6.616)
    },
    "weisweiler": {
        "scene_id": "S2A_T31UGS_20240620",
        "acquisition_timestamp": "2024-06-20T10:36:00Z",
        "lat": 50.859, "lon": 6.319,
        "tif_original": "results_weisweiler_multidate/detection_T31UGS_2024-06-20.tif",
        "tropomi_confirm": False,
        "ch4net_peak_probability": 0.78,
        # ERA5 incoming April 20 (WS2 deliverable)
    },
    "rybnik": {
        "scene_id": "S2B_T34UDA_20240815",
        "acquisition_timestamp": "2024-08-15T09:46:00Z",
        "lat": 50.099, "lon": 18.542,
        "tif_original": "results_validation/rybnik/detection_T34UDA_2024-08-15.tif",
        "tropomi_confirm": False,
        "ch4net_peak_probability": 0.61,
        # ERA5 incoming April 20
    },
    "groningen": {
        "scene_id": "S2B_T31UGV_20240817",
        "acquisition_timestamp": "2024-08-17T10:50:00Z",
        "lat": 53.252, "lon": 6.682,
        "tif_original": "results_validation/groningen/detection_T31UGV_2024-08-17.tif",
        "excluded": True,
        "exclusion_reason": "cfar_suppressed_false_positive",
    },
    "lippendorf": {
        "scene_id": "S2B_T33UUS_20240922",
        "acquisition_timestamp": "2024-09-22T10:16:29Z",
        "lat": 51.178, "lon": 12.378,
        "tif_original": "results_bitemporal/lippendorf/original_S2B_MSIL1C_20240922T101629_N0511_R065_T33UUS_20240922T140318.tif",
        "tif_bitemporal": "results_bitemporal/lippendorf/bitemporal_S2B_MSIL1C_20240922T101629_N0511_R065_T33UUS_20240922T140318.tif",
        "excluded": True,
        "exclusion_reason": "terrain_artifact",
    },
}


def _load_arrays(site_name: str, cfg: dict) -> dict:
    """
    Lazily load band and mask arrays from TIF / npy files for a site.
    Returns dict with keys b11, b12, mask_original, mask_bitemporal (optional).

    In production the notebook will already have arrays loaded in memory;
    this function is only needed for the batch script path.
    """
    arrays: dict = {"b11": None, "b12": None, "mask_original": None, "mask_bitemporal": None}

    # Load mask from GeoTIFF
    tif_orig = cfg.get("tif_original")
    if tif_orig and Path(tif_orig).exists():
        try:
            from PIL import Image
            import numpy as _np
            img = Image.open(tif_orig)
            arrays["mask_original"] = _np.array(img).astype(np.float32)
            if arrays["mask_original"].ndim == 3:
                arrays["mask_original"] = arrays["mask_original"][0]
        except Exception as exc:
            logger.warning("%s: could not load mask from %s: %s", site_name, tif_orig, exc)

    tif_bt = cfg.get("tif_bitemporal")
    if tif_bt and Path(tif_bt).exists():
        try:
            from PIL import Image
            import numpy as _np
            img = Image.open(tif_bt)
            arrays["mask_bitemporal"] = _np.array(img).astype(np.float32)
            if arrays["mask_bitemporal"].ndim == 3:
                arrays["mask_bitemporal"] = arrays["mask_bitemporal"][0]
        except Exception as exc:
            logger.warning("%s: could not load BT mask from %s: %s", site_name, tif_bt, exc)

    # Band arrays: look for npy cache
    npy_root = Path("data/npy_cache")
    scene_id = cfg.get("scene_id", "")
    for band_key, band_label in [("b11", "B11"), ("b12", "B12")]:
        candidates = list(npy_root.glob(f"*{scene_id}*{band_label}*.npy")) + \
                     list(npy_root.glob(f"*{band_label}*{scene_id}*.npy"))
        if candidates:
            arrays[band_key] = np.load(candidates[0])
            logger.info("%s: loaded %s from %s", site_name, band_label, candidates[0])
        else:
            logger.warning("%s: %s npy not found under %s", site_name, band_label, npy_root)

    return arrays


def main():
    parser = argparse.ArgumentParser(description="Run canonical CEMF+IME quantification")
    parser.add_argument("--sites", nargs="*", help="Site slugs to process (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="Skip writing to disk")
    parser.add_argument("--no-era5", action="store_true", help="Use climatological fallback wind")
    args = parser.parse_args()

    target_sites = args.sites or list(SITE_REGISTRY.keys())
    era5_client = None if args.no_era5 else ERA5Client()

    for site_name in target_sites:
        cfg = SITE_REGISTRY.get(site_name)
        if cfg is None:
            logger.warning("Site '%s' not in registry — skipping", site_name)
            continue

        # Excluded sites: write a provenance record but don't run CEMF
        if cfg.get("excluded", False):
            logger.info("Site '%s' is excluded (%s) — writing provenance record only", site_name, cfg.get("exclusion_reason"))
            from src.quantification.canonical_writer import QuantificationRecord, write_quantification_record
            record = QuantificationRecord(
                site=site_name,
                scene_id=cfg["scene_id"],
                acquisition_timestamp=cfg["acquisition_timestamp"],
                plume_centroid_lat=cfg["lat"],
                plume_centroid_lon=cfg["lon"],
                cemf_valid=False,
                excluded=True,
                exclusion_reason=cfg["exclusion_reason"],
            )
            if not args.dry_run:
                write_quantification_record(record)
            continue

        arrays = _load_arrays(site_name, cfg)

        wind_override = cfg.get("wind_override")
        if args.no_era5 and wind_override is None:
            wind_override = {"wind_speed_ms": FALLBACK_WIND_SPEED, "wind_source": FALLBACK_WIND_SOURCE,
                             "wind_dir_deg": None, "era5_u_ms": None, "era5_v_ms": None}

        site_cfg_obj = SiteCfg(
            site=site_name,
            scene_id=cfg["scene_id"],
            acquisition_timestamp=cfg["acquisition_timestamp"],
            lat=cfg["lat"],
            lon=cfg["lon"],
            b11=arrays["b11"],
            b12=arrays["b12"],
            mask_original=arrays["mask_original"],
            mask_bitemporal=arrays["mask_bitemporal"],
            wind_override=wind_override,
            tropomi_confirm=cfg.get("tropomi_confirm", False),
            ch4net_peak_probability=cfg.get("ch4net_peak_probability"),
            retrieval_notes=cfg.get("retrieval_notes", ""),
        )

        try:
            record = run_quantification(site_cfg_obj, dry_run=args.dry_run, era5_client=era5_client)
            logger.info(
                "%-15s  flow=%.1f kg/h  wind=%.2f m/s (%s)  n_pixels=%d",
                site_name, record.flow_rate_kgh or 0,
                record.wind_speed_ms or 0, record.wind_source,
                record.n_plume_pixels,
            )
            if args.dry_run:
                import json
                print(json.dumps(record.to_dict(), indent=2, default=str))
        except Exception as exc:
            logger.error("Site '%s' failed: %s", site_name, exc, exc_info=True)


if __name__ == "__main__":
    main()
