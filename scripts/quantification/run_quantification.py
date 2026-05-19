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
        "scene_id": "S2A_MSIL1C_20240625T103631_N0510_R008_T32ULB_20240625T142035",
        "acquisition_timestamp": "2024-06-25T10:36:31Z",
        "lat": 51.038, "lon": 6.616,
        "tif_original": "results_bitemporal/neurath/original_S2A_MSIL1C_20240625T103631_N0510_R008_T32ULB_20240625T142035.tif",
        "tropomi_confirm": True,
        "ch4net_peak_probability": 0.93,
        "retrieval_notes": "Dual-sensor confirm: TROPOMI DXCH4=+12.2 ppb; S/C=23.04",
        # ERA5 fetched via run_cemf_neurath_belchatow.py --site neurath
    },
    # Neurath second detection: S/C=67.2, CFAR=True — stronger than Jun-25
    "neurath_20240829": {
        "scene_id": "S2B_MSIL1C_20240829T103629_N0511_R008_T32ULB_20240829T124434",
        "acquisition_timestamp": "2024-08-29T10:36:29Z",
        "lat": 51.038, "lon": 6.616,
        "tif_original": "results_bitemporal/neurath/original_S2B_MSIL1C_20240829T103629_N0511_R008_T32ULB_20240829T124434.tif",
        "tropomi_confirm": True,   # same facility; TROPOMI confirmed emitter
        "ch4net_peak_probability": 0.93,
        "retrieval_notes": "Second detection at Neurath: S/C=67.2, CFAR margin=94.96, cv=0.288",
    },
    # Bełchatów strongest detection: S/C=142.9, CFAR=True (Jul-10)
    "belchatow_20240710": {
        "scene_id": "S2A_MSIL1C_20240710T095031_N0510_R079_T34UCB_20240710T133148",
        "acquisition_timestamp": "2024-07-10T09:50:31Z",
        "lat": 51.264, "lon": 19.331,
        "tif_original": "results_bitemporal/belchatow/original_S2A_MSIL1C_20240710T095031_N0510_R079_T34UCB_20240710T133148.tif",
        "tropomi_confirm": False,
        "ch4net_peak_probability": 0.82,
        "retrieval_notes": "Strongest Belchatow detection: S/C=142.9, CFAR margin=46.2, cv=1.105",
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
        # Corrected: T34UCA (not T34UDA), 2024-08-29 (not 2024-08-15)
        # CH4Net on 2024-08-29: S/C=0.612, sc_cfar=0.61 — single-date non-detection
        # Earlier "confirmed emitter" note referred to T34UDA 2024-08-15 (not cached)
        "scene_id": "S2A_MSIL1C_20240829T095031_N0511_R079_T34UCA_20240829T115208",
        "acquisition_timestamp": "2024-08-29T09:50:31Z",
        "lat": 50.135, "lon": 18.522,
        "tif_original": (
            "results_bitemporal/rybnik/"
            "original_S2A_MSIL1C_20240829T095031_N0511_R079_T34UCA_20240829T115208.tif"
        ),
        "tropomi_confirm": False,
        "ch4net_peak_probability": None,
        "excluded": True,
        "exclusion_reason": "single_date_non_detection",
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
    # ── Phase 5: JRC top-10 sites — tiles downloaded 2026-04-17 ─────────────
    # CH4Net run: apply_bitemporal_diff.py 2026-04-17
    # All three are excluded in run_quantification.py; canonical provenance
    # records written by scripts/write_phase5_records.py.
    "turceni": {
        # S/C=1.740 classic detect; CFAR adaptive: sc_cfar=1.506 vs thresh=1.625 — no
        # CEMF deferred: high-background Romanian agricultural terrain (mu_ctrl=0.260)
        # prevents reliable binary-mask isolation. Needs winter-2023 reference tile.
        "scene_id": "S2B_MSIL1C_20240927T093029_N0511_R136_T34TFP_20240927T102421",
        "acquisition_timestamp": "2024-09-27T09:30:29Z",
        "lat": 44.101, "lon": 23.391,
        "tif_original": (
            "results_bitemporal/turceni/"
            "original_S2B_MSIL1C_20240927T093029_N0511_R136_T34TFP_20240927T102421.tif"
        ),
        "tropomi_confirm": False,
        "ch4net_peak_probability": None,
        "excluded": True,
        "exclusion_reason": "cemf_deferred_high_background",
        # Operator: Oltenia Energy Complex, ~2,200 MW lignite (Romania)
    },
    "rovinari": {
        # S/C=0.715 — single-date non-detection on 2024-09-27 (T34TFQ)
        "scene_id": "S2B_MSIL1C_20240927T093029_N0511_R136_T34TFQ_20240927T102421",
        "acquisition_timestamp": "2024-09-27T09:30:29Z",
        "lat": 44.906, "lon": 23.147,
        "tif_original": (
            "results_bitemporal/rovinari/"
            "original_S2B_MSIL1C_20240927T093029_N0511_R136_T34TFQ_20240927T102421.tif"
        ),
        "tropomi_confirm": False,
        "ch4net_peak_probability": None,
        "excluded": True,
        "exclusion_reason": "single_date_non_detection",
        # Operator: Oltenia Energy Complex, ~1,320 MW lignite (Romania)
    },
    "maritsa_east_2": {
        # S/C=0.862 — single-date non-detection on 2024-09-28 (T35TMG)
        "scene_id": "S2B_MSIL1C_20240928T085649_N0511_R007_T35TMG_20240928T113459",
        "acquisition_timestamp": "2024-09-28T08:56:49Z",
        "lat": 42.271, "lon": 26.068,
        "tif_original": (
            "results_bitemporal/maritsa_east_2/"
            "original_S2B_MSIL1C_20240928T085649_N0511_R007_T35TMG_20240928T113459.tif"
        ),
        "tropomi_confirm": False,
        "ch4net_peak_probability": None,
        "excluded": True,
        "exclusion_reason": "single_date_non_detection",
        # Operator: AES-NEK / ContourGlobal, ~1,600 MW lignite (Bulgaria)
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
