"""
scripts/write_multidate_detection_records.py
=============================================
Writes provenance records for two confirmed detections that are NOT yet in
quantification.json:

  1. neurath_20240829   — S/C=67.2, CFAR margin=94.96 (strongest Neurath date)
  2. belchatow_20240710 — S/C=142.9, CFAR margin=46.2 (strongest Bełchatów date)

Both records have cemf_valid=False and flow_rate_kgh=None because CEMF
quantification requires ERA5 wind (run fetch_era5_pending.py first, then
re-run run_cemf_neurath_belchatow.py to populate flow estimates).

Detection is already confirmed — these records establish the detection provenance
so the sites appear in the canonical record even before quantification completes.

Usage:
    python scripts/write_multidate_detection_records.py [--dry-run]
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.quantification.canonical_writer import (
    QuantificationRecord,
    write_quantification_record,
    DEFAULT_QUANT_PATH,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RECORDS = [
    # ── Neurath 2024-08-29: second (stronger) detection ──────────────────────
    # S/C=67.2, CFAR margin=94.96, cv_ctrl=0.288 — cleaner than Jun-25 (cv=0.992)
    # CEMF pending ERA5 fetch. TIF: results_bitemporal/neurath/original_S2B_*.tif
    QuantificationRecord(
        site="neurath",
        scene_id="S2B_MSIL1C_20240829T103629_N0511_R008_T32ULB_20240829T124434",
        acquisition_timestamp="2024-08-29T10:36:29Z",
        plume_centroid_lat=51.038,
        plume_centroid_lon=6.616,
        methodology="CEMF+IME",
        cemf_sensitivity_coeff="4e-7 (Varon 2021 AMT Sec 2.2)",
        mask_source="ch4net_v8_original",
        mask_file=(
            "results_bitemporal/neurath/"
            "original_S2B_MSIL1C_20240829T103629_N0511_R008_T32ULB_20240829T124434.tif"
        ),
        n_plume_pixels=0,
        total_mass_kg=None,
        plume_length_m=None,
        wind_speed_ms=None,
        wind_dir_deg=None,
        wind_source="none",
        era5_u_ms=None,
        era5_v_ms=None,
        flow_rate_kgh=None,
        flow_rate_lower_kgh=None,
        flow_rate_upper_kgh=None,
        uncertainty_pct=50,
        annual_tonnes_if_continuous=None,
        cemf_valid=False,
        excluded=False,
        exclusion_reason=None,
        tropomi_confirm=True,
        ch4net_peak_probability=0.93,
        cloud_cover_quality="clear",
        retrieval_notes=(
            "Second confirmed detection at Neurath: S/C=67.2 (classic), "
            "sc_cfar=97.0, CFAR margin=94.96 (T32ULB, 2024-08-29). "
            "cv_ctrl=0.288 — lower terrain noise than Jun-25 date (cv=0.992). "
            "TROPOMI-confirmed emitter (same facility; Jun-25 DXCH4=+12.2 ppb). "
            "CEMF quantification pending ERA5 wind fetch. "
            "Run: python scripts/fetch_era5_pending.py --site neurath_20240829, "
            "then: python scripts/run_cemf_neurath_belchatow.py --site neurath_20240829"
        ),
    ),

    # ── Bełchatów 2024-07-10: strongest detection across all dates ───────────
    # S/C=142.9, CFAR margin=46.2 — highest S/C across all evaluated Bełchatów dates.
    # cv_ctrl=1.105 (moderate terrain heterogeneity); CFAR still triggered.
    # CEMF pending ERA5 fetch. TIF: results_bitemporal/belchatow/original_S2A_*.tif
    QuantificationRecord(
        site="belchatow",
        scene_id="S2A_MSIL1C_20240710T095031_N0510_R079_T34UCB_20240710T133148",
        acquisition_timestamp="2024-07-10T09:50:31Z",
        plume_centroid_lat=51.264,
        plume_centroid_lon=19.331,
        methodology="CEMF+IME",
        cemf_sensitivity_coeff="4e-7 (Varon 2021 AMT Sec 2.2)",
        mask_source="ch4net_v8_original",
        mask_file=(
            "results_bitemporal/belchatow/"
            "original_S2A_MSIL1C_20240710T095031_N0510_R079_T34UCB_20240710T133148.tif"
        ),
        n_plume_pixels=0,
        total_mass_kg=None,
        plume_length_m=None,
        wind_speed_ms=None,
        wind_dir_deg=None,
        wind_source="none",
        era5_u_ms=None,
        era5_v_ms=None,
        flow_rate_kgh=None,
        flow_rate_lower_kgh=None,
        flow_rate_upper_kgh=None,
        uncertainty_pct=50,
        annual_tonnes_if_continuous=None,
        cemf_valid=False,
        excluded=False,
        exclusion_reason=None,
        tropomi_confirm=False,
        ch4net_peak_probability=0.82,
        cloud_cover_quality="clear",
        retrieval_notes=(
            "Strongest Belchatow detection: S/C=142.9 (classic), "
            "sc_cfar=50.7, CFAR margin=46.2 (T34UCB, 2024-07-10). "
            "cv_ctrl=1.105 — moderate terrain heterogeneity; CFAR triggered. "
            "CEMF quantification pending ERA5 wind fetch. "
            "Run: python scripts/fetch_era5_pending.py --site belchatow_20240710, "
            "then: python scripts/run_cemf_neurath_belchatow.py --site belchatow_20240710"
        ),
    ),
]


def main():
    parser = argparse.ArgumentParser(
        description="Write multi-date detection provenance records to quantification.json"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print records without writing to disk")
    args = parser.parse_args()

    for rec in RECORDS:
        if args.dry_run:
            print(f"\n{'='*60}")
            print(f"  [dry-run] {rec.site} / {rec.scene_id[:40]}")
            print(json.dumps(rec.to_dict(), indent=2, default=str))
        else:
            write_quantification_record(rec)
            logger.info(
                "%-30s  cemf_valid=%-5s  flow_rate_kgh=%-10s  sc_note: %s",
                rec.site + " " + rec.acquisition_timestamp[:10],
                rec.cemf_valid,
                rec.flow_rate_kgh,
                rec.retrieval_notes[:60],
            )

    if not args.dry_run:
        logger.info("Records written to %s", DEFAULT_QUANT_PATH)
        print("\nDetection summary:")
        print("  neurath  2024-08-29  S/C=67.2   CFAR margin=94.96  cv=0.288")
        print("  belchatow 2024-07-10  S/C=142.9  CFAR margin=46.2   cv=1.105")
        print("\nNext: python scripts/fetch_era5_pending.py")
        print("Then: python scripts/run_cemf_neurath_belchatow.py")


if __name__ == "__main__":
    main()
