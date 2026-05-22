"""
scripts/fix_quantification_records.py
======================================
One-shot correction pass to clean up quantification.json after the
scene_id-based upsert was introduced.

Corrections applied:
  1. Weisweiler 2024-09-18 (T31UGS): retract Q̂=2724 kg/h → cemf_valid=False, flow=None
     Reason: South directional control mu=0.020 (43× N/E/W), CV=1.607 contaminates
     the binary CEMF mask regardless of wind. ERA5 5.05 m/s makes terrain-inflated
     mass even larger. Same root cause as the previously retracted 1942 kg/h estimate.

  2. Neurath 2024-08-29 (T32ULB): was overwritten by write_multidate_detection_records.py
     with cemf_valid=False/flow=None. Correct record: Q̂=85 kg/h (ERA5 2.66 m/s ±30%).
     Note: this is plausible — low plume extent on Aug-29 despite high S/C.
     Write the correct CEMF result.

  3. Bełchatów 2024-07-10 (T34UCB): same overwrite issue.
     Correct record: Q̂=1071 kg/h (ERA5 3.32 m/s ±30%). Write the correct CEMF result.

Usage:
    python scripts/fix_quantification_records.py [--dry-run]
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


CORRECTIONS = [

    # ── Weisweiler: retract CEMF (terrain contamination) ─────────────────────
    QuantificationRecord(
        site="weisweiler",
        scene_id="S2B_MSIL1C_20240918T103619_N0511_R008_T31UGS_20240918T142046",
        acquisition_timestamp="2024-09-18T10:36:19Z",
        plume_centroid_lat=50.837,
        plume_centroid_lon=6.322,
        methodology="CEMF+IME",
        cemf_sensitivity_coeff="4e-7 (Varon 2021 AMT Sec 2.2)",
        mask_source="ch4net_v8_original",
        mask_file=(
            "results_bitemporal/weisweiler/"
            "original_S2B_MSIL1C_20240918T103619_N0511_R008_T31UGS_20240918T142046.tif"
        ),
        n_plume_pixels=0,
        total_mass_kg=None,
        plume_length_m=None,
        wind_speed_ms=5.0491,
        wind_dir_deg=53.0,
        wind_source="ERA5_reanalysis",
        era5_u_ms=-4.0313,
        era5_v_ms=-3.0401,
        flow_rate_kgh=None,
        flow_rate_lower_kgh=None,
        flow_rate_upper_kgh=None,
        uncertainty_pct=30,
        annual_tonnes_if_continuous=None,
        cemf_valid=False,
        excluded=False,
        exclusion_reason=None,
        tropomi_confirm=False,
        ch4net_peak_probability=0.42,
        cloud_cover_quality="clear",
        retrieval_notes=(
            "CH4Net S/C=23.46 (T31UGS, 2024-09-18) — classic detection, CFAR not triggered. "
            "ERA5: 5.05 m/s, dir=53.0°. "
            "CEMF retracted: South directional ctrl_mean=0.020 (43× N/E/W ctrl mean ~0.0005), "
            "CV=1.607 — Rhine/Aachen terrain heterogeneity contaminates the binary mask. "
            "Raw CEMF outputs: raw_mass≈1085 kg, raw_flow≈2724 kg/h (physically implausible; "
            "8× Neurath rate for 0.18× the plant capacity). "
            "Corrective path: winter BT (T31UGS_ref_20240127.npy cached) to suppress "
            "stable terrain background before re-running CEMF."
        ),
    ),

    # ── Neurath 2024-08-29: correct CEMF result (overwritten by write_multidate) ──
    # CEMF output: Q̂=85 kg/h, ERA5 2.66 m/s. Low plume extent despite high S/C.
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
        wind_speed_ms=2.6612,
        wind_dir_deg=254.4,
        wind_source="ERA5_reanalysis",
        era5_u_ms=2.5626,
        era5_v_ms=0.7177,
        flow_rate_kgh=85.0,
        flow_rate_lower_kgh=round(85.0 * 0.70, 2),
        flow_rate_upper_kgh=round(85.0 * 1.30, 2),
        uncertainty_pct=30,
        annual_tonnes_if_continuous=round(85.0 * 8760 / 1000, 4),
        cemf_valid=True,
        excluded=False,
        exclusion_reason=None,
        tropomi_confirm=True,
        ch4net_peak_probability=0.93,
        cloud_cover_quality="clear",
        retrieval_notes=(
            "Second confirmed detection at Neurath: S/C=67.2 (classic), "
            "sc_cfar=97.0, CFAR margin=94.96 (T32ULB, 2024-08-29). "
            "cv_ctrl=0.288 — lower terrain CV than Jun-25 (0.992). "
            "ERA5: 2.66 m/s, dir=254.4°. "
            "Q̂=85 kg/h ±30% [60–111] — lower than Jun-25 estimate (338 kg/h) because "
            "fewer plume pixels above threshold on this date despite higher S/C ratio. "
            "TROPOMI-confirmed emitter (same facility; Jun-25 DXCH4=+12.2 ppb)."
        ),
    ),

    # ── Bełchatów 2024-07-10: correct CEMF result (overwritten by write_multidate) ──
    # CEMF output: Q̂=1071 kg/h, ERA5 3.32 m/s.
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
        wind_speed_ms=3.3183,
        wind_dir_deg=124.9,
        wind_source="ERA5_reanalysis",
        era5_u_ms=-2.7207,
        era5_v_ms=1.8996,
        flow_rate_kgh=1071.0,
        flow_rate_lower_kgh=round(1071.0 * 0.70, 2),
        flow_rate_upper_kgh=round(1071.0 * 1.30, 2),
        uncertainty_pct=30,
        annual_tonnes_if_continuous=round(1071.0 * 8760 / 1000, 4),
        cemf_valid=True,
        excluded=False,
        exclusion_reason=None,
        tropomi_confirm=False,
        ch4net_peak_probability=0.82,
        cloud_cover_quality="clear",
        retrieval_notes=(
            "Strongest Belchatow detection: S/C=142.9 (classic), "
            "sc_cfar=50.7, CFAR margin=46.2 (T34UCB, 2024-07-10). "
            "cv_ctrl=1.105 — moderate terrain heterogeneity; CFAR still triggered. "
            "ERA5: 3.32 m/s, dir=124.9°. "
            "Q̂=1071 kg/h ±30% [750–1393]. "
            "2.5× the Aug-24 estimate (426 kg/h) consistent with higher column enhancement "
            "(S/C 142.9 vs 27.3) and stronger wind. "
            "No direct TROPOMI co-location (S5P swath does not cover site on this date)."
        ),
    ),
]


def main():
    parser = argparse.ArgumentParser(
        description="Apply targeted corrections to quantification.json"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print corrected records without writing to disk")
    args = parser.parse_args()

    for rec in CORRECTIONS:
        if args.dry_run:
            print(f"\n{'='*60}")
            print(f"  [dry-run] {rec.site} / {rec.scene_id[:44]}")
            print(f"  flow_rate_kgh={rec.flow_rate_kgh}  cemf_valid={rec.cemf_valid}")
            print(json.dumps(rec.to_dict(), indent=2, default=str)[:400])
        else:
            write_quantification_record(rec)
            logger.info(
                "%-30s  cemf_valid=%-5s  flow=%-10s",
                rec.site + " " + rec.acquisition_timestamp[:10],
                rec.cemf_valid,
                rec.flow_rate_kgh,
            )

    if not args.dry_run:
        logger.info("Corrections written to %s", DEFAULT_QUANT_PATH)
        print("\nCorrected state:")
        print("  weisweiler   2024-09-18  flow=None  (CEMF retracted, terrain CV=1.607)")
        print("  neurath      2024-08-29  flow=85 kg/h  ERA5 ±30%")
        print("  belchatow    2024-07-10  flow=1071 kg/h  ERA5 ±30%")


if __name__ == "__main__":
    main()
