"""
scripts/write_phase5_records.py
================================
Writes canonical provenance records for Phase 5 (JRC top-10) sites and the
Rybnik single-date evaluation.

Results from apply_bitemporal_diff.py run 2026-04-17:

  Site              Orig S/C    CFAR adaptive   Assessment
  turceni             1.740     no (sc=1.51,      Marginal classic detect; CFAR fails
                                thresh=1.63)      High-background terrain (mu_ctrl=0.260)
                                                  CEMF deferred — quantification unreliable
  rovinari            0.715     no (thresh=3.108) Single-date non-detection
  maritsa_east_2      0.862     no (thresh=2.279) Single-date non-detection
  rybnik              0.612     —                 Single-date non-detection (T34UCA 2024-08-29)

None of these sites had reference tiles (winter 2023) so bi-temporal mode was skipped.

Usage:
    python scripts/write_phase5_records.py [--dry-run]
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
    # ── Turceni: marginal classic detect, CEMF deferred ───────────────────────
    QuantificationRecord(
        site="turceni",
        scene_id="S2B_MSIL1C_20240927T093029_N0511_R136_T34TFP_20240927T102421",
        acquisition_timestamp="2024-09-27T09:30:29Z",
        plume_centroid_lat=44.101,
        plume_centroid_lon=23.391,
        methodology="CEMF+IME",
        cemf_sensitivity_coeff="4e-7 (Varon 2021 AMT Sec 2.2)",
        mask_source="ch4net_v8_original",
        mask_file=(
            "results_bitemporal/turceni/"
            "original_S2B_MSIL1C_20240927T093029_N0511_R136_T34TFP_20240927T102421.tif"
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
        ch4net_peak_probability=None,
        cloud_cover_quality="clear",
        retrieval_notes=(
            "Phase 5 single-date eval (2024-09-27, T34TFP). "
            "CH4Net S/C=1.740 (classic, vs nearest ctrl). "
            "CFAR adaptive: sc_cfar=1.506 vs thresh_ratio=1.625 (CV=0.158) — NOT triggered. "
            "mu_ctrl=0.260 — high heterogeneous background across Romanian agricultural terrain; "
            "CEMF binary-mask isolation not possible (>30%% of site crop above any threshold). "
            "CEMF quantification deferred pending: (a) winter-2023 reference tile for "
            "bi-temporal suppression of terrain background, or (b) TROPOMI co-location. "
            "Not excluded — site remains candidate for quantification on additional overpasses."
        ),
    ),

    # ── Rovinari: single-date non-detection ───────────────────────────────────
    QuantificationRecord(
        site="rovinari",
        scene_id="S2B_MSIL1C_20240927T093029_N0511_R136_T34TFQ_20240927T102421",
        acquisition_timestamp="2024-09-27T09:30:29Z",
        plume_centroid_lat=44.906,
        plume_centroid_lon=23.147,
        methodology="CEMF+IME",
        cemf_sensitivity_coeff="4e-7 (Varon 2021 AMT Sec 2.2)",
        mask_source="ch4net_v8_original",
        mask_file=(
            "results_bitemporal/rovinari/"
            "original_S2B_MSIL1C_20240927T093029_N0511_R136_T34TFQ_20240927T102421.tif"
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
        ch4net_peak_probability=None,
        cloud_cover_quality="clear",
        retrieval_notes=(
            "Phase 5 single-date eval (2024-09-27, T34TFQ). "
            "CH4Net S/C=0.715 (classic) — below 1.15 threshold. "
            "CFAR adaptive: thresh_ratio=3.108 (CV=0.653) — highly heterogeneous Jiu valley "
            "terrain raises adaptive threshold above any detectable signal. "
            "Single-date non-detection does not exclude this site — Oltenia Energy Complex "
            "operates Rovinari (~1,320 MW lignite) and is a probable methane emitter. "
            "Additional summer overpasses with lower CV terrain conditions recommended."
        ),
    ),

    # ── Maritsa East 2: single-date non-detection ─────────────────────────────
    QuantificationRecord(
        site="maritsa_east_2",
        scene_id="S2B_MSIL1C_20240928T085649_N0511_R007_T35TMG_20240928T113459",
        acquisition_timestamp="2024-09-28T08:56:49Z",
        plume_centroid_lat=42.271,
        plume_centroid_lon=26.068,
        methodology="CEMF+IME",
        cemf_sensitivity_coeff="4e-7 (Varon 2021 AMT Sec 2.2)",
        mask_source="ch4net_v8_original",
        mask_file=(
            "results_bitemporal/maritsa_east_2/"
            "original_S2B_MSIL1C_20240928T085649_N0511_R007_T35TMG_20240928T113459.tif"
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
        ch4net_peak_probability=None,
        cloud_cover_quality="clear",
        retrieval_notes=(
            "Phase 5 single-date eval (2024-09-28, T35TMG). "
            "CH4Net S/C=0.862 (classic) — below 1.15 threshold. "
            "CFAR adaptive: thresh_ratio=2.279 (CV=0.376) — Thracian plain / agricultural "
            "terrain moderate heterogeneity raises adaptive threshold. "
            "Maritsa East complex (AES-NEK / ContourGlobal, ~1,600 MW lignite, Bulgaria) "
            "remains a JRC-listed priority site. "
            "Additional summer overpasses with better atmospheric conditions recommended."
        ),
    ),

    # ── Rybnik: single-date non-detection ─────────────────────────────────────
    # Earlier record had wrong tile (T34UDA 2024-08-15); actual tile is T34UCA 2024-08-29.
    # CH4Net S/C=0.612 (sc_cfar=0.61) — site below control mean on this date.
    QuantificationRecord(
        site="rybnik",
        scene_id="S2A_MSIL1C_20240829T095031_N0511_R079_T34UCA_20240829T115208",
        acquisition_timestamp="2024-08-29T09:50:31Z",
        plume_centroid_lat=50.135,
        plume_centroid_lon=18.522,
        methodology="CEMF+IME",
        cemf_sensitivity_coeff="4e-7 (Varon 2021 AMT Sec 2.2)",
        mask_source="ch4net_v8_original",
        mask_file=(
            "results_bitemporal/rybnik/"
            "original_S2A_MSIL1C_20240829T095031_N0511_R079_T34UCA_20240829T115208.tif"
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
        ch4net_peak_probability=None,
        cloud_cover_quality="clear",
        retrieval_notes=(
            "Single-date eval (2024-08-29, T34UCA). "
            "CH4Net S/C=0.612 (sc_cfar=0.61 vs mu_ctrl=0.000138, CV=1.642) — site below "
            "control mean; clear non-detection on this date. "
            "Previous SITE_REGISTRY entry had wrong tile (T34UDA 2024-08-15, not cached); "
            "corrected to T34UCA 2024-08-29. "
            "KWK Rybnik-Chwałowice (~1,775 MW coal) remains a candidate emitter. "
            "Recommend download of June-July 2024 dates on T34UCA for lower vegetation CV."
        ),
    ),
]


def main():
    parser = argparse.ArgumentParser(
        description="Write Phase 5 + Rybnik provenance records to quantification.json"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print records without writing to disk")
    args = parser.parse_args()

    for rec in RECORDS:
        if args.dry_run:
            print(f"\n{'='*60}")
            print(f"  [dry-run] {rec.site}")
            print(json.dumps(rec.to_dict(), indent=2, default=str))
        else:
            write_quantification_record(rec)
            logger.info(
                "%-20s  cemf_valid=%-5s  flow_rate_kgh=%-10s  notes: %.80s",
                rec.site, rec.cemf_valid, rec.flow_rate_kgh, rec.retrieval_notes,
            )

    if not args.dry_run:
        logger.info("All records written to %s", DEFAULT_QUANT_PATH)
        print("\nProvenance summary:")
        print("  turceni        — marginal S/C=1.74 classic detect; CEMF deferred (high background)")
        print("  rovinari       — single-date non-detection (S/C=0.715)")
        print("  maritsa_east_2 — single-date non-detection (S/C=0.862)")
        print("  rybnik         — single-date non-detection (S/C=0.612, corrected T34UCA)")


if __name__ == "__main__":
    main()
