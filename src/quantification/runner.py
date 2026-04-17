"""
runner.py
=========
Single orchestration entry point for the CEMF+IME+ERA5 quantification pipeline.

Steps 6–10 of the canonical provenance chain (per WS1+WS2 Integration Plan):
  6. Mask selection  — honours per-site bitemporal rule
  7. CEMF retrieval  — src/quantification/cemf.run_cemf()
  8. ERA5 wind       — src/ingestion/era5_client.ERA5Client.get_wind()
  9. IME inversion   — src/quantification/ime.CEMFIntegratedMassEnhancement.estimate_from_cemf()
 10. Canonical write — src/quantification/canonical_writer.write_quantification_record()

Usage (batch):
    from src.quantification.runner import run_quantification
    record = run_quantification(site_cfg, dry_run=False)

Usage (notebook):
    from src.quantification.runner import run_quantification
    record = run_quantification(site_cfg)   # ERA5 fetched automatically

site_cfg is a plain dict with the fields described in SiteCfg below.
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from src.quantification.cemf import run_cemf, downsample_mask, CEMFResult
from src.quantification.ime import CEMFIntegratedMassEnhancement
from src.ingestion.era5_client import ERA5Client
from src.quantification.canonical_writer import (
    QuantificationRecord,
    write_quantification_record,
    DEFAULT_QUANT_PATH,
)

logger = logging.getLogger(__name__)

# ── Per-site bitemporal rule ──────────────────────────────────────────────────
# Sites listed in USE_BITEMPORAL must have their CEMF run on the BT-differenced
# mask. All others use the original CH4Net mask.
#
# Rationale: BT differencing is only appropriate where the primary detection
# signal is a genuine spectral change between dates (e.g. seasonal vegetation
# contrast suppresses the gas signal in original mask). For industrial sites
# where CH4Net fires on a stable absorption signal, BT typically kills the
# real plume (confirmed empirically for Bełchatów: S/C 27.3→1.94).
#
# Single source of truth — risk_model.py and stress_test.py must not
# independently re-implement this logic.

USE_BITEMPORAL: dict[str, bool] = {
    "lippendorf": True,   # BT proves it's terrain (S/C 155→0.19)
    "groningen":  True,   # coastal vegetation contrast; CFAR handles gas signal
}

SKIP_BITEMPORAL: dict[str, bool] = {
    "belchatow":  True,   # BT destroys real signal (Gap 8)
    "neurath":    True,
    "weisweiler": True,
    "rybnik":     True,
    "maasvlakte": True,
}

# Default for any site not listed: skip bitemporal (conservative for industrial sites)
_BITEMPORAL_DEFAULT = False


def _use_bitemporal(site_name: str) -> bool:
    if site_name in USE_BITEMPORAL:
        return True
    if site_name in SKIP_BITEMPORAL:
        return False
    return _BITEMPORAL_DEFAULT


# ── Site config ───────────────────────────────────────────────────────────────

@dataclass
class SiteCfg:
    """
    Configuration for one CEMF+IME run.  All path fields accept str or Path.
    """
    site: str                          # canonical slug (e.g. "neurath")
    scene_id: str                      # Sentinel-2 product ID
    acquisition_timestamp: str         # ISO 8601 (e.g. "2024-06-25T10:36:31Z")
    lat: float                         # plume centroid latitude
    lon: float                         # plume centroid longitude

    # Band arrays — load externally and pass in (keeps runner I/O-agnostic)
    b11: np.ndarray = field(default=None, repr=False)
    b12: np.ndarray = field(default=None, repr=False)
    mask_original: np.ndarray = field(default=None, repr=False)   # CH4Net 10m mask
    mask_bitemporal: np.ndarray = field(default=None, repr=False)  # BT-differenced mask (optional)

    # Optional overrides
    wind_override: Optional[dict] = None   # pre-fetched wind dict (bypasses ERA5)
    era5_hour: str = "12:00"
    cloud_cover_quality: str = "clear"
    ch4net_peak_probability: Optional[float] = None
    tropomi_confirm: bool = False
    retrieval_notes: str = ""

    # Output location
    quant_path: str = DEFAULT_QUANT_PATH
    mask_file: Optional[str] = None    # path to mask TIFF for provenance


def run_quantification(
    site_cfg: SiteCfg,
    dry_run: bool = False,
    era5_client: Optional[ERA5Client] = None,
) -> QuantificationRecord:
    """
    Orchestrate one full CEMF+IME+ERA5 quantification run.

    Args:
        site_cfg:    SiteCfg for this site/scene
        dry_run:     if True, skip writing to disk (used to verify schema)
        era5_client: pre-instantiated ERA5Client (created if None)

    Returns:
        QuantificationRecord with all fields populated
    """
    site = site_cfg.site
    logger.info("run_quantification: site=%s scene=%s", site, site_cfg.scene_id)

    # ── Step 6: mask selection ────────────────────────────────────────────────
    use_bt = _use_bitemporal(site)
    if use_bt:
        if site_cfg.mask_bitemporal is None:
            logger.warning("%s: bitemporal rule=True but mask_bitemporal is None; falling back to original", site)
            mask = site_cfg.mask_original
            mask_source = "ch4net_v8_original"
        else:
            mask = site_cfg.mask_bitemporal
            mask_source = "ch4net_v8_bitemporal"
    else:
        mask = site_cfg.mask_original
        mask_source = "ch4net_v8_original"

    if mask is None:
        raise ValueError(f"run_quantification: no mask available for site '{site}'")

    # Downsample from 10m to 20m for SWIR alignment
    mask_20m = downsample_mask(mask)

    # ── Step 7: CEMF retrieval ────────────────────────────────────────────────
    b11_20m = site_cfg.b11[::2, ::2] if site_cfg.b11 is not None and site_cfg.b11.shape[0] > mask_20m.shape[0] else site_cfg.b11
    b12_20m = site_cfg.b12[::2, ::2] if site_cfg.b12 is not None and site_cfg.b12.shape[0] > mask_20m.shape[0] else site_cfg.b12

    cemf_result: CEMFResult = run_cemf(
        b11=b11_20m,
        b12=b12_20m,
        mask=mask_20m,
        scene_id=site_cfg.scene_id,
        timestamp=site_cfg.acquisition_timestamp,
    )

    if not cemf_result.retrieval_valid:
        logger.warning("%s: CEMF retrieval invalid — %s", site, cemf_result.warning)

    # ── Step 8: ERA5 wind ─────────────────────────────────────────────────────
    if site_cfg.wind_override is not None:
        wind = site_cfg.wind_override
        logger.info("%s: using pre-fetched wind %.2f m/s (%s)", site, wind["wind_speed_ms"], wind["wind_source"])
    else:
        client = era5_client or ERA5Client()
        date_str = site_cfg.acquisition_timestamp[:10]  # "YYYY-MM-DD"
        wind = client.get_wind(site_cfg.lat, site_cfg.lon, date_str, hour=site_cfg.era5_hour)
        logger.info("%s: ERA5 wind %.2f m/s from %s", site, wind["wind_speed_ms"], wind["wind_source"])

    # ── Step 9: IME inversion ─────────────────────────────────────────────────
    ime = CEMFIntegratedMassEnhancement()
    qr = ime.estimate_from_cemf(cemf_result, wind_speed_ms=wind["wind_speed_ms"], wind_source=wind["wind_source"])

    # ── Step 10: Build canonical record ───────────────────────────────────────
    uncertainty_pct = 30  # Phase 3 will expose this as UNCERTAINTY_PCT constant

    notes = site_cfg.retrieval_notes or (
        f"Wind source: {wind['wind_source']}; "
        f"mask: {mask_source}; "
        f"BT rule: {use_bt}."
    )

    record = QuantificationRecord(
        site=site,
        scene_id=site_cfg.scene_id,
        acquisition_timestamp=site_cfg.acquisition_timestamp,
        plume_centroid_lat=site_cfg.lat,
        plume_centroid_lon=site_cfg.lon,
        methodology="CEMF+IME",
        cemf_sensitivity_coeff="4e-7 (Varon 2021 AMT Sec 2.2)",
        mask_source=mask_source,
        mask_file=site_cfg.mask_file,
        n_plume_pixels=cemf_result.n_plume_pixels,
        total_mass_kg=round(cemf_result.total_mass_kg, 4) if cemf_result.total_mass_kg else None,
        plume_length_m=round(qr.plume_length_m, 1) if qr.plume_length_m else None,
        wind_speed_ms=wind["wind_speed_ms"],
        wind_dir_deg=wind.get("wind_dir_deg"),
        wind_source=wind["wind_source"],
        era5_u_ms=wind.get("era5_u_ms"),
        era5_v_ms=wind.get("era5_v_ms"),
        flow_rate_kgh=qr.flow_rate_kgh,
        flow_rate_lower_kgh=round(qr.flow_rate_kgh * (1 - uncertainty_pct / 100), 2),
        flow_rate_upper_kgh=round(qr.flow_rate_kgh * (1 + uncertainty_pct / 100), 2),
        uncertainty_pct=uncertainty_pct,
        annual_tonnes_if_continuous=round(qr.flow_rate_kgh * 8760 / 1000, 4) if qr.flow_rate_kgh else None,
        cemf_valid=cemf_result.retrieval_valid,
        excluded=False,
        exclusion_reason=None,
        tropomi_confirm=site_cfg.tropomi_confirm,
        ch4net_peak_probability=site_cfg.ch4net_peak_probability,
        cloud_cover_quality=site_cfg.cloud_cover_quality,
        retrieval_notes=notes,
    )

    # ── Write to disk ─────────────────────────────────────────────────────────
    if dry_run:
        logger.info("%s: dry_run=True — skipping disk write", site)
    else:
        write_quantification_record(record, path=site_cfg.quant_path)

    return record
