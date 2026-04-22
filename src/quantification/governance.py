"""
src/quantification/governance.py
==================================
Input-degradation governance controls for the CH4Net quantification pipeline.

SR 11-7 / MRM framing (Pillar 2 — Ongoing Monitoring and Process Verification):
  When any element of the pipeline runs on a fallback rather than a primary
  input, the record must be:
    (i)  flagged with a structured degradation tag
    (ii) reported with automatically widened uncertainty
    (iii) available for exclusion from downstream aggregate statistics

This pattern is identical to "automatic conservative response under input
uncertainty" controls in bank VaR systems — e.g., a VaR model that widens
its interval whenever the volatility surface is from a stale source.

Degradation sources tracked
───────────────────────────
  WIND_FALLBACK          Climatological 3.5 m/s used instead of ERA5 reanalysis.
                         Base penalty: σ_wind_pct inflated from 20% → 50%.
                         Overall combined σ widens by ~60% (quadrature).

  BITEMPORAL_MISSING     No winter-reference image available; bitemporal
                         differencing suppressed. Terrain artifacts uncontrolled.
                         Penalty: +15 pp additive on mask uncertainty.

  CLOUD_FRACTION_HIGH    Scene cloud fraction > 15% in tile footprint.
                         Penalty: quantification flagged for review; no
                         automatic inflation (cloud coverage is binary: scene
                         is processed or not, not a smooth degradation).

  SOLAR_ZENITH_EXTREME   Solar zenith angle > 70° at acquisition time.
                         Penalty: +5 pp on sensitivity coefficient uncertainty
                         (atmospheric path length increases near horizon).

  ERA5_STALE             ERA5 query fell back to a different hour ± 6h from
                         acquisition. Penalty: +10 pp on wind uncertainty.

Pre-calibrated penalty factors
──────────────────────────────
  WIND_FALLBACK:     σ_wind becomes 50% (vs. 20% ERA5 primary)
  BITEMPORAL_MISSING: σ_mask  += 15 pp
  SOLAR_ZENITH_EXTREME: σ_coeff += 5 pp
  ERA5_STALE:        σ_wind  += 10 pp (absolute)

These values are documented here and in the uncertainty decomposition report.
They are NOT derived from data at this stage — they are expert-conservative
priors pending calibration against controlled-release experiments (WS5).
This is identical to how regulatory capital models treat unverified risk
factors pending internal model approval: assume worst-case until calibrated.

Usage
─────
    from src.quantification.governance import assess_degradation, apply_governance

    flags = assess_degradation(record)
    updated_record = apply_governance(record, flags)

    # Or in one call:
    updated_record = apply_governance_to_record(record)
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional

logger = logging.getLogger(__name__)

# ── Penalty calibration table ─────────────────────────────────────────────────
# Units: percentage points (pp) additive to the relevant σ_X source.
# "additive" means σ_new = σ_baseline + PENALTY (not multiplicative).

PENALTY_WIND_FALLBACK_PCT          = 30.0   # σ_wind 20% → 50%
PENALTY_BITEMPORAL_MISSING_PCT     = 15.0   # σ_mask  += 15 pp
PENALTY_SOLAR_ZENITH_EXTREME_PCT   =  5.0   # σ_coeff += 5 pp
PENALTY_ERA5_STALE_PCT             = 10.0   # σ_wind  += 10 pp

# σ baselines (from uncertainty.py / Varon 2021)
SIGMA_WIND_ERA5   = 20.0    # % when ERA5 is primary
SIGMA_WIND_FALLBACK = 50.0  # % when climatological fallback is used
SIGMA_COEFF_BASE  = 15.0    # % sensitivity coefficient (fixed)

# Solar zenith threshold (degrees)
SOLAR_ZENITH_EXTREME_DEG = 70.0

# Cloud fraction threshold
CLOUD_FRACTION_HIGH_PCT = 15.0

# ERA5 staleness threshold (hours)
ERA5_STALE_HOURS = 4.0


@dataclass
class DegradationFlags:
    """Structured record of input-quality conditions for one quantification record."""

    wind_fallback: bool = False
    """Climatological wind (3.5 m/s) used instead of ERA5 reanalysis."""

    bitemporal_missing: bool = False
    """Bitemporal reference scene absent; terrain artifacts uncontrolled."""

    cloud_fraction_high: bool = False
    """Scene cloud fraction exceeds threshold; quantification for review."""

    solar_zenith_extreme: bool = False
    """Solar zenith angle > 70° at acquisition; path-length uncertainty elevated."""

    era5_stale: bool = False
    """ERA5 wind from a non-nearest hour (>4h offset from acquisition)."""

    # Penalty magnitudes actually applied (pp)
    delta_sigma_wind_pp:  float = 0.0
    delta_sigma_mask_pp:  float = 0.0
    delta_sigma_coeff_pp: float = 0.0

    @property
    def any_flag(self) -> bool:
        return (self.wind_fallback or self.bitemporal_missing or
                self.cloud_fraction_high or self.solar_zenith_extreme or
                self.era5_stale)

    @property
    def flag_names(self) -> list[str]:
        names = []
        if self.wind_fallback:       names.append("WIND_FALLBACK")
        if self.bitemporal_missing:  names.append("BITEMPORAL_MISSING")
        if self.cloud_fraction_high: names.append("CLOUD_FRACTION_HIGH")
        if self.solar_zenith_extreme: names.append("SOLAR_ZENITH_EXTREME")
        if self.era5_stale:          names.append("ERA5_STALE")
        return names

    def to_dict(self) -> dict:
        return {
            "flags":                  self.flag_names,
            "any_degradation":        self.any_flag,
            "delta_sigma_wind_pp":    self.delta_sigma_wind_pp,
            "delta_sigma_mask_pp":    self.delta_sigma_mask_pp,
            "delta_sigma_coeff_pp":   self.delta_sigma_coeff_pp,
            "penalty_reference":      "governance.py — pre-calibrated conservative prior",
        }


def assess_degradation(record: dict) -> DegradationFlags:
    """
    Inspect a quantification record and return a DegradationFlags instance.

    Args:
        record: A quantification.json record (dict).

    Returns:
        DegradationFlags populated from the record fields.
    """
    flags = DegradationFlags()

    # ── Wind source ────────────────────────────────────────────────────────────
    wind_source = record.get("wind_source", "") or ""
    if "fallback" in wind_source.lower() or "climatological" in wind_source.lower():
        flags.wind_fallback = True
        flags.delta_sigma_wind_pp = PENALTY_WIND_FALLBACK_PCT  # additive delta: σ_wind 20% → 50%
        logger.debug("Flagged WIND_FALLBACK for scene %s", record.get("scene_id", "?")[:30])

    # ERA5 stale: wind_source is "ERA5_reanalysis" but era5_hour_offset is large
    era5_offset = record.get("era5_hour_offset_h")
    if era5_offset is not None and abs(era5_offset) > ERA5_STALE_HOURS:
        flags.era5_stale = True
        flags.delta_sigma_wind_pp = max(flags.delta_sigma_wind_pp,
                                        PENALTY_ERA5_STALE_PCT)
        logger.debug("Flagged ERA5_STALE (offset=%.1fh) for scene %s",
                     era5_offset, record.get("scene_id", "?")[:30])

    # ── Mask source / bitemporal ───────────────────────────────────────────────
    mask_source = record.get("mask_source", "") or ""
    # If the site is in the skip_bitemporal list and the mask is original, that is
    # intentional and does NOT trigger BITEMPORAL_MISSING.
    # BITEMPORAL_MISSING fires when bitemporal was EXPECTED but not available.
    if record.get("bitemporal_missing", False):
        flags.bitemporal_missing = True
        flags.delta_sigma_mask_pp += PENALTY_BITEMPORAL_MISSING_PCT

    # ── Cloud fraction ─────────────────────────────────────────────────────────
    cloud = record.get("cloud_cover_pct") or record.get("cloud_fraction_pct")
    if cloud is not None and float(cloud) > CLOUD_FRACTION_HIGH_PCT:
        flags.cloud_fraction_high = True
        logger.debug("Flagged CLOUD_FRACTION_HIGH (%.1f%%) for scene %s",
                     cloud, record.get("scene_id", "?")[:30])

    # Cloud quality field (legacy string encoding)
    cloud_quality = record.get("cloud_cover_quality", "") or ""
    if cloud_quality in ("high_cloud", "partly_cloudy", "cloudy"):
        flags.cloud_fraction_high = True

    # ── Solar zenith ───────────────────────────────────────────────────────────
    sza = record.get("solar_zenith_deg")
    if sza is not None and float(sza) > SOLAR_ZENITH_EXTREME_DEG:
        flags.solar_zenith_extreme = True
        flags.delta_sigma_coeff_pp += PENALTY_SOLAR_ZENITH_EXTREME_PCT
        logger.debug("Flagged SOLAR_ZENITH_EXTREME (%.1f°) for scene %s",
                     sza, record.get("scene_id", "?")[:30])

    return flags


def inflated_uncertainty(
    sigma_wind_pct:  float,
    sigma_coeff_pct: float,
    sigma_mask_pct:  float,
    sigma_bg_pct:    float,
    flags: DegradationFlags,
    n_mc: int = 10_000,
    rng_seed: int = 99,
) -> dict:
    """
    Recompute combined uncertainty with governance penalty applied.

    The base σ values are inflated by the penalty deltas, then re-propagated
    through 10k Monte Carlo samples.

    Returns a dict with the same schema as monte_carlo_combined() in
    uncertainty_decomposition.py, plus governance_applied=True.
    """
    import numpy as np

    # Apply penalties
    eff_wind  = sigma_wind_pct  + flags.delta_sigma_wind_pp
    eff_coeff = sigma_coeff_pct + flags.delta_sigma_coeff_pp
    eff_mask  = sigma_mask_pct  + flags.delta_sigma_mask_pp
    eff_bg    = sigma_bg_pct    # no governance penalty on background

    rng = np.random.default_rng(rng_seed)

    def safe(v):
        return 0.0 if (v is None or np.isnan(v)) else v / 100.0

    f_w = np.clip(rng.normal(1.0, safe(eff_wind),  n_mc), 0.01, None)
    f_c = np.clip(rng.normal(1.0, safe(eff_coeff), n_mc), 0.01, None)
    f_m = np.clip(rng.normal(1.0, safe(eff_mask),  n_mc), 0.01, None)
    f_b = np.clip(rng.normal(1.0, safe(eff_bg),    n_mc), 0.01, None)

    # We don't have the base_flow here — return relative stats only
    # Caller must multiply by base_flow to get absolute CI
    samples_relative = f_w * f_c * f_m * f_b  # relative to base_flow = 1

    mean_r  = float(np.mean(samples_relative))
    std_r   = float(np.std(samples_relative, ddof=1))
    p5_r    = float(np.percentile(samples_relative, 5))
    p95_r   = float(np.percentile(samples_relative, 95))

    sigma_mc_pct = round(100.0 * std_r / mean_r, 1) if mean_r > 0 else float("nan")
    quad_pct     = round(100.0 * (safe(eff_wind)**2 + safe(eff_coeff)**2 +
                                   safe(eff_mask)**2 + safe(eff_bg)**2)**0.5, 1)

    return {
        "governance_applied":        True,
        "effective_sigma_wind_pct":  round(eff_wind,  1),
        "effective_sigma_coeff_pct": round(eff_coeff, 1),
        "effective_sigma_mask_pct":  round(eff_mask,  1),
        "effective_sigma_bg_pct":    round(eff_bg,    1),
        "sigma_combined_mc_pct":     sigma_mc_pct,
        "sigma_combined_quad_pct":   quad_pct,
        "p5_relative":               round(p5_r,  4),
        "p95_relative":              round(p95_r, 4),
        "n_mc_samples":              n_mc,
        "penalty_reference":         "governance.py — pre-calibrated conservative prior",
    }


def apply_governance(record: dict, flags: DegradationFlags) -> dict:
    """
    Add governance fields to a quantification record in-place.

    Adds:
      record["input_degradation"]          — DegradationFlags.to_dict()
      record["governance_sigma_inflated"]  — True/False
      record["exclude_from_aggregates"]    — True if cloud flag set

    If uncertainty_decomposition is present, also adds
      record["uncertainty_decomposition"]["governance_inflated"] block.

    Returns the (mutated) record.
    """
    record["input_degradation"] = flags.to_dict()
    record["governance_sigma_inflated"] = flags.any_flag

    # Cloud-flagged records should not contribute to aggregate statistics
    record["exclude_from_aggregates"] = flags.cloud_fraction_high

    # If uncertainty decomposition is available, add inflated version
    ud = record.get("uncertainty_decomposition")
    if ud and flags.any_flag:
        # Pull base σ values (use defaults if not present)
        base_wind  = ud.get("sigma_wind_pct",       20.0)
        base_coeff = ud.get("sigma_coeff_pct",       15.0)
        base_mask  = ud.get("sigma_mask_pct",         0.0) or 0.0
        base_bg    = ud.get("sigma_background_pct",   0.0) or 0.0

        inflated = inflated_uncertainty(
            base_wind, base_coeff, base_mask, base_bg, flags
        )
        ud["governance_inflated"] = inflated

        # Widen the 90% CI using the relative factors
        base_flow = record.get("flow_rate_kgh") or record.get("reported_flow_kgh")
        if base_flow and base_flow > 0:
            ud["governance_inflated"]["ci_90_low_kgh"]  = round(
                base_flow * inflated["p5_relative"], 2)
            ud["governance_inflated"]["ci_90_high_kgh"] = round(
                base_flow * inflated["p95_relative"], 2)

    if flags.any_flag:
        logger.info("Governance flags applied to scene %s: %s",
                    record.get("scene_id", "?")[:40],
                    ", ".join(flags.flag_names))
    return record


def apply_governance_to_record(record: dict) -> dict:
    """Convenience wrapper: assess + apply in one call."""
    flags = assess_degradation(record)
    return apply_governance(record, flags)


def apply_governance_to_all(records: list[dict]) -> tuple[list[dict], dict]:
    """
    Apply governance assessment to every record in a list.

    Returns:
        (updated_records, summary_dict) where summary_dict has per-flag counts.
    """
    summary = {
        "total":                0,
        "any_flag":             0,
        "wind_fallback":        0,
        "bitemporal_missing":   0,
        "cloud_fraction_high":  0,
        "solar_zenith_extreme": 0,
        "era5_stale":           0,
    }
    updated = []
    for rec in records:
        flags = assess_degradation(rec)
        apply_governance(rec, flags)
        summary["total"] += 1
        if flags.any_flag:          summary["any_flag"] += 1
        if flags.wind_fallback:     summary["wind_fallback"] += 1
        if flags.bitemporal_missing: summary["bitemporal_missing"] += 1
        if flags.cloud_fraction_high: summary["cloud_fraction_high"] += 1
        if flags.solar_zenith_extreme: summary["solar_zenith_extreme"] += 1
        if flags.era5_stale:        summary["era5_stale"] += 1
        updated.append(rec)
    return updated, summary
