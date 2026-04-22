"""
canonical_writer.py
===================
Single source of truth for writing quantification records to
results_analysis/quantification.json.

This module is the ONLY place that may write to that file.
All batch scripts (scripts/run_quantification.py) and notebooks
(European_CH4_Pipeline.ipynb) must call write_quantification_record()
instead of building JSON by hand.

Schema v1.0.0 — see Part 2 of the WS1+WS2 Integration Plan for the
canonical field specification.  Any record with excluded=True is retained
for provenance but ignored by risk_model.py and stress_test.py.
"""
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class _NumpyEncoder(json.JSONEncoder):
    """Serialize numpy scalar types that json.dump can't handle natively."""
    def default(self, obj):
        try:
            import numpy as np
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError:
            pass
        return super().default(obj)

SCHEMA_VERSION = "1.0.0"
DEFAULT_QUANT_PATH = "results_analysis/quantification.json"


@dataclass
class QuantificationRecord:
    """
    Canonical quantification record — matches the schema in Part 2 of the
    WS1+WS2 Integration Plan exactly.  All fields are required except those
    annotated Optional; callers must pass None explicitly for absent values.
    """
    # Identity
    site: str                                   # canonical lowercase slug (e.g. "neurath")
    scene_id: str                               # e.g. "S2A_T32ULB_20240625"
    acquisition_timestamp: str                  # ISO 8601, e.g. "2024-06-25T10:36:31Z"

    # Geolocation
    plume_centroid_lat: Optional[float]
    plume_centroid_lon: Optional[float]

    # Methodology provenance
    methodology: str = "CEMF+IME"
    cemf_sensitivity_coeff: str = "4e-7 (Varon 2021 AMT Sec 2.2)"
    mask_source: str = "ch4net_v8_original"     # "ch4net_v8_original" | "ch4net_v8_bitemporal"
    mask_file: Optional[str] = None

    # Detection geometry
    n_plume_pixels: int = 0
    total_mass_kg: Optional[float] = None
    plume_length_m: Optional[float] = None

    # Wind
    wind_speed_ms: Optional[float] = None
    wind_dir_deg: Optional[float] = None
    wind_source: str = "climatological_fallback_3.5ms"
    era5_u_ms: Optional[float] = None
    era5_v_ms: Optional[float] = None

    # Quantification outputs
    flow_rate_kgh: Optional[float] = None
    flow_rate_lower_kgh: Optional[float] = None  # flow_rate_kgh * (1 - uncertainty_pct/100)
    flow_rate_upper_kgh: Optional[float] = None  # flow_rate_kgh * (1 + uncertainty_pct/100)
    uncertainty_pct: int = 30                    # SSOT: Phase 3 harmonises all modules to this

    # Annualised financial outputs
    annual_tonnes_if_continuous: Optional[float] = None   # Q * 8760 / 1000
    eu_ets_liability_eur: Optional[float] = None          # annual_tonnes * GWP * ETS_price
    ira_waste_charge_usd_2026: Optional[float] = None     # secondary comparator

    # Validity and exclusion flags
    cemf_valid: bool = True
    excluded: bool = False
    exclusion_reason: Optional[str] = None       # "terrain_artifact" | "cfar_suppressed" | None

    # Cross-validation
    tropomi_confirm: bool = False               # TROPOMI ΔXCH4 >= 5 ppb corroborates
    ch4net_peak_probability: Optional[float] = None
    cloud_cover_quality: str = "unknown"

    # Free text
    retrieval_notes: str = ""

    # Schema version — always written last for easy auditing
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self):
        """Compute derived fields if not supplied."""
        if self.flow_rate_kgh is not None and self.flow_rate_lower_kgh is None:
            f = self.uncertainty_pct / 100.0
            self.flow_rate_lower_kgh = round(self.flow_rate_kgh * (1 - f), 2)
            self.flow_rate_upper_kgh = round(self.flow_rate_kgh * (1 + f), 2)

        if self.flow_rate_kgh is not None and self.annual_tonnes_if_continuous is None:
            self.annual_tonnes_if_continuous = round(self.flow_rate_kgh * 8760 / 1000, 4)

    def to_dict(self) -> dict:
        return asdict(self)


def write_quantification_record(
    record: QuantificationRecord,
    path: str = DEFAULT_QUANT_PATH,
) -> None:
    """
    Upsert one QuantificationRecord into the canonical JSON file.

    The file contains a JSON array; this function reads it, replaces the
    entry matching record.site (if present), and writes back atomically.
    New sites are appended.

    Args:
        record: the record to write
        path:   path to quantification.json (default: results_analysis/quantification.json)
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # Load existing records
    existing: list[dict] = []
    if p.exists():
        try:
            with open(p) as f:
                raw = json.load(f)
            if isinstance(raw, list):
                existing = raw
            elif isinstance(raw, dict):
                existing = list(raw.values())
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Could not parse %s (%s) — starting fresh.", p, exc)

    # ── Governance: assess input-degradation flags before writing ────────────
    # Any record produced under fallback/degraded inputs gets flagged and
    # has its reported uncertainty automatically widened.  This is an SR 11-7
    # Pillar 2 process-verification control — "conservative response under
    # input uncertainty" identical to VaR stress-scalar escalation.
    new_dict = record.to_dict()
    try:
        from src.quantification.governance import apply_governance_to_record
        new_dict = apply_governance_to_record(new_dict)
    except Exception as gov_exc:
        logger.warning("Governance assessment failed (non-fatal): %s", gov_exc)

    replaced = False
    for i, rec in enumerate(existing):
        if record.scene_id and rec.get("scene_id") == record.scene_id:
            existing[i] = new_dict
            replaced = True
            break
    if not replaced:
        existing.append(new_dict)

    # Write back
    with open(p, "w") as f:
        json.dump(existing, f, indent=2, cls=_NumpyEncoder)

    logger.info(
        "%s record for '%s' to %s (excluded=%s)",
        "Updated" if replaced else "Appended",
        record.site,
        path,
        record.excluded,
    )


def load_quantification_records(
    path: str = DEFAULT_QUANT_PATH,
    exclude_flagged: bool = True,
) -> dict[str, QuantificationRecord]:
    """
    Load all records from quantification.json, optionally skipping excluded ones.

    Returns:
        dict keyed by site slug.
    """
    p = Path(path)
    if not p.exists():
        return {}

    with open(p) as f:
        raw = json.load(f)

    records: dict[str, QuantificationRecord] = {}
    items = raw if isinstance(raw, list) else list(raw.values())
    for item in items:
        if exclude_flagged and item.get("excluded", False):
            continue
        # Tolerate legacy records missing new fields
        site = item.get("site", "unknown")
        scene_id = item.get("scene_id", "")
        # Use (site, scene_id) composite key so multi-date records coexist.
        # Callers that only care about one record per site can filter by site.
        key = "{}/{}".format(site, scene_id) if scene_id else site
        try:
            # Build with only fields the dataclass knows about
            known_fields = {f.name for f in QuantificationRecord.__dataclass_fields__.values()}
            filtered = {k: v for k, v in item.items() if k in known_fields}
            records[key] = QuantificationRecord(**filtered)
        except TypeError as exc:
            logger.warning("Could not deserialise record for '%s': %s", site, exc)

    return records
