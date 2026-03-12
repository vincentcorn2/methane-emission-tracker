"""
Time-series emission record logger.
Appends one JSON record per detection to a log file for
downstream climate scenario testing.
"""
import json
import os
from datetime import datetime
from typing import Optional


def log_emission_record(
    scene_id: str,
    timestamp: str,
    plume_centroid_lat: float,
    plume_centroid_lon: float,
    flow_rate_kgh: float,
    flow_rate_lower_kgh: float,
    flow_rate_upper_kgh: float,
    ch4net_peak_probability: float,
    cloud_cover_quality: str,
    wind_speed_ms: float,
    wind_source: str,
    n_plume_pixels: int,
    total_mass_kg: float,
    annual_tonnes: Optional[float] = None,
    ira_liability_usd: Optional[float] = None,
    log_path: str = "results/emission_timeseries.jsonl",
):
    """
    Append one emission record to the time-series log.
    Uses JSONL format (one JSON object per line) for easy streaming.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    record = {
        "logged_at": datetime.utcnow().isoformat(),
        "scene_id": scene_id,
        "acquisition_timestamp": timestamp,
        "plume_centroid_lat": plume_centroid_lat,
        "plume_centroid_lon": plume_centroid_lon,
        "q_cemf_kg_per_hour": flow_rate_kgh,
        "q_lower_kg_per_hour": flow_rate_lower_kgh,
        "q_upper_kg_per_hour": flow_rate_upper_kgh,
        "uncertainty_pct": 40,
        "ch4net_peak_probability": ch4net_peak_probability,
        "cloud_cover_quality": cloud_cover_quality,
        "wind_speed_ms": wind_speed_ms,
        "wind_source": wind_source,
        "n_plume_pixels": n_plume_pixels,
        "pixel_area_m2": 400,
        "total_mass_kg": total_mass_kg,
        "annual_tonnes_if_continuous": annual_tonnes,
        "ira_waste_charge_usd_2026": ira_liability_usd,
        "methodology": "CEMF+IME",
        "sensor": "Sentinel-2 L1C",
        "retrieval_notes": "Wind source: " + wind_source,
    }

    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")

    return record
