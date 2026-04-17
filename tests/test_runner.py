"""
tests/test_runner.py
====================
Unit tests for src/quantification/runner.py and canonical_writer.py.

Tests verify:
  - Deterministic QuantificationRecord from synthetic B11/B12/mask input
  - Per-site bitemporal rule (Lippendorf → BT; Bełchatów → original)
  - canonical_writer upsert/append semantics
  - EXCLUDED_SITES guard in RiskModel and StressTestEngine
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.quantification.runner import (
    SiteCfg,
    run_quantification,
    _use_bitemporal,
    USE_BITEMPORAL,
    SKIP_BITEMPORAL,
)
from src.quantification.canonical_writer import (
    QuantificationRecord,
    write_quantification_record,
    load_quantification_records,
    SCHEMA_VERSION,
)
from src.api.risk_model import EXCLUDED_SITES


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_synthetic_scene(h: int = 60, w: int = 60, seed: int = 42):
    """
    Generate a deterministic synthetic Sentinel-2 scene with a small plume.
    """
    rng = np.random.default_rng(seed)
    b11 = rng.uniform(0.05, 0.15, (h, w)).astype(np.float32)
    b12 = b11 * 0.85  # Baseline: B12 ~ 85% of B11

    # Synthetic plume: 10x10 block with enhanced B12 absorption (methane signal)
    plume_mask = np.zeros((h, w), dtype=np.float32)
    plume_mask[20:30, 20:30] = 1.0
    b12[20:30, 20:30] *= 0.90  # 10% additional absorption in plume region

    return b11, b12, plume_mask


@pytest.fixture
def synthetic_scene():
    return _make_synthetic_scene()


@pytest.fixture
def tmp_quant_path(tmp_path):
    return str(tmp_path / "quantification.json")


# ── Bitemporal rule tests ──────────────────────────────────────────────────────

def test_use_bitemporal_lippendorf():
    """Lippendorf must always use bitemporal mask (terrain artefact confirmed)."""
    assert _use_bitemporal("lippendorf") is True


def test_use_bitemporal_groningen():
    """Groningen must always use bitemporal mask."""
    assert _use_bitemporal("groningen") is True


def test_skip_bitemporal_belchatow():
    """Bełchatów must always use original mask (BT kills real signal)."""
    assert _use_bitemporal("belchatow") is False


def test_skip_bitemporal_neurath():
    """Neurath must use original mask (industrial site, no seasonal contrast)."""
    assert _use_bitemporal("neurath") is False


def test_skip_bitemporal_weisweiler():
    assert _use_bitemporal("weisweiler") is False


def test_default_bitemporal_unknown_site():
    """Unknown sites default to original mask (conservative)."""
    assert _use_bitemporal("some_new_site_xyz") is False


# ── QuantificationRecord tests ────────────────────────────────────────────────

def test_quantification_record_derives_bounds():
    """__post_init__ must auto-derive lower/upper bounds from uncertainty_pct."""
    r = QuantificationRecord(
        site="test", scene_id="S2A_X_20240101", acquisition_timestamp="2024-01-01T10:00:00Z",
        plume_centroid_lat=51.0, plume_centroid_lon=6.5,
        flow_rate_kgh=400.0, uncertainty_pct=30,
    )
    assert r.flow_rate_lower_kgh == pytest.approx(400.0 * 0.70, rel=1e-3)
    assert r.flow_rate_upper_kgh == pytest.approx(400.0 * 1.30, rel=1e-3)


def test_quantification_record_schema_version():
    """All records must carry the current schema version."""
    r = QuantificationRecord(
        site="test", scene_id="S2A_X_20240101", acquisition_timestamp="2024-01-01T10:00:00Z",
        plume_centroid_lat=51.0, plume_centroid_lon=6.5,
    )
    assert r.schema_version == SCHEMA_VERSION


def test_quantification_record_annual_tonnes():
    """annual_tonnes_if_continuous = flow_rate_kgh * 8760 / 1000."""
    r = QuantificationRecord(
        site="test", scene_id="S2A_X_20240101", acquisition_timestamp="2024-01-01T10:00:00Z",
        plume_centroid_lat=51.0, plume_centroid_lon=6.5,
        flow_rate_kgh=500.0,
    )
    expected = 500.0 * 8760 / 1000
    assert r.annual_tonnes_if_continuous == pytest.approx(expected, rel=1e-3)


# ── canonical_writer upsert tests ─────────────────────────────────────────────

def test_write_new_record(tmp_quant_path):
    """Writing to a non-existent file creates it with one record."""
    r = QuantificationRecord(
        site="alpha", scene_id="S2A_X_20240101", acquisition_timestamp="2024-01-01T10:00:00Z",
        plume_centroid_lat=51.0, plume_centroid_lon=6.5,
        flow_rate_kgh=300.0,
    )
    write_quantification_record(r, path=tmp_quant_path)
    with open(tmp_quant_path) as f:
        data = json.load(f)
    assert len(data) == 1
    assert data[0]["site"] == "alpha"
    assert data[0]["flow_rate_kgh"] == pytest.approx(300.0)


def test_upsert_replaces_existing(tmp_quant_path):
    """Writing two records with the same site slug updates rather than appends."""
    r1 = QuantificationRecord(
        site="beta", scene_id="S2A_X_20240101", acquisition_timestamp="2024-01-01T10:00:00Z",
        plume_centroid_lat=51.0, plume_centroid_lon=6.5,
        flow_rate_kgh=100.0,
    )
    r2 = QuantificationRecord(
        site="beta", scene_id="S2A_X_20240101", acquisition_timestamp="2024-01-01T10:00:00Z",
        plume_centroid_lat=51.0, plume_centroid_lon=6.5,
        flow_rate_kgh=200.0,
    )
    write_quantification_record(r1, path=tmp_quant_path)
    write_quantification_record(r2, path=tmp_quant_path)
    with open(tmp_quant_path) as f:
        data = json.load(f)
    assert len(data) == 1
    assert data[0]["flow_rate_kgh"] == pytest.approx(200.0)


def test_multiple_sites_append(tmp_quant_path):
    """Different sites accumulate as separate records."""
    for site in ["site_a", "site_b", "site_c"]:
        write_quantification_record(QuantificationRecord(
            site=site, scene_id="S2A_X", acquisition_timestamp="2024-01-01T10:00:00Z",
            plume_centroid_lat=51.0, plume_centroid_lon=6.5,
        ), path=tmp_quant_path)
    with open(tmp_quant_path) as f:
        data = json.load(f)
    assert len(data) == 3


def test_excluded_record_written_but_filtered(tmp_quant_path):
    """Excluded records are written to disk but filtered by load_quantification_records."""
    r_excl = QuantificationRecord(
        site="lipp", scene_id="S2A_X", acquisition_timestamp="2024-01-01T10:00:00Z",
        plume_centroid_lat=51.2, plume_centroid_lon=12.4,
        excluded=True, exclusion_reason="terrain_artifact",
    )
    r_good = QuantificationRecord(
        site="maas", scene_id="S2A_Y", acquisition_timestamp="2024-01-02T10:00:00Z",
        plume_centroid_lat=51.9, plume_centroid_lon=4.0,
        flow_rate_kgh=420.0,
    )
    write_quantification_record(r_excl, path=tmp_quant_path)
    write_quantification_record(r_good, path=tmp_quant_path)

    # Raw file: both records present
    with open(tmp_quant_path) as f:
        data = json.load(f)
    assert len(data) == 2

    # load_quantification_records with exclude_flagged=True omits excluded record
    records = load_quantification_records(tmp_quant_path, exclude_flagged=True)
    assert "lipp" not in records
    assert "maas" in records

    # With exclude_flagged=False both are returned
    records_all = load_quantification_records(tmp_quant_path, exclude_flagged=False)
    assert "lipp" in records_all


# ── run_quantification determinism test ───────────────────────────────────────

def test_run_quantification_deterministic(synthetic_scene, tmp_quant_path):
    """
    Given a fixed synthetic scene and pre-supplied wind, run_quantification()
    must return the same flow_rate_kgh on every call.
    """
    b11, b12, mask = synthetic_scene

    wind = {
        "wind_speed_ms": 3.5,
        "wind_dir_deg": 225.0,
        "wind_source": "test_fixture",
        "era5_u_ms": -2.47,
        "era5_v_ms": -2.47,
    }

    cfg = SiteCfg(
        site="test_site",
        scene_id="S2A_TEST_20240625",
        acquisition_timestamp="2024-06-25T10:36:31Z",
        lat=51.0, lon=6.5,
        b11=b11, b12=b12,
        mask_original=mask,
        wind_override=wind,
        quant_path=tmp_quant_path,
    )

    r1 = run_quantification(cfg, dry_run=True)
    r2 = run_quantification(cfg, dry_run=True)

    assert r1.flow_rate_kgh == pytest.approx(r2.flow_rate_kgh, rel=1e-6)
    assert r1.uncertainty_pct == 30
    assert r1.wind_source == "test_fixture"
    assert r1.schema_version == SCHEMA_VERSION


def test_run_quantification_uses_original_mask_for_belchatow(synthetic_scene, tmp_quant_path):
    """
    For Bełchatów, runner must select original mask even when BT mask is also supplied.
    """
    b11, b12, mask_orig = synthetic_scene
    mask_bt = np.zeros_like(mask_orig)   # simulate dead BT mask

    cfg = SiteCfg(
        site="belchatow",
        scene_id="S2B_T34UCB_20240824",
        acquisition_timestamp="2024-08-24T09:40:00Z",
        lat=51.264, lon=19.331,
        b11=b11, b12=b12,
        mask_original=mask_orig,
        mask_bitemporal=mask_bt,
        wind_override={"wind_speed_ms": 2.19, "wind_dir_deg": 245.0,
                       "wind_source": "ERA5_reanalysis", "era5_u_ms": -1.55, "era5_v_ms": -1.55},
        quant_path=tmp_quant_path,
    )

    record = run_quantification(cfg, dry_run=True)
    # Should have non-zero pixels from original mask; BT (all zeros) would give 0
    assert record.n_plume_pixels > 0, "Runner used BT mask (zero pixels) instead of original"
    assert record.mask_source == "ch4net_v8_original"


def test_run_quantification_uses_bt_mask_for_lippendorf(synthetic_scene, tmp_quant_path):
    """
    For Lippendorf, runner must select the bitemporal mask even when original is supplied.
    """
    b11, b12, mask_orig = synthetic_scene
    # BT mask: small patch (simulates terrain suppressed by BT differencing → near-zero)
    mask_bt = np.zeros_like(mask_orig)
    mask_bt[5:7, 5:7] = 1.0  # tiny residual

    cfg = SiteCfg(
        site="lippendorf",
        scene_id="S2B_T33UUS_20240922",
        acquisition_timestamp="2024-09-22T10:16:29Z",
        lat=51.178, lon=12.378,
        b11=b11, b12=b12,
        mask_original=mask_orig,
        mask_bitemporal=mask_bt,
        wind_override={"wind_speed_ms": 3.5, "wind_dir_deg": 200.0,
                       "wind_source": "test", "era5_u_ms": None, "era5_v_ms": None},
        quant_path=tmp_quant_path,
    )

    record = run_quantification(cfg, dry_run=True)
    assert record.mask_source == "ch4net_v8_bitemporal"
    # BT mask has only 4 pixels → much fewer than original (100 pixels)
    assert record.n_plume_pixels < 20


# ── EXCLUDED_SITES guard tests ────────────────────────────────────────────────

def test_excluded_sites_registered():
    """EXCLUDED_SITES must contain Lippendorf and Groningen."""
    assert "lippendorf" in EXCLUDED_SITES
    assert "groningen" in EXCLUDED_SITES


def test_risk_model_returns_excluded_tier_for_lippendorf():
    """RiskModel.site_risk('lippendorf') must return risk_tier='EXCLUDED' with no EUR liability."""
    from src.api.risk_model import RiskModel
    m = RiskModel()
    result = m.site_risk("lippendorf")
    assert result["risk_tier"] == "EXCLUDED"
    assert result["carbon_liability_eur"] is None
    assert result["exclusion_reason"] == "terrain_artifact"


def test_risk_model_returns_excluded_tier_for_groningen():
    """RiskModel.site_risk('groningen') must return risk_tier='EXCLUDED'."""
    from src.api.risk_model import RiskModel
    m = RiskModel()
    result = m.site_risk("groningen")
    assert result["risk_tier"] == "EXCLUDED"
    assert result["carbon_liability_eur"] is None


def test_stress_test_excluded_returns_zero_flow():
    """StressTestEngine._get_site_detection must return 0 flow for excluded sites."""
    from src.stress_testing.stress_test import StressTestEngine
    engine = StressTestEngine()
    det = engine._get_site_detection("lippendorf")
    assert det["flow_rate_kgh"] == 0.0
    assert det["p_detect"] == 0.0
    assert det["flow_source"] == "excluded"

    det_g = engine._get_site_detection("groningen")
    assert det_g["flow_rate_kgh"] == 0.0
