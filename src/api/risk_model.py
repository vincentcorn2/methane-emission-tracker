"""
risk_model.py
=============
Computes per-asset methane emission risk scores from CH4Net multi-date
validation results and optional TROPOMI cross-validation data.

Risk score components
---------------------
1. p_detect         — empirical detection probability (fraction of valid S2 dates
                       with S/C > 1.15), with Wilson 95% CI.
2. mean_flow_kgh    — mean CH4 flow rate on detection dates (from CEMF/IME
                       quantification or S/C-based proxy).
3. annual_tCO2e     — annualised GHG liability = p_detect × mean_flow_kgh
                       × 8760 hrs × GWP / 1000.
4. carbon_eur_yr    — annual carbon liability at EU ETS spot price.
5. tropomi_score    — fraction of dates with dual-sensor confirmation (TROPOMI
                       ΔXCHₓ ≥ 5 ppb AND S/C > 1.15).  0 if no TROPOMI data.
6. risk_tier        — HIGH / MEDIUM / LOW / UNDETECTED based on p_detect and
                       tropomi_score.

Uncertainty bounds
------------------
All monetary/mass figures carry a 90% confidence interval computed by
propagating the Wilson CI on p_detect through the flow-rate estimate.
Flow-rate uncertainty uses a ±50% heuristic (see CEMF caveat in run_quant_fixed.py
re: pixel-area factor).

Ticker mapping
--------------
Maps each site to its parent company equity ticker for portfolio aggregation.
Companies:  RWE AG → RWE.DE, LEAG → private (no ticker), PGE → PGE.WA,
            Vattenfall → private, NAM → Shell/Exxon JV (SHEL.L / XOM).

Usage
-----
  from src.api.risk_model import RiskModel
  model = RiskModel()
  score = model.site_risk("belchatow")
  portfolio = model.portfolio_risk(["RWE.DE", "PGE.WA"])
"""

import json
import math
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# EU ETS carbon price (€/tCO2e) — update periodically
ETS_PRICE_EUR_PER_TONNE = 65.0

# CH4 GWP-100 (IPCC AR6)
CH4_GWP100 = 29.8

# Operating hours per year (lignite plants: ~85% capacity factor)
HOURS_PER_YEAR = 8760 * 0.85

# Flow-rate uncertainty: SSOT is src/quantification/uncertainty.py
# ±30% when ERA5 wind used (post-ERA5 quadrature budget from WS2 Tech Report S4)
# ±50% when climatological fallback used
# This constant is kept for backward-compatibility with stress_test.py imports;
# the actual per-record value is stored in uncertainty_pct of QuantificationRecord.
from src.quantification.uncertainty import UNCERTAINTY_PCT_ERA5, UNCERTAINTY_PCT_FALLBACK
FLOW_UNCERTAINTY_FACTOR = UNCERTAINTY_PCT_ERA5 / 100.0   # 0.30

# Detection probability threshold for risk tiers
P_DETECT_HIGH   = 0.60
P_DETECT_MEDIUM = 0.30

# TROPOMI dual-sensor threshold for upgrading confidence
TROPOMI_CONFIRM_THRESH = 0.50  # fraction of dates with dual-sensor confirm

# S/C → flow rate proxy (kg/hr per unit S/C above background)
# Calibrated from Weisweiler: S/C=23.461 → ~420 kg/hr (CEMF run_quant_fixed)
# and Bełchatów: mean S/C=60 → ~1500 kg/hr.  Linear proxy: ~25 kg/hr per S/C unit.
SC_TO_KGHR_PROXY = 25.0

# ── Site exclusion registry ───────────────────────────────────────────────────
# Sites listed here are excluded from all risk scoring and financial outputs.
# They are retained in quantification.json for provenance (excluded=true),
# but RiskModel.site_risk() short-circuits them and StressTestEngine treats
# them as zero-emission.  Add new exclusions here — never silently drop records.
EXCLUDED_SITES: dict[str, str] = {
    "lippendorf": "terrain_artifact",                    # S/C=155.4 is terrain contrast (BT→0.19)
    "groningen":  "cfar_suppressed_false_positive",      # nl_grijpskerk compressor spill; CFAR kills signal
}

# ── Ticker / operator map ─────────────────────────────────────────────────────

SITE_OPERATOR_MAP = {
    # RWE AG (Frankfurt: RWE.DE)
    "weisweiler":    {"operator": "RWE AG",   "ticker": "RWE.DE",  "exchange": "XETRA"},
    "neurath":       {"operator": "RWE AG",   "ticker": "RWE.DE",  "exchange": "XETRA"},
    "niederaussem":  {"operator": "RWE AG",   "ticker": "RWE.DE",  "exchange": "XETRA"},
    # LEAG (private — formerly Vattenfall Germany; no public ticker)
    "lippendorf":    {"operator": "LEAG",     "ticker": None,      "exchange": None},
    "boxberg":       {"operator": "LEAG",     "ticker": None,      "exchange": None},
    "jaenschwalde":  {"operator": "LEAG",     "ticker": None,      "exchange": None},
    "schwarze_pumpe":{"operator": "LEAG",     "ticker": None,      "exchange": None},
    # PGE Polska Grupa Energetyczna (Warsaw: PGE.WA)
    "belchatow":     {"operator": "PGE S.A.", "ticker": "PGE.WA",  "exchange": "WSE"},
    "rybnik":        {"operator": "PGE S.A.", "ticker": "PGE.WA",  "exchange": "WSE"},
    "turow":         {"operator": "PGE S.A.", "ticker": "PGE.WA",  "exchange": "WSE"},
    # Vattenfall (state-owned Swedish; Maasvlakte sold to Uniper in 2024)
    "maasvlakte":    {"operator": "Uniper SE","ticker": "UN01.DE", "exchange": "XETRA"},
    # NAM (Shell 50% / ExxonMobil 50% JV)
    "groningen":     {"operator": "NAM (Shell/XOM JV)",
                                              "ticker": "SHEL.L",  "exchange": "LSE"},
}

# Reverse map: ticker → list of sites
TICKER_SITES: dict[str, list[str]] = {}
for _site, _info in SITE_OPERATOR_MAP.items():
    _t = _info["ticker"]
    if _t:
        TICKER_SITES.setdefault(_t, []).append(_site)


class RiskModel:
    """
    Loads pre-computed validation results and exposes risk scoring methods.

    Reads:
      results_analysis/multidate_validation.json  — S/C per site × date
      results_analysis/tropomi_validation.json    — TROPOMI ΔXCHₓ (optional)
      results_analysis/quantification.json        — CEMF/IME flow rates (optional)
    """

    def __init__(
        self,
        multidate_path: str = "results_analysis/multidate_validation.json",
        tropomi_path:   str = "results_analysis/tropomi_validation.json",
        quant_path:     str = "results_analysis/quantification.json",
    ):
        self._multidate: dict = {}
        self._tropomi:   dict = {}
        self._quant:     dict = {}

        if Path(multidate_path).exists():
            with open(multidate_path) as f:
                self._multidate = json.load(f)
            logger.info("Loaded multidate results for %d sites", len(self._multidate))
        else:
            logger.warning("multidate_validation.json not found — risk scores will be null")

        if Path(tropomi_path).exists():
            with open(tropomi_path) as f:
                self._tropomi = json.load(f)
            logger.info("Loaded TROPOMI results for %d sites", len(self._tropomi))

        if Path(quant_path).exists():
            with open(quant_path) as f:
                raw_quant = json.load(f)
            # Normalize: support both list-of-records and dict keyed by site name
            if isinstance(raw_quant, list):
                self._quant = {r["site"]: r for r in raw_quant if "site" in r}
            else:
                self._quant = raw_quant
            logger.info("Loaded quantification results for %d sites", len(self._quant))

    # ── Core helpers ──────────────────────────────────────────────────────────

    BAD_SCENE_MEAN = 0.4257

    def _is_bad_scene(self, date_record: dict) -> bool:
        sm = date_record.get("site_mean")
        return sm is not None and abs(sm - self.BAD_SCENE_MEAN) < 0.01

    @staticmethod
    def _wilson_ci(n_success: int, n_total: int, z: float = 1.645) -> tuple[float, float]:
        """Wilson score interval (default: 90% CI, z=1.645)."""
        if n_total == 0:
            return (0.0, 0.0)
        p = n_success / n_total
        denom = 1 + z**2 / n_total
        centre = (p + z**2 / (2 * n_total)) / denom
        margin = (z * math.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2))) / denom
        return (max(0.0, round(centre - margin, 4)),
                min(1.0, round(centre + margin, 4)))

    # ── Per-site risk ─────────────────────────────────────────────────────────

    def site_risk(self, site_name: str) -> dict:
        """
        Compute full risk score for one site.

        Returns a dict with all risk components, confidence intervals,
        and financial liability estimates.
        """
        result: dict = {"site": site_name}

        # ── Exclusion short-circuit ───────────────────────────────────────
        # Excluded sites return immediately with no flow rate or EUR liability.
        # They are retained in the JSON for provenance but must never pollute
        # any portfolio or stress-test output with artefact-derived numbers.
        if site_name in EXCLUDED_SITES:
            op_info = SITE_OPERATOR_MAP.get(site_name, {})
            return {
                "site":              site_name,
                "operator":          op_info.get("operator"),
                "ticker":            op_info.get("ticker"),
                "exchange":          op_info.get("exchange"),
                "risk_tier":         "EXCLUDED",
                "exclusion_reason":  EXCLUDED_SITES[site_name],
                "n_valid_dates":     0,
                "n_detections":      0,
                "p_detect":          None,
                "p_detect_lo_90":    None,
                "p_detect_hi_90":    None,
                "mean_sc_detected":  None,
                "flow_rate_kgh":     None,
                "flow_rate_source":  "excluded",
                "annual_tCO2e":      None,
                "annual_tCO2e_lo_90": None,
                "annual_tCO2e_hi_90": None,
                "carbon_liability_eur": None,
                "carbon_eur_lo_90":  None,
                "carbon_eur_hi_90":  None,
                "ets_price_eur_tonne": ETS_PRICE_EUR_PER_TONNE,
                "gwp100_ch4":        CH4_GWP100,
                "tropomi_score":     None,
                "n_dual_sensor_confirms": 0,
            }

        # ── Operator / ticker ─────────────────────────────────────────────
        op_info = SITE_OPERATOR_MAP.get(site_name, {})
        result.update({
            "operator": op_info.get("operator"),
            "ticker":   op_info.get("ticker"),
            "exchange": op_info.get("exchange"),
        })

        # ── Multi-date S/C stats ──────────────────────────────────────────
        md = self._multidate.get(site_name, {})
        dates = md.get("dates", {})

        valid_dates = {
            d: r for d, r in dates.items()
            if not self._is_bad_scene(r) and r.get("sc_ratio") is not None
        }
        n_valid   = len(valid_dates)
        n_detect  = sum(1 for r in valid_dates.values() if r.get("classic_detect"))
        sc_detect = [r["sc_ratio"] for r in valid_dates.values() if r.get("classic_detect")]

        p_detect   = round(n_detect / n_valid, 4) if n_valid > 0 else None
        ci_lo, ci_hi = self._wilson_ci(n_detect, n_valid) if n_valid > 0 else (None, None)
        mean_sc    = round(float(sum(sc_detect) / len(sc_detect)), 3) if sc_detect else None

        result.update({
            "n_valid_dates":      n_valid,
            "n_detections":       n_detect,
            "p_detect":           p_detect,
            "p_detect_lo_90":     ci_lo,
            "p_detect_hi_90":     ci_hi,
            "mean_sc_detected":   mean_sc,
        })

        # ── Flow rate (kg/hr) ─────────────────────────────────────────────
        # Prefer CEMF/IME quantification; fall back to S/C proxy.
        quant = self._quant.get(site_name, {})
        flow_kgh = quant.get("flow_rate_kgh")
        flow_source = "cemf_ime"

        if flow_kgh is None and mean_sc is not None:
            # S/C proxy: ~25 kg/hr per S/C unit above background
            flow_kgh = round(max(0.0, mean_sc - 1.0) * SC_TO_KGHR_PROXY, 1)
            flow_source = "sc_proxy"

        result["flow_rate_kgh"]    = flow_kgh
        result["flow_rate_source"] = flow_source

        # ── Annual GHG liability ──────────────────────────────────────────
        if p_detect is not None and flow_kgh is not None:
            # Expected annual CH4 emission (kg/yr)
            exp_kgyr  = p_detect * flow_kgh * HOURS_PER_YEAR
            exp_tCO2e = round(exp_kgyr * CH4_GWP100 / 1000, 1)
            exp_eur   = round(exp_tCO2e * ETS_PRICE_EUR_PER_TONNE, 0)

            # Uncertainty: propagate CI on p_detect + flow ±50%
            p_lo_eff  = ci_lo if ci_lo is not None else p_detect * 0.5
            p_hi_eff  = ci_hi if ci_hi is not None else p_detect * 1.5
            f_lo      = flow_kgh * (1 - FLOW_UNCERTAINTY_FACTOR)
            f_hi      = flow_kgh * (1 + FLOW_UNCERTAINTY_FACTOR)

            tco2e_lo  = round(p_lo_eff * f_lo * HOURS_PER_YEAR * CH4_GWP100 / 1000, 1)
            tco2e_hi  = round(p_hi_eff * f_hi * HOURS_PER_YEAR * CH4_GWP100 / 1000, 1)
            eur_lo    = round(tco2e_lo * ETS_PRICE_EUR_PER_TONNE, 0)
            eur_hi    = round(tco2e_hi * ETS_PRICE_EUR_PER_TONNE, 0)
        else:
            exp_tCO2e = tco2e_lo = tco2e_hi = None
            exp_eur   = eur_lo = eur_hi = None

        result.update({
            "annual_tCO2e":          exp_tCO2e,
            "annual_tCO2e_lo_90":    tco2e_lo,
            "annual_tCO2e_hi_90":    tco2e_hi,
            "carbon_liability_eur":  exp_eur,
            "carbon_eur_lo_90":      eur_lo,
            "carbon_eur_hi_90":      eur_hi,
            "ets_price_eur_tonne":   ETS_PRICE_EUR_PER_TONNE,
            "gwp100_ch4":            CH4_GWP100,
        })

        # ── TROPOMI dual-sensor score ─────────────────────────────────────
        trop = self._tropomi.get(site_name, {})
        trop_dates = trop.get("dates", {})
        n_dual = sum(
            1 for d in trop_dates.values()
            if d.get("s2_detect") and d.get("trop_detect")
        )
        n_trop_valid = sum(
            1 for d in trop_dates.values()
            if not d.get("is_bad_scene") and "error" not in d.get("tropomi", {})
        )
        tropomi_score = round(n_dual / n_trop_valid, 3) if n_trop_valid > 0 else None
        result.update({
            "tropomi_score":        tropomi_score,
            "n_dual_sensor_confirms": n_dual,
        })

        # ── Risk tier ─────────────────────────────────────────────────────
        if p_detect is None:
            tier = "NO_DATA"
        elif p_detect >= P_DETECT_HIGH:
            tier = "HIGH"
        elif p_detect >= P_DETECT_MEDIUM:
            tier = "MEDIUM" if (tropomi_score is None or tropomi_score >= 0) else "MEDIUM_UNCONFIRMED"
        elif p_detect > 0:
            tier = "LOW"
        else:
            tier = "UNDETECTED"

        # Upgrade to HIGH if strong TROPOMI confirmation even with moderate p_detect
        if tier == "MEDIUM" and tropomi_score is not None and tropomi_score >= TROPOMI_CONFIRM_THRESH:
            tier = "HIGH_DUAL_SENSOR"

        result["risk_tier"] = tier

        return result

    # ── Portfolio aggregation ─────────────────────────────────────────────────

    def portfolio_risk(
        self,
        tickers: list[str],
        lookback_days: int = 90,
    ) -> dict:
        """
        Aggregate emission risk for a portfolio of equity tickers.

        Args:
            tickers:       list of equity ticker symbols (e.g. ["RWE.DE", "PGE.WA"])
            lookback_days: unused for now (future: filter to recent detections only)

        Returns:
            {
              total_annual_tCO2e:      float,
              total_carbon_eur:        float,
              data_coverage_pct:       float,
              per_ticker:              {ticker: {sites: [...], aggregated_risk: {...}}},
              unmatched_tickers:       [str],
            }
        """
        per_ticker: dict = {}
        unmatched: list[str] = []
        total_tCO2e = 0.0
        total_eur   = 0.0
        covered     = 0

        for ticker in tickers:
            sites = TICKER_SITES.get(ticker)
            if not sites:
                unmatched.append(ticker)
                continue

            covered += 1
            ticker_tCO2e = 0.0
            ticker_eur   = 0.0
            site_scores  = []

            for site_name in sites:
                score = self.site_risk(site_name)
                site_scores.append(score)
                if score.get("annual_tCO2e") is not None:
                    ticker_tCO2e += score["annual_tCO2e"]
                if score.get("carbon_liability_eur") is not None:
                    ticker_eur   += score["carbon_liability_eur"]

            # Aggregate uncertainty across sites (sum of lower/upper bounds)
            tco2e_lo = sum(s.get("annual_tCO2e_lo_90") or 0 for s in site_scores)
            tco2e_hi = sum(s.get("annual_tCO2e_hi_90") or 0 for s in site_scores)
            eur_lo   = sum(s.get("carbon_eur_lo_90") or 0 for s in site_scores)
            eur_hi   = sum(s.get("carbon_eur_hi_90") or 0 for s in site_scores)

            # Highest risk tier across sites
            tier_order = ["HIGH_DUAL_SENSOR", "HIGH", "MEDIUM", "LOW", "UNDETECTED", "NO_DATA"]
            tiers = [s.get("risk_tier", "NO_DATA") for s in site_scores]
            max_tier = min(tiers, key=lambda t: tier_order.index(t) if t in tier_order else 99)

            per_ticker[ticker] = {
                "sites":               [s["site"] for s in site_scores],
                "annual_tCO2e":        round(ticker_tCO2e, 1),
                "annual_tCO2e_lo_90":  round(tco2e_lo, 1),
                "annual_tCO2e_hi_90":  round(tco2e_hi, 1),
                "carbon_liability_eur": round(ticker_eur, 0),
                "carbon_eur_lo_90":    round(eur_lo, 0),
                "carbon_eur_hi_90":    round(eur_hi, 0),
                "risk_tier":           max_tier,
                "ets_price_eur_tonne": ETS_PRICE_EUR_PER_TONNE,
                "site_scores":         site_scores,
            }

            total_tCO2e += ticker_tCO2e
            total_eur   += ticker_eur

        coverage_pct = round(100 * covered / len(tickers), 1) if tickers else 0.0

        return {
            "total_annual_tCO2e":    round(total_tCO2e, 1),
            "total_carbon_eur":      round(total_eur, 0),
            "data_coverage_pct":     coverage_pct,
            "per_ticker":            per_ticker,
            "unmatched_tickers":     unmatched,
            "ets_price_eur_tonne":   ETS_PRICE_EUR_PER_TONNE,
        }
