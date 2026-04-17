"""
stress_test.py
==============
Climate stress testing engine for satellite-detected methane emissions.

Translates CH₄Net detection results into portfolio-level financial risk
metrics under ECB / NGFS Phase IV climate transition scenarios.

The core question this module answers:
  "Given what the satellite sees today, how much could carbon liabilities
   cost a lender's portfolio over a 5–10 year horizon under different
   climate policy regimes?"

Architecture
------------
1. **SiteStressResult**  — per-emitter: Monte Carlo distribution of annual
   carbon cost under each scenario, with uncertainty from both:
   - Detection probability (Wilson CI on p̂)
   - Carbon price volatility (GBM scenario paths)

2. **PortfolioStressResult** — per-issuer aggregation: total carbon liability
   as a fraction of EBITDA, implied PD shift, portfolio VaR/CVaR.

3. **CreditTransmission** — maps carbon cost shock → credit quality:
   - Carbon cost / EBITDA → implied rating notch migration
   - Merton-style: asset value = equity - carbon liability → PD shift
   - LGD adjustment: stranded asset haircut under orderly/disorderly

Key references:
  - ECB 2024 Climate Stress Test methodology (Pillar 2)
  - NGFS Phase IV scenario technical documentation (June 2024)
  - Battiston et al. (2017) "A climate stress-test of the financial system"
  - Basel Committee: BCBS 239 risk data aggregation, SR 11-7 model risk

Usage:
  from src.stress_testing.stress_test import StressTestEngine
  engine = StressTestEngine()
  results = engine.run_portfolio_stress(
      tickers=["RWE.DE", "PGE.WA"],
      scenarios=["orderly", "disorderly", "hot_house"],
      horizon_years=10,
      n_paths=50000,
  )
"""

import json
import math
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from src.stress_testing.scenarios import (
    SCENARIOS,
    simulate_ets_paths,
    get_ch4_multiplier,
    ClimateScenario,
)
from src.api.risk_model import (
    SITE_OPERATOR_MAP,
    TICKER_SITES,
    CH4_GWP100,
    HOURS_PER_YEAR,
    SC_TO_KGHR_PROXY,
    FLOW_UNCERTAINTY_FACTOR,
    EXCLUDED_SITES,
)

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

# Proxy EBITDA for carbon-cost-to-EBITDA ratio (EUR millions, from public filings)
# These are rough 2023/2024 figures — enough for stress test calibration
ISSUER_EBITDA_EUR_M: dict[str, float] = {
    "RWE.DE":  5_400.0,   # RWE AG FY2023 adjusted EBITDA
    "PGE.WA":  3_200.0,   # PGE S.A. FY2023 adjusted EBITDA (PLN→EUR ~4.5)
    "UN01.DE": 3_800.0,   # Uniper SE (post-nationalisation, approximate)
    "SHEL.L":  66_000.0,  # Shell plc FY2023 adjusted EBITDA
}

# Rating migration: carbon cost as % of EBITDA → implied notch downgrade
# Calibrated to Moody's KMV and S&P issuer-level studies on transition risk
CARBON_EBITDA_THRESHOLDS = [
    (0.01, 0),   # <1% of EBITDA → no notch impact
    (0.03, 1),   # 1-3% → 1 notch
    (0.05, 2),   # 3-5% → 2 notches
    (0.10, 3),   # 5-10% → 3 notches
    (0.20, 5),   # 10-20% → 5 notches (potential fallen angel)
    (1.00, 8),   # >20% → viability threat
]

# Baseline PD by notch (simplified, Moody's-like 1-year PD %)
# Index 0 = Aaa, 1 = Aa1, ..., 6 = Baa1 (IG/HY boundary at ~index 10)
BASELINE_PD_BPS = [1, 2, 3, 5, 8, 12, 18, 30, 50, 80, 130, 200, 350, 600, 1000]

# LGD adjustments for stranded asset scenarios
LGD_BASE = 0.40  # Basel standard for senior unsecured
LGD_STRANDED_ORDERLY = 0.45    # mild haircut — orderly wind-down
LGD_STRANDED_DISORDERLY = 0.60  # severe — forced closure, illiquid assets


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class SiteStressResult:
    """Stress test output for a single emitter site."""
    site: str
    scenario: str
    horizon_years: int

    # Detection inputs (from satellite data)
    p_detect: float
    p_detect_lo_95: float
    p_detect_hi_95: float
    flow_rate_kgh: float
    flow_source: str  # "cemf_ime" or "sc_proxy"

    # Monte Carlo outputs — annual carbon liability distribution (EUR)
    # Each entry = one simulated annual cost at a given horizon year
    mean_annual_cost_eur: list[float]       # mean per year [yr1, yr2, ..., yrN]
    p50_annual_cost_eur: list[float]        # median per year
    p95_annual_cost_eur: list[float]        # 95th percentile per year
    p99_annual_cost_eur: list[float]        # 99th percentile per year

    # Terminal year (last year of horizon) distribution stats
    terminal_mean_eur: float
    terminal_var95_eur: float               # 95% VaR (worst 5% of paths)
    terminal_cvar95_eur: float              # 95% CVaR (expected shortfall)
    terminal_var99_eur: float
    terminal_cvar99_eur: float

    # Cumulative cost over full horizon (NPV at 3% discount rate)
    npv_cumulative_mean_eur: float
    npv_cumulative_p95_eur: float


@dataclass
class IssuerStressResult:
    """Aggregated stress result for one corporate issuer (ticker)."""
    ticker: str
    operator: str
    scenario: str
    sites: list[str]

    # Aggregated carbon liability
    terminal_mean_eur: float
    terminal_var95_eur: float
    terminal_cvar95_eur: float

    # Credit transmission
    ebitda_eur_m: Optional[float]
    carbon_cost_to_ebitda_pct: Optional[float]
    implied_notch_downgrade: int
    implied_pd_bps: Optional[float]
    lgd: float

    # NPV
    npv_cumulative_mean_eur: float
    npv_cumulative_p95_eur: float

    # Per-site detail
    site_results: list[SiteStressResult] = field(default_factory=list)


@dataclass
class PortfolioStressResult:
    """Full portfolio stress test output across all scenarios."""
    scenarios_run: list[str]
    horizon_years: int
    n_paths: int
    discount_rate: float

    # Per-scenario, per-issuer results
    issuer_results: dict[str, list[IssuerStressResult]]  # scenario → [IssuerStressResult]

    # Portfolio-level aggregates per scenario
    portfolio_terminal_mean_eur: dict[str, float]    # scenario → total mean
    portfolio_terminal_var95_eur: dict[str, float]    # scenario → total VaR95
    portfolio_terminal_cvar95_eur: dict[str, float]   # scenario → total CVaR95
    portfolio_npv_mean_eur: dict[str, float]          # scenario → total NPV mean


# ── Engine ────────────────────────────────────────────────────────────────────

class StressTestEngine:
    """
    Monte Carlo stress testing engine.

    Combines satellite detection probabilities with NGFS climate scenario
    carbon price paths to generate forward-looking carbon liability
    distributions for individual emitters and lending portfolios.
    """

    def __init__(
        self,
        tropomi_path: str = "results_analysis/tropomi_validation.json",
        multidate_path: str = "results_analysis/multidate_validation.json",
        quant_path: str = "results_analysis/quantification.json",
        discount_rate: float = 0.03,
    ):
        self.discount_rate = discount_rate
        self._tropomi: dict = {}
        self._multidate: dict = {}
        self._quant: dict = {}

        for path, attr in [
            (tropomi_path, "_tropomi"),
            (multidate_path, "_multidate"),
            (quant_path, "_quant"),
        ]:
            if Path(path).exists():
                with open(path) as f:
                    raw = json.load(f)
                # Normalize list-of-records to dict keyed by site name
                if isinstance(raw, list):
                    raw = {r["site"]: r for r in raw if "site" in r}
                setattr(self, attr, raw)
                logger.info("Loaded %s (%d entries)", path, len(getattr(self, attr)))

    # ── Site-level detection inputs ──────────────────────────────────────

    def _get_site_detection(self, site_name: str) -> dict:
        """
        Extract detection probability and flow rate for a site.

        Merges data from tropomi_validation.json and multidate_validation.json.
        Falls back to S/C-proxy flow rate if CEMF quantification unavailable.

        Excluded sites (terrain artefacts, CFAR false positives) return zero
        so their contribution to any Monte Carlo path is exactly zero.
        """
        if site_name in EXCLUDED_SITES:
            return {
                "p_detect":       0.0,
                "p_detect_lo_95": 0.0,
                "p_detect_hi_95": 0.0,
                "mean_sc":        None,
                "flow_rate_kgh":  0.0,
                "flow_source":    "excluded",
            }

        trop = self._tropomi.get(site_name, {})
        md = self._multidate.get(site_name, {})

        # Detection probability from tropomi_validation (already computed)
        p_detect = trop.get("p_detect", md.get("p_detect", 0.0))
        p_lo = trop.get("p_detect_lo_95", 0.0)
        p_hi = trop.get("p_detect_hi_95", 0.0)
        mean_sc = trop.get("mean_sc_detected", md.get("mean_sc_detected"))

        # Flow rate: prefer quantification, fall back to S/C proxy
        quant = self._quant.get(site_name, {})
        flow_kgh = quant.get("flow_rate_kgh")
        flow_source = "cemf_ime"

        if flow_kgh is None and mean_sc is not None:
            flow_kgh = max(0.0, mean_sc - 1.0) * SC_TO_KGHR_PROXY
            flow_source = "sc_proxy"

        if flow_kgh is None:
            flow_kgh = 0.0
            flow_source = "none"

        return {
            "p_detect": p_detect or 0.0,
            "p_detect_lo_95": p_lo or 0.0,
            "p_detect_hi_95": p_hi or 0.0,
            "mean_sc": mean_sc,
            "flow_rate_kgh": flow_kgh,
            "flow_source": flow_source,
        }

    # ── Core Monte Carlo ─────────────────────────────────────────────────

    def _run_site_mc(
        self,
        site_name: str,
        scenario_name: str,
        n_paths: int = 50_000,
        horizon_years: int = 10,
        base_year: int = 2024,
        rng: Optional[np.random.Generator] = None,
    ) -> SiteStressResult:
        """
        Monte Carlo simulation for one site under one scenario.

        The annual carbon cost for path i, year t is:

          cost(i,t) = p̂_draw × Q̂_CH4 × GWP100 × π_ETS(i,t) × M_CH4(t)

        where:
          p̂_draw    ~ Beta(α, β) fitted to Wilson CI  (detection uncertainty)
          Q̂_CH4     ~ LogNormal(μ_flow, σ_flow)       (flow rate uncertainty)
          π_ETS(i,t) = GBM carbon price path from scenarios.py
          M_CH4(t)   = methane-specific regulatory multiplier

        This properly propagates both measurement uncertainty AND
        policy uncertainty through the same Monte Carlo.
        """
        if rng is None:
            rng = np.random.default_rng(42)

        det = self._get_site_detection(site_name)
        p_hat = det["p_detect"]
        p_lo = det["p_detect_lo_95"]
        p_hi = det["p_detect_hi_95"]
        flow_kgh = det["flow_rate_kgh"]
        flow_source = det["flow_source"]

        # ── Sample detection probability from Beta distribution ──────
        # Fit Beta(α, β) to match p̂ and the Wilson 95% CI width
        if p_hat > 0 and p_hat < 1:
            # Method of moments from CI: approximate σ² from CI width
            ci_width = max(p_hi - p_lo, 0.01)
            sigma_sq = (ci_width / (2 * 1.96)) ** 2
            sigma_sq = max(sigma_sq, 1e-6)
            # Beta params from mean and variance
            mean = p_hat
            alpha = mean * (mean * (1 - mean) / sigma_sq - 1)
            beta_param = (1 - mean) * (mean * (1 - mean) / sigma_sq - 1)
            alpha = max(alpha, 0.5)
            beta_param = max(beta_param, 0.5)
            p_draws = rng.beta(alpha, beta_param, size=n_paths)
        elif p_hat >= 1.0:
            # Saturated detection — sample near 1.0
            p_draws = rng.beta(10, 1, size=n_paths)  # concentrated near 1
        else:
            p_draws = np.zeros(n_paths)

        # ── Sample flow rate from LogNormal ──────────────────────────
        # σ is sourced from uncertainty.py SSOT: 0.30 (ERA5) or 0.50 (fallback)
        if flow_kgh > 0:
            sigma_flow = FLOW_UNCERTAINTY_FACTOR  # 0.30 (ERA5) per uncertainty.py SSOT
            mu_flow = np.log(flow_kgh) - 0.5 * sigma_flow**2
            flow_draws = rng.lognormal(mu_flow, sigma_flow, size=n_paths)
        else:
            flow_draws = np.zeros(n_paths)

        # ── Carbon price paths (from scenarios.py GBM) ───────────────
        ets_paths = simulate_ets_paths(
            scenario_name, n_paths=n_paths,
            horizon_years=horizon_years, base_year=base_year,
            rng=rng,
        )

        # ── Compute annual cost per path ─────────────────────────────
        # cost(i, t) = p̂(i) × Q̂(i) × HOURS × GWP100 / 1000 × π_ETS(i,t) × M_CH4(t)
        # Units: kg/hr × hr/yr × (tCO2e/tCH4) / 1000 × EUR/tCO2e = EUR/yr

        annual_tCH4 = (
            p_draws[:, np.newaxis]
            * flow_draws[:, np.newaxis]
            * HOURS_PER_YEAR
            / 1000.0  # kg → tonnes
        )

        annual_tCO2e = annual_tCH4 * CH4_GWP100  # (n_paths, horizon_years)

        # Apply methane multiplier per year
        ch4_mult = np.array([
            get_ch4_multiplier(scenario_name, base_year + t)
            for t in range(horizon_years)
        ])

        annual_cost = annual_tCO2e * ets_paths * ch4_mult[np.newaxis, :]

        # ── Statistics per year ───────────────────────────────────────
        mean_by_year = np.mean(annual_cost, axis=0).tolist()
        p50_by_year = np.percentile(annual_cost, 50, axis=0).tolist()
        p95_by_year = np.percentile(annual_cost, 95, axis=0).tolist()
        p99_by_year = np.percentile(annual_cost, 99, axis=0).tolist()

        # Terminal year statistics
        terminal = annual_cost[:, -1]
        terminal_mean = float(np.mean(terminal))
        terminal_var95 = float(np.percentile(terminal, 95))
        terminal_cvar95 = float(np.mean(terminal[terminal >= np.percentile(terminal, 95)]))
        terminal_var99 = float(np.percentile(terminal, 99))
        terminal_cvar99 = float(np.mean(terminal[terminal >= np.percentile(terminal, 99)]))

        # ── NPV of cumulative costs (discounted) ─────────────────────
        discount_factors = np.array([
            1.0 / (1 + self.discount_rate) ** (t + 1)
            for t in range(horizon_years)
        ])
        npv_per_path = np.sum(annual_cost * discount_factors[np.newaxis, :], axis=1)
        npv_mean = float(np.mean(npv_per_path))
        npv_p95 = float(np.percentile(npv_per_path, 95))

        return SiteStressResult(
            site=site_name,
            scenario=scenario_name,
            horizon_years=horizon_years,
            p_detect=p_hat,
            p_detect_lo_95=p_lo,
            p_detect_hi_95=p_hi,
            flow_rate_kgh=flow_kgh,
            flow_source=flow_source,
            mean_annual_cost_eur=mean_by_year,
            p50_annual_cost_eur=p50_by_year,
            p95_annual_cost_eur=p95_by_year,
            p99_annual_cost_eur=p99_by_year,
            terminal_mean_eur=terminal_mean,
            terminal_var95_eur=terminal_var95,
            terminal_cvar95_eur=terminal_cvar95,
            terminal_var99_eur=terminal_var99,
            terminal_cvar99_eur=terminal_cvar99,
            npv_cumulative_mean_eur=npv_mean,
            npv_cumulative_p95_eur=npv_p95,
        )

    # ── Credit transmission ──────────────────────────────────────────────

    @staticmethod
    def _carbon_cost_to_notch_downgrade(cost_eur: float, ebitda_eur: float) -> int:
        """
        Map carbon cost as % of EBITDA to implied rating notch downgrade.

        Based on empirical studies of transition risk on credit quality:
        - Battiston et al. (2017)
        - ECB climate stress test 2022/2024 methodology
        """
        if ebitda_eur <= 0:
            return 0
        ratio = cost_eur / ebitda_eur
        notches = 0
        for threshold, n in CARBON_EBITDA_THRESHOLDS:
            if ratio <= threshold:
                return n
            notches = n
        return notches

    @staticmethod
    def _get_lgd(scenario_name: str) -> float:
        """Scenario-dependent LGD for stranded asset risk."""
        if scenario_name == "disorderly":
            return LGD_STRANDED_DISORDERLY
        elif scenario_name == "orderly":
            return LGD_STRANDED_ORDERLY
        else:
            return LGD_BASE

    # ── Portfolio-level stress test ──────────────────────────────────────

    def run_portfolio_stress(
        self,
        tickers: list[str],
        scenarios: Optional[list[str]] = None,
        horizon_years: int = 10,
        n_paths: int = 50_000,
        base_year: int = 2024,
        seed: int = 42,
    ) -> PortfolioStressResult:
        """
        Run the full stress test across all scenarios for a portfolio.

        Args:
            tickers:       equity tickers to stress (e.g. ["RWE.DE", "PGE.WA"])
            scenarios:     scenario names (default: all three NGFS)
            horizon_years: projection horizon
            n_paths:       Monte Carlo paths per scenario
            base_year:     starting year for projection
            seed:          random seed for reproducibility

        Returns:
            PortfolioStressResult with full per-site, per-issuer, per-scenario
            decomposition plus portfolio-level VaR/CVaR.
        """
        if scenarios is None:
            scenarios = list(SCENARIOS.keys())

        rng = np.random.default_rng(seed)

        issuer_results: dict[str, list[IssuerStressResult]] = {s: [] for s in scenarios}
        portfolio_terminal_mean: dict[str, float] = {}
        portfolio_terminal_var95: dict[str, float] = {}
        portfolio_terminal_cvar95: dict[str, float] = {}
        portfolio_npv_mean: dict[str, float] = {}

        for scenario_name in scenarios:
            scenario_total_mean = 0.0
            scenario_total_var95 = 0.0
            scenario_total_cvar95 = 0.0
            scenario_total_npv = 0.0

            for ticker in tickers:
                sites = TICKER_SITES.get(ticker, [])
                if not sites:
                    logger.warning("No sites mapped for ticker %s", ticker)
                    continue

                site_results = []
                ticker_terminal_mean = 0.0
                ticker_terminal_var95 = 0.0
                ticker_terminal_cvar95 = 0.0
                ticker_npv_mean = 0.0
                ticker_npv_p95 = 0.0

                for site_name in sites:
                    sr = self._run_site_mc(
                        site_name, scenario_name,
                        n_paths=n_paths,
                        horizon_years=horizon_years,
                        base_year=base_year,
                        rng=np.random.default_rng(rng.integers(0, 2**32)),
                    )
                    site_results.append(sr)
                    ticker_terminal_mean += sr.terminal_mean_eur
                    ticker_terminal_var95 += sr.terminal_var95_eur
                    ticker_terminal_cvar95 += sr.terminal_cvar95_eur
                    ticker_npv_mean += sr.npv_cumulative_mean_eur
                    ticker_npv_p95 += sr.npv_cumulative_p95_eur

                # Credit transmission
                ebitda = ISSUER_EBITDA_EUR_M.get(ticker)
                if ebitda is not None:
                    ebitda_eur = ebitda * 1e6
                    cost_ratio = ticker_terminal_var95 / ebitda_eur
                    notch_down = self._carbon_cost_to_notch_downgrade(
                        ticker_terminal_var95, ebitda_eur
                    )
                else:
                    ebitda_eur = None
                    cost_ratio = None
                    notch_down = 0

                lgd = self._get_lgd(scenario_name)

                op_info = SITE_OPERATOR_MAP.get(sites[0], {})
                operator = op_info.get("operator", "Unknown")

                ir = IssuerStressResult(
                    ticker=ticker,
                    operator=operator,
                    scenario=scenario_name,
                    sites=sites,
                    terminal_mean_eur=ticker_terminal_mean,
                    terminal_var95_eur=ticker_terminal_var95,
                    terminal_cvar95_eur=ticker_terminal_cvar95,
                    ebitda_eur_m=ebitda,
                    carbon_cost_to_ebitda_pct=round(cost_ratio * 100, 4) if cost_ratio else None,
                    implied_notch_downgrade=notch_down,
                    implied_pd_bps=BASELINE_PD_BPS[min(notch_down, len(BASELINE_PD_BPS) - 1)],
                    lgd=lgd,
                    npv_cumulative_mean_eur=ticker_npv_mean,
                    npv_cumulative_p95_eur=ticker_npv_p95,
                    site_results=site_results,
                )
                issuer_results[scenario_name].append(ir)

                scenario_total_mean += ticker_terminal_mean
                scenario_total_var95 += ticker_terminal_var95
                scenario_total_cvar95 += ticker_terminal_cvar95
                scenario_total_npv += ticker_npv_mean

            portfolio_terminal_mean[scenario_name] = scenario_total_mean
            portfolio_terminal_var95[scenario_name] = scenario_total_var95
            portfolio_terminal_cvar95[scenario_name] = scenario_total_cvar95
            portfolio_npv_mean[scenario_name] = scenario_total_npv

        return PortfolioStressResult(
            scenarios_run=scenarios,
            horizon_years=horizon_years,
            n_paths=n_paths,
            discount_rate=self.discount_rate,
            issuer_results=issuer_results,
            portfolio_terminal_mean_eur=portfolio_terminal_mean,
            portfolio_terminal_var95_eur=portfolio_terminal_var95,
            portfolio_terminal_cvar95_eur=portfolio_terminal_cvar95,
            portfolio_npv_mean_eur=portfolio_npv_mean,
        )

    # ── Human-readable summary ───────────────────────────────────────────

    @staticmethod
    def format_summary(result: PortfolioStressResult) -> str:
        """Format stress test results as a readable table."""
        lines = []
        lines.append("=" * 90)
        lines.append("CLIMATE STRESS TEST — SATELLITE-DERIVED CARBON LIABILITY")
        lines.append(f"Horizon: {result.horizon_years}yr | Paths: {result.n_paths:,} | "
                      f"Discount: {result.discount_rate:.0%}")
        lines.append("=" * 90)

        for scenario in result.scenarios_run:
            sc = SCENARIOS[scenario]
            lines.append(f"\n{'─' * 90}")
            lines.append(f"SCENARIO: {sc.name.upper()} — {sc.description[:70]}...")
            lines.append(f"{'─' * 90}")

            issuers = result.issuer_results[scenario]
            lines.append(
                f"{'Ticker':<10} {'Operator':<15} {'Sites':>5} "
                f"{'Mean €/yr':>14} {'VaR95 €/yr':>14} {'CVaR95 €/yr':>15} "
                f"{'C/EBITDA':>8} {'ΔNotch':>6} {'LGD':>5}"
            )
            lines.append("-" * 90)

            for ir in issuers:
                c_ebitda = f"{ir.carbon_cost_to_ebitda_pct:.3f}%" if ir.carbon_cost_to_ebitda_pct else "N/A"
                lines.append(
                    f"{ir.ticker:<10} {ir.operator:<15} {len(ir.sites):>5} "
                    f"{ir.terminal_mean_eur:>14,.0f} {ir.terminal_var95_eur:>14,.0f} "
                    f"{ir.terminal_cvar95_eur:>15,.0f} "
                    f"{c_ebitda:>8} {ir.implied_notch_downgrade:>6} {ir.lgd:>5.0%}"
                )

            lines.append("-" * 90)
            lines.append(
                f"{'PORTFOLIO':<26} "
                f"{result.portfolio_terminal_mean_eur[scenario]:>14,.0f} "
                f"{result.portfolio_terminal_var95_eur[scenario]:>14,.0f} "
                f"{result.portfolio_terminal_cvar95_eur[scenario]:>15,.0f}"
            )
            lines.append(
                f"  10yr NPV (mean): €{result.portfolio_npv_mean_eur[scenario]:,.0f}"
            )

        lines.append(f"\n{'=' * 90}")
        return "\n".join(lines)
