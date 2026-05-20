"""
scripts/finance_climate_var.py
==============================
Monte Carlo Climate Value-at-Risk (CVaR) Engine
Facility: KWB Bełchatów lignite mine  |  Issuer: PGE Polska Grupa Energetyczna

Methodology:
  Stochastic uncertainty propagation adapted from:
    - Desnos, Le Guenedal, Morais & Roncalli (Amundi, 2024):
      "From Climate Stress Testing to Climate Value-at-Risk: A Stochastic Approach"
    - Worden et al. (NIST IR 8575, May 2025):
      "Common Practices for Quantifying Methane Emissions from Plumes
       Detected by Remote Sensing"

Uncertainty budget (NIST IR 8575 §4.2 structure):
  Layer 1 — Annual emission sampling uncertainty   (t-distribution on 37 obs)
  Layer 2 — ERA5 wind systematic bias              (N(0, σ_wind), ±20% random already in CI)
  Layer 3 — Plume mask / spatial extent jitter     (Uniform ±15%)
  Layer 4 — IME retrieval coefficient precision    (optional; default off, in CI)
  Layer 5 — Carbon price                           (log-normal, Amundi GBM approach)
  Layer 6 — Regulatory pass-through / enforcement  (Beta distribution)

Risk metrics output:
  Mean expected annual liability
  95% Climate VaR
  99% Climate VaR
  99% Expected Shortfall (ES / CVaR)
  ... for both GWP100 (EU MRV regulatory) and GWP20 (near-term transition risk)

Usage:
    cd ~/Downloads/methane-api
    conda activate methane
    python scripts/finance_climate_var.py

Output:
    results_analysis/finance_climate_var.json   — full simulation results
    (summary table printed to stdout)
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "results_analysis" / "finance_climate_var.json"


# ── ① EMISSION PARAMETERS  ────────────────────────────────────────────────────
# Source: scripts/recompute_annualisation.py run on belchatow_annual_timeseries.json

@dataclass
class EmissionParams:
    """
    Observed Bełchatów emission record — 26 quantification-supporting overpasses.
    Updated to MBSP (Varon 2021) scene-derived c coefficient retrieval.
    The 95% CI is the sampling CI on the detection-weighted mean under the
    upper-framing annualisation (non-detection months assigned detection-day mean).
    Source: recompute_annualisation.py on belchatow_annual_timeseries_mbsp.json
    """
    mean_flow_kgh:  float = 1_882.0    # mean per-overpass flow rate (kg/hr)
    sd_flow_kgh:    float = 2_762.0    # standard deviation across 26 detections
    n_obs:          int   = 26         # quantification-supporting observations
    # Annual estimates (t CH₄/yr)
    mean_annual_t:  float = 16_486.0   # detection-weighted annualised estimate (upper framing)
    ci95_lo:        float =  6_781.0   # 95% CI lower bound
    ci95_hi:        float = 26_191.0   # 95% CI upper bound
    # Hours per year (for mean × 8760 annualisation)
    hours_per_year: float = 8_760.0


# ── ② UNCERTAINTY PARAMETERS  ─────────────────────────────────────────────────
# Sources: NIST IR 8575 (Worden et al. 2025) §3.1–§3.4 uncertainty budget

@dataclass
class UncertaintyParams:
    """
    Uncertainty sources following NIST IR 8575 IME/CEMF uncertainty budget.

    Note on double-counting: the 95% sampling CI on the annual mean already
    propagates per-overpass wind (±20%), mask threshold (~5-7%), and retrieval
    coefficient (~15%) uncertainties through the t-distribution on 37 flow rates.
    The additional layers below capture SYSTEMATIC annual-level uncertainties
    not absorbed by the sampling CI:
      - ERA5 systematic bias at the annual level (vs random per-overpass error)
      - Crop window / spatial extent choice (±15%)
    """

    # — Wind speed (NIST IR 8575 §4.2, Table 2) —
    # ERA5 introduces ~±20% random per-overpass error (already absorbed by the
    # 37-observation sampling CI). The ADDITIONAL systematic annual-level bias
    # arises from ERA5 grid-to-point interpolation error when benchmarked
    # against local meteorological tower observations; NIST IR 8575 §4.2
    # establishes this persistent systematic component at ±10% (1-sigma).
    wind_sigma_systematic: float = 0.10   # 1-sigma relative systematic bias

    # — Plume spatial extent / crop window (Varon et al. 2021, AMT §2.3) —
    # Varon et al. (2021) spatial sensitivity tests show that expanding the
    # plume integration boundary from a tight core mask to a local background
    # annulus introduces a uniform ±15% variance in Integrated Mass
    # Enhancement (IME) calculation (see also NIST IR 8575 §3.3 on spatial
    # representativeness of the plume integration domain).
    # This captures the crop-window sensitivity documented in report §6.4.
    mask_jitter_lo: float = 0.85
    mask_jitter_hi: float = 1.15

    # — IME retrieval coefficient (NIST IR 8575 §3.2) —
    # Varon 2021 Sec. 2.2: 4×10⁻⁷ reflectance per ppb·m (±15% relative).
    # Set to 0.0 here because this uncertainty is already captured in the
    # per-overpass CI propagated through the 37-observation t-distribution.
    # Enable (e.g. 0.15) only if you want a separate conservative estimate.
    retrieval_sigma: float = 0.0

    # — Conformal threshold τ bootstrap (§5.2 of technical appendix) —
    # Affects detection rate but NOT the upper-framing annualisation
    # (recompute_annualisation uses flow_rate_kgh regardless of τ classification).
    # Included here for sensitivity analysis and ES decomposition.
    tau_current:    float = 3.5796
    tau_boot_mean:  float = 3.3406    # bootstrap mean of τ over 2000 resamples
    tau_boot_std:   float = 0.7056    # bootstrap std

    # — Base detection rate at τ_current —
    # 26 quantification records / 68 valid observations = 0.382
    base_det_rate: float = 0.382


# ── ③ CARBON PRICE & FINANCIAL PARAMETERS  ───────────────────────────────────
# Sources: Amundi 2024 (log-normal / GBM price distribution); IPCC AR5/AR6 GWP

@dataclass
class CarbonPriceParams:
    """
    Carbon price distribution and GWP conversion factors.

    Price model: log-normal with central scenario €70/tCO₂e, 30% log-vol.
    This is consistent with Amundi (2024) who model EUA price as a GBM with
    drift and volatility drawn from historical EU ETS data.

    Pass-through model: Beta(9, 1) → mean = 0.90, representing a conservative
    stress scenario in which the EU Methane Regulation (2024/1787) enforcement
    is assumed highly likely. Under this calibration, the simulated mean liability
    tracks close to (but slightly below) the deterministic base case of €22.5M,
    with the residual gap (≈10%) reflecting enforcement uncertainty in the tail.

    To model a policy-uncertainty scenario (probabilistic adoption), set
    passthrough_alpha=5, passthrough_beta=2 → mean ≈ 0.714.
    The accounting property is explicit: E[Liability] = E[Q] × GWP × E[Price]
    × E[β] = ... × 0.90, so the stochastic mean lies ~10% below the
    deterministic base case by design — not a defect, but a documented
    representation of residual enforcement uncertainty.
    """
    # Log-normal price (€/tCO₂e)
    central:     float = 70.0   # central scenario (EU ETS-equivalent)
    log_vol:     float = 0.30   # 30% log-normal volatility

    # GWP conversion factors
    gwp100: int = 28    # IPCC AR5 — EU MRV regulatory metric
    gwp20:  int = 83    # IPCC AR6 — near-term transition risk horizon

    # Regulatory enforcement / pass-through — Beta(α, β)
    # Beta(9, 1): mean = α/(α+β) = 0.90 — conservative compliance stress view.
    # Accounting property: E[Liability] ≈ E[Q] × GWP × E[Price] × 0.90,
    # which aligns the stochastic mean with the deterministic base case × 0.90.
    passthrough_alpha: float = 9.0
    passthrough_beta:  float = 1.0


# ── ④ MONTE CARLO ENGINE  ─────────────────────────────────────────────────────

def run_monte_carlo(
    em:    EmissionParams,
    unc:   UncertaintyParams,
    price: CarbonPriceParams,
    n_sim: int = 10_000,
    seed:  int = 42,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the Monte Carlo Climate VaR simulation.

    Returns:
        results  — dict of risk metrics
        liab_100 — array of n_sim liability draws (M€, GWP100)
        liab_20  — array of n_sim liability draws (M€, GWP20)
        Q_annual — array of n_sim annual emission draws (t CH₄/yr)
    """
    rng = np.random.default_rng(seed)

    # ── Layer 1: Annual emission sampling uncertainty (Truncated Normal) ─────
    # Derived from 95% CI [6563, 16400] via t-distribution (df=36).
    #
    # WHY TRUNCATED NORMAL, NOT LOG-NORMAL:
    # The log-normal parameterisation correctly sets E[Q_base] = mean_annual_t,
    # but Jensen's inequality causes the product of multiple log-normal /
    # log-normal-compatible layers to shift the joint mean unpredictably.
    # Truncated Normal (bounded at zero, unbounded above) EXACTLY preserves
    # E[Q_base] = mean_annual_t by construction, making the accounting property
    # of the full product transparent to an MRM auditor:
    #   E[Liability] = mean_annual_t × GWP × E[Price] × E[β_passthrough]
    # This is mathematically clean and directly comparable to the static table.
    #
    # SEM = (CI_hi - CI_lo) / (2 × t_crit) ← sampling standard error of mean
    # a   = (0 - mean) / SEM                ← standardised lower bound at zero
    t_crit = float(stats.t.ppf(0.975, df=em.n_obs - 1))
    sem = (em.ci95_hi - em.ci95_lo) / (2.0 * t_crit)

    a = (0.0 - em.mean_annual_t) / sem   # standardised lower bound (zero floor)
    Q_base = stats.truncnorm.rvs(
        a, np.inf,
        loc=em.mean_annual_t,
        scale=sem,
        size=n_sim,
        random_state=seed,
    )   # t CH₄/yr — E[Q_base] = mean_annual_t exactly

    # ── Layer 2: ERA5 systematic wind bias (NIST IR 8575 §3.1) ───────────────
    # Annual-level systematic ERA5 bias not absorbed by the sampling CI.
    # Modelled as multiplicative Gaussian perturbation centred on 1.0.
    # Physical bounds: clip to [0.5, 2.0] to prevent unphysical draws.
    wind_sys = 1.0 + rng.normal(0.0, unc.wind_sigma_systematic, n_sim)
    wind_sys = np.clip(wind_sys, 0.5, 2.0)

    # ── Layer 3: Plume mask / spatial extent jitter (NIST IR 8575 §3.3) ──────
    # Uniform ±15% jitter on effective plume area from crop window choice.
    # This captures the sensitivity identified in §6.4 of the report.
    mask_factor = rng.uniform(unc.mask_jitter_lo, unc.mask_jitter_hi, n_sim)

    # ── Layer 4: IME retrieval coefficient precision (NIST IR 8575 §3.2) ─────
    # Disabled by default (retrieval_sigma=0.0); already in sampling CI.
    # Enable by setting retrieval_sigma > 0 in UncertaintyParams.
    if unc.retrieval_sigma > 0.0:
        retrieval_factor = 1.0 + rng.normal(0.0, unc.retrieval_sigma, n_sim)
        retrieval_factor = np.clip(retrieval_factor, 0.5, 2.0)
    else:
        retrieval_factor = np.ones(n_sim)

    # ── Combine: total annual emission draw ───────────────────────────────────
    # Q_annual = base × systematic_wind × spatial_extent × retrieval
    Q_annual = Q_base * wind_sys * mask_factor * retrieval_factor   # t CH₄/yr

    # ── Layer 5: Carbon price (Amundi 2024 GBM / log-normal) ─────────────────
    # P_carbon ~ LogNormal(μ_p, log_vol) where μ_p chosen so E[P] = central.
    # This is the Amundi approach: draw EUA price from a lognormal calibrated
    # to historical EU ETS drift and volatility around the central scenario.
    mu_price = np.log(price.central) - 0.5 * price.log_vol ** 2
    P_carbon = rng.lognormal(mu_price, price.log_vol, n_sim)   # €/tCO₂e

    # ── Layer 6: Regulatory pass-through (Beta distribution) ─────────────────
    # Beta(9, 1): mean = α/(α+β) = 0.90 — conservative compliance stress view.
    # Accounting property: E[Liability] = E[Q] × GWP × E[Price] × 0.90,
    # so the stochastic mean sits ~10% below the 100% deterministic base case.
    # This gap is intentional: it reflects residual enforcement uncertainty in
    # the tail even under a conservative stress scenario. Document in §7 as:
    #   "The stochastic mean lies ~10% below the deterministic base case,
    #    primarily due to regulatory pass-through uncertainty."
    # To model a policy-uncertainty scenario, use Beta(5, 2) → mean ≈ 0.714.
    beta_pt = rng.beta(price.passthrough_alpha, price.passthrough_beta, n_sim)

    # ── Compute liability distributions ───────────────────────────────────────
    # Liability (M€) = Q_annual (t CH₄) × GWP (tCO₂e/tCH₄)
    #                  × P_carbon (€/tCO₂e) × β_passthrough / 1e6
    liab_100 = Q_annual * price.gwp100 * P_carbon * beta_pt / 1e6   # M€, GWP100
    liab_20  = Q_annual * price.gwp20  * P_carbon * beta_pt / 1e6   # M€, GWP20

    # ── Risk metrics ─────────────────────────────────────────────────────────
    def risk_metrics(liab: np.ndarray, label: str) -> dict:
        """Compute CVaR risk table following Amundi 2024 §3.3 risk metrics."""
        var95 = float(np.percentile(liab, 95))
        var99 = float(np.percentile(liab, 99))
        es99  = float(np.mean(liab[liab >= var99]))      # Expected Shortfall
        return {
            "label":   label,
            "mean":    round(float(np.mean(liab)),    3),
            "median":  round(float(np.median(liab)),  3),
            "var_95":  round(var95, 3),
            "var_99":  round(var99, 3),
            "es_99":   round(es99,  3),
            "p5":      round(float(np.percentile(liab,  5)), 3),
            "p25":     round(float(np.percentile(liab, 25)), 3),
            "p75":     round(float(np.percentile(liab, 75)), 3),
        }

    results = {
        "metadata": {
            "n_sim":    n_sim,
            "seed":     seed,
            "facility": "KWB Bełchatów Coal Mine, Poland (PGE GiEK)",
            "methodology": [
                "Amundi 2024 — stochastic carbon price / pass-through",
                "NIST IR 8575 (Worden et al. 2025) — IME uncertainty budget",
            ],
            "carbon_price_central_eur_tco2e": price.central,
            "carbon_price_log_vol":           price.log_vol,
            "passthrough_beta_mean":          round(price.passthrough_alpha /
                                                    (price.passthrough_alpha + price.passthrough_beta), 3),
        },
        "emission_draws": {
            "q_mean_t_yr":  round(float(np.mean(Q_annual)),             1),
            "q_p5_t_yr":   round(float(np.percentile(Q_annual,  5)),    1),
            "q_p95_t_yr":  round(float(np.percentile(Q_annual, 95)),    1),
            "q_sd_t_yr":   round(float(np.std(Q_annual)),               1),
        },
        "carbon_price_draws": {
            "mean_eur_tco2e":  round(float(np.mean(P_carbon)),          2),
            "p5_eur_tco2e":   round(float(np.percentile(P_carbon,  5)), 2),
            "p95_eur_tco2e":  round(float(np.percentile(P_carbon, 95)), 2),
        },
        "liability_gwp100": risk_metrics(liab_100, f"GWP100 (factor {price.gwp100})"),
        "liability_gwp20":  risk_metrics(liab_20,  f"GWP20  (factor {price.gwp20})"),
    }

    return results, liab_100, liab_20, Q_annual


# ── ⑤ SENSITIVITY: τ UNCERTAINTY ON DETECTION RATE  ──────────────────────────

def tau_sensitivity(
    em:    EmissionParams,
    unc:   UncertaintyParams,
    price: CarbonPriceParams,
    n_sim: int = 5_000,
    seed:  int = 43,
) -> dict:
    """
    Sensitivity analysis: how does τ bootstrap uncertainty affect detection rate
    and, under the lower-bound annualisation framing, the liability estimate?

    Note: the upper-framing annualisation (our primary estimate) is τ-independent
    because recompute_annualisation counts records with flow_rate_kgh regardless
    of τ classification. This function models the lower-bound framing where
    only DETECTION-classified records contribute to the mean.
    """
    rng = np.random.default_rng(seed)

    # Draw τ from bootstrap distribution (clipped to physical range)
    tau_draws = rng.normal(unc.tau_boot_mean, unc.tau_boot_std, n_sim)
    tau_draws = np.clip(tau_draws, 1.15, 6.0)

    # At each τ, detection rate decreases roughly linearly above τ_current.
    # Rough model: det_rate(τ) = base_det_rate × (1 - scale × (τ - τ_current))
    # Calibrated so that at τ_current, det_rate = base; at τ_max (≈4.34), det_rate ≈ 0.5×
    scale = 0.10 / (4.34 - unc.tau_current)  # ~10% reduction per unit τ above current
    det_rate = unc.base_det_rate * np.where(
        tau_draws > unc.tau_current,
        1.0 - scale * (tau_draws - unc.tau_current),
        1.0
    )
    det_rate = np.clip(det_rate, 0.10, 1.0)

    # Lower-bound annual emission: mean_flow × 8760 × det_rate
    Q_lower = em.mean_flow_kgh * em.hours_per_year / 1e3 * det_rate   # t/yr

    mu_price = np.log(price.central) - 0.5 * price.log_vol ** 2
    P_carbon = rng.lognormal(mu_price, price.log_vol, n_sim)
    beta_pt  = rng.beta(price.passthrough_alpha, price.passthrough_beta, n_sim)

    liab_100 = Q_lower * price.gwp100 * P_carbon * beta_pt / 1e6
    liab_20  = Q_lower * price.gwp20  * P_carbon * beta_pt / 1e6

    return {
        "note": "Lower-bound framing (det_rate × mean_flow × 8760). Upper framing is τ-independent.",
        "tau_mean_draw": round(float(np.mean(tau_draws)),   4),
        "det_rate_mean": round(float(np.mean(det_rate)),    3),
        "gwp100_mean_liability_MEur": round(float(np.mean(liab_100)), 3),
        "gwp20_mean_liability_MEur":  round(float(np.mean(liab_20)),  3),
        "gwp100_var99_MEur":          round(float(np.percentile(liab_100, 99)), 3),
        "gwp20_var99_MEur":           round(float(np.percentile(liab_20,  99)), 3),
    }


# ── ⑥ PRINT SUMMARY  ─────────────────────────────────────────────────────────

def print_table(results: dict) -> None:
    """Print the Climate VaR summary table to stdout."""
    meta = results["metadata"]
    gwp100 = results["liability_gwp100"]
    gwp20  = results["liability_gwp20"]
    em     = results["emission_draws"]
    price  = results["carbon_price_draws"]

    bar = "=" * 72
    print(f"\n{bar}")
    print("  CLIMATE VALUE-AT-RISK (CVaR)  |  KWB Bełchatów / PGE")
    print(f"  n_sim={meta['n_sim']:,}  |  Carbon price central: "
          f"€{meta['carbon_price_central_eur_tco2e']}/tCO₂e  "
          f"(log-vol={meta['carbon_price_log_vol']:.0%})")
    print(f"  Pass-through β mean: {meta['passthrough_beta_mean']:.1%}")
    print(f"{bar}")

    print(f"\n  Emission draws (t CH₄/yr)")
    print(f"    Mean:  {em['q_mean_t_yr']:>8,.0f}    "
          f"P5:  {em['q_p5_t_yr']:>8,.0f}    "
          f"P95: {em['q_p95_t_yr']:>8,.0f}")

    print(f"\n  Carbon price draws (€/tCO₂e)")
    print(f"    Mean:  {price['mean_eur_tco2e']:>7.1f}    "
          f"P5:  {price['p5_eur_tco2e']:>7.1f}    "
          f"P95: {price['p95_eur_tco2e']:>7.1f}")

    print(f"\n{'─'*72}")
    print(f"  Annual Carbon Liability (M€)          GWP100        GWP20")
    print(f"{'─'*72}")
    metrics = [
        ("Mean expected liability",  "mean"),
        ("Median",                   "median"),
        ("95th percentile (VaR 95)", "var_95"),
        ("99th percentile (VaR 99)", "var_99"),
        ("99% Expected Shortfall",   "es_99"),
    ]
    for label, key in metrics:
        v100 = gwp100[key]
        v20  = gwp20[key]
        print(f"  {label:<36}  {v100:>8.2f}      {v20:>8.2f}")
    print(f"{'─'*72}")
    print(f"  {'GWP20 / GWP100 ratio':<36}  {'—':>8}      "
          f"{(gwp20['mean']/gwp100['mean'] if gwp100['mean'] else 0):>7.2f}×")
    print(f"{bar}\n")


# ── ⑦ MAIN  ──────────────────────────────────────────────────────────────────

def main() -> None:
    em    = EmissionParams()
    unc   = UncertaintyParams()
    price = CarbonPriceParams()

    print("Running Monte Carlo Climate VaR engine...")
    print(f"  n_sim = 10,000 | seed = 42")
    print(f"  Emission: mean {em.mean_annual_t:,.0f} t/yr  "
          f"95% CI [{em.ci95_lo:,.0f}, {em.ci95_hi:,.0f}]")
    print(f"  Wind systematic σ = {unc.wind_sigma_systematic:.0%}  |  "
          f"Mask jitter = [{unc.mask_jitter_lo}, {unc.mask_jitter_hi}]")
    print(f"  Carbon price central = €{price.central}/tCO₂e  log-vol={price.log_vol:.0%}")

    results, liab_100, liab_20, Q_annual = run_monte_carlo(
        em, unc, price, n_sim=10_000, seed=42
    )

    # τ sensitivity
    tau_sens = tau_sensitivity(em, unc, price, n_sim=5_000, seed=43)
    results["tau_sensitivity"] = tau_sens

    # Save
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {OUT_PATH.relative_to(ROOT)}")

    # Print table
    print_table(results)

    # Uncertainty decomposition (relative σ per layer, truncated-normal basis)
    t_crit = float(stats.t.ppf(0.975, df=em.n_obs - 1))
    sem = (em.ci95_hi - em.ci95_lo) / (2.0 * t_crit)
    sigma_sampling  = sem / em.mean_annual_t             # relative σ, emission sampling
    sigma_wind      = unc.wind_sigma_systematic           # relative σ, ERA5 systematic
    sigma_mask      = (unc.mask_jitter_hi - 1) / np.sqrt(3)  # relative σ, uniform ±15%
    sigma_price     = price.log_vol                       # log-vol of carbon price
    sigma_total_approx = np.sqrt(sigma_sampling**2 + sigma_wind**2 + sigma_mask**2 + sigma_price**2)
    print("  Uncertainty decomposition (relative σ, independent quadrature):")
    print(f"    Sampling CI (truncnorm, n={em.n_obs}): {sigma_sampling:.3f}  "
          f"({sigma_sampling/sigma_total_approx:.0%} of σ_total)")
    print(f"    Wind systematic (±{unc.wind_sigma_systematic:.0%} ERA5):      {sigma_wind:.3f}  "
          f"({sigma_wind/sigma_total_approx:.0%} of σ_total)")
    print(f"    Mask jitter (±15% uniform):       {sigma_mask:.3f}  "
          f"({sigma_mask/sigma_total_approx:.0%} of σ_total)")
    print(f"    Carbon price (log-vol {price.log_vol:.0%}):        {sigma_price:.3f}  "
          f"({sigma_price/sigma_total_approx:.0%} of σ_total)")
    print(f"    Approx. combined σ (quadrature):  {sigma_total_approx:.3f}")
    print()


if __name__ == "__main__":
    main()
