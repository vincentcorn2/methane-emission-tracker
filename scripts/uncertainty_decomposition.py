"""
scripts/uncertainty_decomposition.py
=====================================
WS1 Deliverable: 4-source uncertainty decomposition for confirmed CH4 detections.

For each confirmed 2024 detection, produces:

  Source                      | 1σ (%)  | Method
  ────────────────────────────────────────────────────────────
  Wind speed (ERA5 → site)    |  ±20%   | Literature prior (Varon 2021)
  Sensitivity coefficient      |  ±15%   | Literature prior (Varon 2021)
  Plume mask threshold         |  ±Z%    | Bootstrap over p* range
  Background annulus           |  ±W%    | Jackknife N/S/E/W quadrants
  Combined (Monte Carlo 10k)   |  ±tot%  | Independence assumed

Key design decisions
────────────────────
• Spatial crop: 500×500 px (5km × 5km) centred on facility coordinates.
  Same crop as used in run_cemf_neurath_belchatow.py.  The full tile is
  NOT loaded into memory (avoids OOM on 10980×10980 arrays).

• Probability thresholds: per-detection, calibrated as multiples of the
  control-mean probability value (not the generic 0.18 CH4NetDetector
  threshold, which applies to rescaled [0,1] sigmoid outputs).

• Bootstrap range: [0.3×p*, min(10×p*, 0.50)] log-spaced, 50 steps.
  This explores ≈1.5 decades around the operating threshold — large
  enough to characterise sensitivity, small enough to stay physical.

• Background jackknife: divides non-plume pixels into N/S/E/W quadrants
  relative to plume centroid; drops one quadrant per run.

References:
  Varon et al. 2021, AMT 14:2771–2785 (Table 2)
  Lakshminarayanan et al. 2017 NeurIPS (epistemic/aleatoric framework)

Usage:
    conda activate methane
    cd methane-api
    python scripts/uncertainty_decomposition.py
    python scripts/uncertainty_decomposition.py --update-quant  # write back to quant.json
    python scripts/uncertainty_decomposition.py --n-mc 20000    # more MC samples
    python scripts/uncertainty_decomposition.py --site neurath
"""

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np

try:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None  # suppress decompression bomb warning
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

try:
    import rasterio
    _RASTERIO_AVAILABLE = True
except ImportError:
    _RASTERIO_AVAILABLE = False

# ── Logging ────────────────────────────────────────────────────────────────────
Path("results_analysis").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("results_analysis/uncertainty_decomposition.log"),
    ],
)
log = logging.getLogger(__name__)

# ── Physical constants (mirrors src/quantification/cemf.py) ───────────────────
DRY_AIR_COLUMN   = 2.1e25      # molecules/m²
MOL_WEIGHT_CH4   = 0.016       # kg/mol
AVOGADRO         = 6.022e23
PIXEL_AREA_20M   = 400.0       # m² at 20m native SWIR resolution
PIXEL_SIZE_20M   = 20.0        # m

SENSITIVITY_COEFF = 4e-7       # reflectance / (ppb·m), Varon 2021 Sec 2.2

# Band indices in 12-band .npy stack
# Order: B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B11 B12
IDX_B11 = 10
IDX_B12 = 11

CROP_PX = 500   # 10m pixels → 5km × 5km crop around facility

# ── Uncertainty literature priors ─────────────────────────────────────────────
SIGMA_WIND_PCT   = 20.0   # ERA5 10m → plume-layer (Varon 2021 AMT Table 2)
SIGMA_COEFF_PCT  = 15.0   # CEMF sensitivity coeff (Varon 2021 AMT Table 2)


# ── Detection registry ────────────────────────────────────────────────────────
# Thresholds from run_cemf_neurath_belchatow.py (calibrated as N× ctrl_mean).
# Wind speeds from quantification.json where available; None → ERA5 fetch/fallback.
DETECTIONS = [
    {
        "site":           "neurath",
        "label":          "Neurath 2024-08-29",
        "scene_id":       "S2B_MSIL1C_20240829T103629_N0511_R008_T32ULB_20240829T124434",
        "acquisition_ts": "2024-08-29T10:36:29Z",
        "lat": 51.038, "lon": 6.616,
        "site_row": 4324, "site_col": 3286,   # T32ULB pixel coords
        "prob_thresh": 0.0007,                 # 5× ctrl_mu=0.000149
        "tif": "results_bitemporal/neurath/original_S2B_MSIL1C_20240829T103629_N0511_R008_T32ULB_20240829T124434.tif",
        "npy": "data/npy_cache/S2B_MSIL1C_20240829T103629_N0511_R008_T32ULB_20240829T124434.npy",
        "wind_ms":          2.6612,            # ERA5 from quantification.json
        "wind_source":      "ERA5_reanalysis",
        "reported_flow_kgh": 85.0,
    },
    {
        "site":           "belchatow",
        "label":          "Bełchatów 2024-07-10",
        "scene_id":       "S2A_MSIL1C_20240710T095031_N0510_R079_T34UCB_20240710T133148",
        "acquisition_ts": "2024-07-10T09:50:31Z",
        "lat": 51.264, "lon": 19.331,
        "site_row": 1949, "site_col": 8356,   # T34UCB pixel coords
        "prob_thresh": 0.002,                  # 5× ctrl_mu=0.000499
        "tif": "results_bitemporal/belchatow/original_S2A_MSIL1C_20240710T095031_N0510_R079_T34UCB_20240710T133148.tif",
        "npy": "data/npy_cache/S2A_MSIL1C_20240710T095031_N0510_R079_T34UCB_20240710T133148.npy",
        "wind_ms":          3.3183,            # ERA5 from quantification.json
        "wind_source":      "ERA5_reanalysis",
        "reported_flow_kgh": 1071.0,
    },
    {
        "site":           "neurath",
        "label":          "Neurath 2024-06-25",
        "scene_id":       "S2A_MSIL1C_20240625T103631_N0510_R008_T32ULB_20240625T142035",
        "acquisition_ts": "2024-06-25T10:36:31Z",
        "lat": 51.038, "lon": 6.616,
        "site_row": 4324, "site_col": 3286,
        "prob_thresh": 0.016,                  # 3× ctrl_mean=0.00526
        "tif": "results_bitemporal/neurath/original_S2A_MSIL1C_20240625T103631_N0510_R008_T32ULB_20240625T142035.tif",
        "npy": "data/npy_cache/S2A_MSIL1C_20240625T103631_N0510_R008_T32ULB_20240625T142035.npy",
        "wind_ms":          None,              # not in quantification.json → fetch/fallback
        "wind_source":      None,
        "reported_flow_kgh": 338.0,
    },
    {
        "site":           "belchatow",
        "label":          "Bełchatów 2024-08-24",
        "scene_id":       "S2B_MSIL1C_20240824T094549_N0511_R079_T34UCB_20240824T115611",
        "acquisition_ts": "2024-08-24T09:45:49Z",
        "lat": 51.264, "lon": 19.331,
        "site_row": 1949, "site_col": 8356,
        "prob_thresh": 0.0002,                 # 5× ctrl_mean=3.9e-05
        "tif": "results_bitemporal/belchatow/original_S2B_MSIL1C_20240824T094549_N0511_R079_T34UCB_20240824T115611.tif",
        "npy": "data/npy_cache/S2B_MSIL1C_20240824T094549_N0511_R079_T34UCB_20240824T115611.npy",
        "wind_ms":          2.19,              # pre-retrieved from session record
        "wind_source":      "ERA5_reanalysis",
        "reported_flow_kgh": 426.0,
    },
    {
        "site":           "turkmenistan",
        "label":          "Turkmenistan 2021-01-01 (L2A, pipeline validation)",
        "scene_id":       "S2B_T40SBE_20210101_Turkmenistan",
        "acquisition_ts": "2021-01-01T07:43:00Z",
        "lat": 35.173, "lon": 54.754,
        "site_row": 5240, "site_col": 4774,
        "prob_thresh": 0.001,
        "tif": "results_bitemporal/turkmenistan/original_S2B_T40SBE_20210101_Turkmenistan.tif",
        "npy": "data/npy_cache/S2B_T40SBE_20210101_Turkmenistan.npy",
        "wind_ms":           1.58,
        "wind_source":       "ERA5_reanalysis",
        "reported_flow_kgh": 225.0,
    },
]


# ── Spatial crop ──────────────────────────────────────────────────────────────

def load_site_crop(det: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load B11, B12, and probability map for a 500×500px crop around the facility.
    Returns (b11_10m, b12_10m, prob_10m) all as float32 [0,1], shape (500, 500).
    """
    npy_path = Path(det["npy"])
    tif_path = Path(det["tif"])
    row, col = det["site_row"], det["site_col"]
    half = CROP_PX // 2
    r0, r1 = row - half, row + half
    c0, c1 = col - half, col + half

    # Memory-mapped npy — only reads the crop slice
    arr = np.load(npy_path, mmap_mode="r")
    b11_10m = arr[r0:r1, c0:c1, IDX_B11].astype(np.float32) / 255.0
    b12_10m = arr[r0:r1, c0:c1, IDX_B12].astype(np.float32) / 255.0
    log.info("  B11 crop: %s  range=[%.4f, %.4f]",
             b11_10m.shape, b11_10m.min(), b11_10m.max())
    log.info("  B12 crop: %s  range=[%.4f, %.4f]",
             b12_10m.shape, b12_10m.min(), b12_10m.max())

    # Load TIF probability map (PIL crops without loading full image)
    if _PIL_AVAILABLE:
        img = Image.open(tif_path)
        prob_10m = np.array(img.crop((c0, r0, c1, r1))).astype(np.float32)
    elif _RASTERIO_AVAILABLE:
        with rasterio.open(tif_path) as src:
            window = rasterio.windows.Window(c0, r0, CROP_PX, CROP_PX)
            prob_10m = src.read(1, window=window).astype(np.float32)
    else:
        raise ImportError("Need either Pillow or rasterio to read TIF files")

    # Normalize prob map to [0, 1] regardless of saved dtype
    # PIL may return uint16 (0-65535) or uint8 (0-255) or float32 (0-1)
    if prob_10m.max() > 1.0:
        prob_10m = prob_10m / prob_10m.max()
    log.info("  Prob crop: %s  range=[%.6f, %.6f]",
             prob_10m.shape, prob_10m.min(), prob_10m.max())
    return b11_10m, b12_10m, prob_10m


# ── CEMF + IME (inline, no torch dependency) ──────────────────────────────────

def cemf_flow_rate(b11: np.ndarray, b12: np.ndarray,
                   mask_20m: np.ndarray, wind_ms: float,
                   sensitivity: float = SENSITIVITY_COEFF) -> dict:
    """
    Compute CEMF-derived flow rate for a given plume mask and wind speed.
    b11, b12: float32 reflectance [0,1] at 20m.
    mask_20m: binary (0/1) uint8 or bool at 20m.
    Returns dict with flow_kgh, total_mass_kg, plume_length_m, valid, warning.
    """
    mask_bool  = mask_20m.astype(bool)
    background = ~mask_bool

    if background.sum() < 50:
        return {"flow_kgh": 0.0, "total_mass_kg": 0.0, "plume_length_m": 0.0,
                "n_plume_pixels": int(mask_bool.sum()),
                "valid": False, "warning": "too_few_background_pixels"}

    mu_b11 = float(b11[background].mean())
    mu_b12 = float(b12[background].mean())

    if mu_b11 < 1e-6:
        return {"flow_kgh": 0.0, "total_mass_kg": 0.0, "plume_length_m": 0.0,
                "n_plume_pixels": int(mask_bool.sum()),
                "valid": False, "warning": "near_zero_b11_background"}

    if mask_bool.sum() < 4:
        return {"flow_kgh": 0.0, "total_mass_kg": 0.0, "plume_length_m": 0.0,
                "n_plume_pixels": 0, "valid": False, "warning": "empty_plume_mask"}

    d_b11 = b11 - mu_b11
    d_b12 = b12 - mu_b12
    dxch4 = np.clip((d_b12 - 0.5 * d_b11) / (mu_b11 * sensitivity), 0, None)

    mass_per_pixel = (
        dxch4[mask_bool] * 1e-9
        * DRY_AIR_COLUMN * MOL_WEIGHT_CH4 / AVOGADRO
        * PIXEL_AREA_20M
    )
    total_mass = float(mass_per_pixel.sum())

    rows, cols = np.where(mask_bool)
    plume_length_m = max(
        float(max(rows.max() - rows.min(), cols.max() - cols.min())) * PIXEL_SIZE_20M,
        PIXEL_SIZE_20M
    )
    residence_s = plume_length_m / max(wind_ms, 0.5)
    flow_kgh    = round(total_mass / residence_s * 3600.0, 4)

    return {
        "flow_kgh":       flow_kgh,
        "total_mass_kg":  total_mass,
        "plume_length_m": plume_length_m,
        "n_plume_pixels": int(mask_bool.sum()),
        "valid":          True,
        "warning":        None,
    }


def cemf_flow_with_bg_mask(b11, b12, mask_20m, bg_bool, wind_ms,
                            sensitivity=SENSITIVITY_COEFF):
    """CEMF with an explicit background boolean mask (for bg jackknife)."""
    if bg_bool.sum() < 50:
        return {"flow_kgh": 0.0, "valid": False, "warning": "too_few_bg_pixels"}

    mu_b11 = float(b11[bg_bool].mean())
    mu_b12 = float(b12[bg_bool].mean())
    if mu_b11 < 1e-6:
        return {"flow_kgh": 0.0, "valid": False, "warning": "near_zero_b11"}

    d_b11 = b11 - mu_b11
    d_b12 = b12 - mu_b12
    dxch4 = np.clip((d_b12 - 0.5 * d_b11) / (mu_b11 * sensitivity), 0, None)

    plume_bool = mask_20m.astype(bool)
    if plume_bool.sum() < 4:
        return {"flow_kgh": 0.0, "valid": False, "warning": "empty_plume"}

    mass_per_pixel = (
        dxch4[plume_bool] * 1e-9
        * DRY_AIR_COLUMN * MOL_WEIGHT_CH4 / AVOGADRO
        * PIXEL_AREA_20M
    )
    total_mass = float(mass_per_pixel.sum())
    rows, cols = np.where(plume_bool)
    plume_length_m = max(
        float(max(rows.max() - rows.min(), cols.max() - cols.min())) * PIXEL_SIZE_20M,
        PIXEL_SIZE_20M
    )
    flow_kgh = round(total_mass / max(plume_length_m / max(wind_ms, 0.5), 1e-6) * 3600.0, 4)
    return {"flow_kgh": flow_kgh, "total_mass_kg": total_mass, "valid": True, "warning": None}


# ── Wind fetching ─────────────────────────────────────────────────────────────

WIND_FALLBACK_MS = 3.5

def fetch_era5_wind(lat, lon, ts):
    """ERA5 wind lookup with fallback to climatological 3.5 m/s."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from src.ingestion.era5_client import ERA5Client, FALLBACK_WIND_SPEED
        result = ERA5Client().get_wind(lat, lon, ts)
        wind_ms   = float(result["wind_speed_ms"])
        wind_dir  = result.get("wind_dir_deg") or 0.0
        # Detect fallback: ERA5Client returns FALLBACK_WIND_SPEED when CDS unavailable
        if wind_ms == FALLBACK_WIND_SPEED and result.get("wind_source", "").startswith("climatological"):
            log.warning("    ERA5 returned fallback wind %.1f m/s", wind_ms)
            return wind_ms, result.get("wind_source", f"climatological_fallback_{wind_ms}ms")
        log.info("    ERA5 wind: %.3f m/s, dir=%.1f°", wind_ms, wind_dir)
        return wind_ms, "ERA5_reanalysis"
    except Exception as e:
        log.warning("    ERA5 unavailable (%s) → %.1f m/s fallback", type(e).__name__, WIND_FALLBACK_MS)
        return WIND_FALLBACK_MS, f"climatological_fallback_{WIND_FALLBACK_MS}ms"


# ── Source 3: Mask threshold bootstrap ────────────────────────────────────────

def bootstrap_mask_threshold(b11, b12, prob_10m, wind_ms, canonical_thresh,
                              n_steps=50):
    """
    Bootstrap flow rate over thresholds from 0.3×p* to min(10×p*, 0.50).
    Reports σ/mean as % uncertainty.
    """
    lo = 0.3 * canonical_thresh
    hi = min(10.0 * canonical_thresh, 0.50)
    thresholds = np.geomspace(lo, hi, n_steps)

    flows = []
    for p_star in thresholds:
        mask_10m = (prob_10m >= p_star).astype(np.uint8)
        mask_20m = mask_10m[::2, ::2]
        if mask_20m.sum() < 4:
            continue
        result = cemf_flow_rate(b11[::2, ::2], b12[::2, ::2], mask_20m, wind_ms)
        if result["valid"] and result["flow_kgh"] > 0:
            flows.append({"p_star": float(p_star), "flow_kgh": result["flow_kgh"],
                          "n_px": result["n_plume_pixels"]})

    if len(flows) < 4:
        log.warning("    Mask bootstrap: only %d valid steps — σ unreliable", len(flows))
        return {"sigma_pct": float("nan"), "n_valid": len(flows),
                "note": "insufficient_range", "flows_summary": flows}

    flow_vals = [x["flow_kgh"] for x in flows]
    mean_flow = float(np.mean(flow_vals))
    std_flow  = float(np.std(flow_vals, ddof=1))
    sigma_pct = round(100.0 * std_flow / mean_flow, 1) if mean_flow > 0 else float("nan")

    log.info("    Mask bootstrap: n=%d  mean=%.1f  std=%.1f  σ=%.1f%%",
             len(flows), mean_flow, std_flow, sigma_pct)

    # Summary: key quantiles + canonical threshold
    key_idxs = [0, len(flows)//4, len(flows)//2, 3*len(flows)//4, len(flows)-1]
    flows_summary = [flows[i] for i in sorted(set(key_idxs))]
    return {
        "sigma_pct":        sigma_pct,
        "n_valid":          len(flows),
        "threshold_range":  [round(float(lo), 6), round(float(hi), 6)],
        "canonical_thresh": canonical_thresh,
        "mean_flow_kgh":    round(mean_flow, 2),
        "std_flow_kgh":     round(std_flow, 2),
        "flows_summary":    [{k: round(v, 5) if isinstance(v, float) else v
                              for k, v in f.items()} for f in flows_summary],
    }


# ── Source 4: Background annulus jackknife ────────────────────────────────────

def jackknife_background_annulus(b11_20m, b12_20m, mask_20m, wind_ms):
    """
    Jackknife over 4 directional quadrants of the background pixels.
    For each run, use only the 3 quadrants that are NOT dropped.
    """
    plume_bool = mask_20m.astype(bool)
    bg_bool    = ~plume_bool

    if plume_bool.sum() < 4:
        return {"sigma_pct": float("nan"), "note": "empty_plume"}

    rows_p, cols_p = np.where(plume_bool)
    c_row = float(rows_p.mean())
    c_col = float(cols_p.mean())

    H, W = mask_20m.shape
    # Build row and column index grids
    ri = np.arange(H, dtype=float).reshape(-1, 1) * np.ones((1, W))
    ci = np.ones((H, 1), dtype=float) * np.arange(W, dtype=float).reshape(1, -1)

    quadrants = {
        "N": bg_bool & (ri < c_row),
        "S": bg_bool & (ri >= c_row),
        "W": bg_bool & (ci < c_col),
        "E": bg_bool & (ci >= c_col),
    }

    flows = {}
    for drop_q in list(quadrants.keys()):
        bg_kept = np.zeros((H, W), dtype=bool)
        for q, qm in quadrants.items():
            if q != drop_q:
                bg_kept |= qm.astype(bool)
        result = cemf_flow_with_bg_mask(b11_20m, b12_20m, mask_20m, bg_kept, wind_ms)
        if result["valid"] and result["flow_kgh"] > 0:
            flows[drop_q] = result["flow_kgh"]
            log.info("    BG jackknife (drop %s): %.1f kg/h", drop_q, result["flow_kgh"])
        else:
            log.warning("    BG jackknife (drop %s): invalid (%s)",
                        drop_q, result.get("warning", "?"))

    if len(flows) < 2:
        return {"sigma_pct": float("nan"), "n_valid": len(flows),
                "flows": flows, "note": "too_few_valid_jackknife_runs"}

    flow_vals = list(flows.values())
    mean_flow = float(np.mean(flow_vals))
    std_flow  = float(np.std(flow_vals, ddof=1))
    sigma_pct = round(100.0 * std_flow / mean_flow, 1) if mean_flow > 0 else float("nan")

    log.info("    BG jackknife: n=%d  mean=%.1f  std=%.1f  σ=%.1f%%",
             len(flow_vals), mean_flow, std_flow, sigma_pct)
    return {
        "sigma_pct":         sigma_pct,
        "n_valid":           len(flows),
        "mean_flow_kgh":     round(mean_flow, 2),
        "std_flow_kgh":      round(std_flow, 2),
        "per_quadrant_kgh":  {q: round(v, 2) for q, v in flows.items()},
    }


# ── Combined Monte Carlo ───────────────────────────────────────────────────────

def monte_carlo_combined(base_flow_kgh, sigma_wind_pct, sigma_coeff_pct,
                          sigma_mask_pct, sigma_bg_pct,
                          n_samples=10_000, rng_seed=42):
    """
    10k-sample MC propagation through Q = Q₀ × f_wind × f_coeff × f_mask × f_bg.
    Each fᵢ ~ Normal(1, σᵢ/100) — valid for σ < ~50%.
    Unknown (NaN) sources contribute zero variance.
    """
    rng = np.random.default_rng(rng_seed)

    def safe(v):
        return 0.0 if (v is None or np.isnan(v)) else v / 100.0

    σ_w, σ_c, σ_m, σ_b = safe(sigma_wind_pct), safe(sigma_coeff_pct), \
                           safe(sigma_mask_pct), safe(sigma_bg_pct)

    f_w = np.clip(rng.normal(1.0, σ_w, n_samples), 0.01, None)
    f_c = np.clip(rng.normal(1.0, σ_c, n_samples), 0.01, None)
    f_m = np.clip(rng.normal(1.0, σ_m, n_samples), 0.01, None)
    f_b = np.clip(rng.normal(1.0, σ_b, n_samples), 0.01, None)

    samples   = base_flow_kgh * f_w * f_c * f_m * f_b
    mean_s    = float(np.mean(samples))
    std_s     = float(np.std(samples, ddof=1))
    p5, p95   = float(np.percentile(samples, 5)), float(np.percentile(samples, 95))
    sigma_pct = round(100.0 * std_s / mean_s, 1) if mean_s > 0 else float("nan")
    quad_pct  = round(100.0 * np.sqrt(σ_w**2 + σ_c**2 + σ_m**2 + σ_b**2), 1)

    return {
        "n_samples":             n_samples,
        "mean_kgh":              round(mean_s, 2),
        "std_kgh":               round(std_s, 2),
        "p5_kgh":                round(p5, 2),
        "p95_kgh":               round(p95, 2),
        "sigma_pct_mc":          sigma_pct,
        "sigma_pct_quadrature":  quad_pct,
    }


# ── Print table ───────────────────────────────────────────────────────────────

def print_table(label, base_flow, wind_ms, wind_source, unc):
    """Print a formatted uncertainty budget table for one detection."""
    mc = unc["combined_mc"]
    print()
    print("=" * 68)
    print(f"  {label}")
    print(f"  Reported Q̂: {base_flow:.1f} kg/h   ERA5: {wind_ms:.3f} m/s ({wind_source})")
    print("=" * 68)
    print(f"  {'Source':<33}  {'1σ (%)':>7}  Method")
    print("  " + "-" * 63)

    rows = [
        ("Wind speed (ERA5 → site)",
         unc["wind"]["sigma_pct"],
         "±20%, Varon 2021 AMT Table 2"),
        ("Sensitivity coeff (4e-7)",
         unc["coeff"]["sigma_pct"],
         "±15%, Varon 2021 AMT Table 2"),
        ("Plume mask threshold",
         unc["mask"]["sigma_pct"],
         f"Bootstrap n={unc['mask'].get('n_valid','?')} "
         f"range=[{unc['mask'].get('threshold_range',['?','?'])[0]:.4g},"
         f"{unc['mask'].get('threshold_range',['?','?'])[-1]:.4g}]"),
        ("Background annulus",
         unc["background"]["sigma_pct"],
         f"Jackknife N/S/E/W "
         f"n={unc['background'].get('n_valid','?')}"),
    ]
    for src, sigma, method in rows:
        if sigma is None or (isinstance(sigma, float) and np.isnan(sigma)):
            sigma_str = "   n/a"
        else:
            sigma_str = f"{sigma:>6.1f}%"
        print(f"  {src:<33}  {sigma_str}  {method}")

    print("  " + "-" * 63)
    print(f"  {'Combined (Monte Carlo, 10k)':<33}  {mc['sigma_pct_mc']:>6.1f}%  "
          f"independent sources")
    print(f"  {'  quadrature √Σσ²':<33}  {mc['sigma_pct_quadrature']:>6.1f}%")
    print(f"  {'  90% CI (5th–95th pct)':<33}  "
          f"[{mc['p5_kgh']:.1f}, {mc['p95_kgh']:.1f}] kg/h")
    print()

    lo30 = round(base_flow * 0.70, 1)
    hi30 = round(base_flow * 1.30, 1)
    print(f"  Canonical ±30% bounds: [{lo30}, {hi30}] kg/h")
    print(f"  MC 90% CI:             [{mc['p5_kgh']:.1f}, {mc['p95_kgh']:.1f}] kg/h")

    diff = mc['sigma_pct_mc'] - 30.0
    flag = "⚠  WIDER than canonical" if diff > 5 else ("✓  consistent with canonical" if abs(diff) <= 5 else "✓  narrower than canonical")
    print(f"  Combined σ vs ±30%: {flag} ({mc['sigma_pct_mc']:.1f}% vs 30%)")
    print("=" * 68)


# ── Process one detection ─────────────────────────────────────────────────────

def run_one_detection(det, n_mc):
    label = det["label"]
    log.info("=" * 60)
    log.info("Processing: %s", label)

    # Data paths
    for key in ("tif", "npy"):
        if not Path(det[key]).exists():
            return {"label": label, "status": "error",
                    "error": f"{key}_not_found: {det[key]}"}

    # Wind
    wind_ms     = det["wind_ms"]
    wind_source = det.get("wind_source") or "unknown"
    if wind_ms is None:
        log.info("  Fetching ERA5 wind for %s %s...", det["site"], det["acquisition_ts"][:10])
        wind_ms, wind_source = fetch_era5_wind(det["lat"], det["lon"], det["acquisition_ts"])
        det["wind_ms"]     = wind_ms
        det["wind_source"] = wind_source
    log.info("  Wind: %.4f m/s  source=%s", wind_ms, wind_source)

    # Load crops
    log.info("  Loading %d×%d px crops from npy + TIF...", CROP_PX, CROP_PX)
    try:
        b11_10m, b12_10m, prob_10m = load_site_crop(det)
    except Exception as e:
        return {"label": label, "status": "error", "error": str(e)}

    # Downsample to 20m
    b11_20m = b11_10m[::2, ::2]
    b12_20m = b12_10m[::2, ::2]

    # Canonical base flow
    mask_10m = (prob_10m >= det["prob_thresh"]).astype(np.uint8)
    mask_20m = mask_10m[::2, ::2]
    base_result = cemf_flow_rate(b11_20m, b12_20m, mask_20m, wind_ms)
    base_flow   = base_result["flow_kgh"]
    log.info("  Canonical CEMF: %.1f kg/h  (n_px=%d, thresh=%.4g)",
             base_flow, base_result["n_plume_pixels"], det["prob_thresh"])
    log.info("  Reported flow:  %.1f kg/h", det["reported_flow_kgh"])

    # ── Source 1: Wind ────────────────────────────────────────────────────────
    unc_wind = {
        "sigma_pct": SIGMA_WIND_PCT,
        "method":    "literature_prior",
        "reference": "Varon 2021 AMT 14:2771-2785 Table 2",
        "note":      "ERA5 10m → plume-layer transport uncertainty",
    }

    # ── Source 2: Sensitivity coefficient ────────────────────────────────────
    unc_coeff = {
        "sigma_pct": SIGMA_COEFF_PCT,
        "method":    "literature_prior",
        "reference": "Varon 2021 AMT 14:2771-2785 Table 2",
        "note":      "CEMF sensitivity = 4e-7 reflectance/(ppb·m) ± 15%",
    }

    # ── Source 3: Mask threshold bootstrap ───────────────────────────────────
    log.info("  [3/4] Mask threshold bootstrap (50 log-spaced thresholds)...")
    unc_mask = bootstrap_mask_threshold(
        b11_10m, b12_10m, prob_10m, wind_ms,
        canonical_thresh=det["prob_thresh"],
        n_steps=50
    )

    # ── Source 4: Background annulus jackknife ────────────────────────────────
    log.info("  [4/4] Background annulus jackknife (N/S/E/W)...")
    unc_bg = jackknife_background_annulus(b11_20m, b12_20m, mask_20m, wind_ms)

    # ── Combined MC ───────────────────────────────────────────────────────────
    log.info("  Monte Carlo combination (%d samples)...", n_mc)
    ref_flow = det["reported_flow_kgh"] if det["reported_flow_kgh"] else base_flow
    unc_mc = monte_carlo_combined(
        base_flow_kgh   = ref_flow,
        sigma_wind_pct  = unc_wind["sigma_pct"],
        sigma_coeff_pct = unc_coeff["sigma_pct"],
        sigma_mask_pct  = unc_mask["sigma_pct"],
        sigma_bg_pct    = unc_bg["sigma_pct"],
        n_samples       = n_mc,
    )

    return {
        "label":              label,
        "site":               det["site"],
        "scene_id":           det["scene_id"],
        "acquisition_ts":     det["acquisition_ts"],
        "wind_ms":            round(wind_ms, 4),
        "wind_source":        wind_source,
        "reported_flow_kgh":  det["reported_flow_kgh"],
        "cemf_base_flow_kgh": round(base_flow, 2),
        "n_plume_pixels_20m": base_result["n_plume_pixels"],
        "canonical_thresh":   det["prob_thresh"],
        "uncertainty": {
            "wind":        unc_wind,
            "coeff":       unc_coeff,
            "mask":        unc_mask,
            "background":  unc_bg,
            "combined_mc": unc_mc,
        },
        "status": "ok",
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="WS1: 4-source uncertainty decomposition for confirmed CH4 detections"
    )
    parser.add_argument("--n-mc", type=int, default=10_000,
                        help="Monte Carlo samples (default: 10000)")
    parser.add_argument("--update-quant", action="store_true",
                        help="Write uncertainty_decomposition fields to quantification.json")
    parser.add_argument("--site", default=None,
                        help="Only process this site (neurath or belchatow)")
    args = parser.parse_args()

    print()
    print("=" * 68)
    print("  CH4Net WS1 — Uncertainty Decomposition")
    print(f"  Detections: {len(DETECTIONS)}   MC samples: {args.n_mc:,}")
    print(f"  σ_wind={SIGMA_WIND_PCT}%   σ_coeff={SIGMA_COEFF_PCT}%  (both: Varon 2021)")
    print("=" * 68)

    targets = DETECTIONS
    if args.site:
        targets = [d for d in targets if d["site"] == args.site]

    all_results = []
    for det in targets:
        try:
            r = run_one_detection(det, args.n_mc)
            all_results.append(r)
            if r["status"] == "ok":
                print_table(r["label"], r["reported_flow_kgh"],
                            r["wind_ms"], r["wind_source"], r["uncertainty"])
        except Exception as e:
            log.exception("Unexpected error for %s: %s", det["label"], e)
            all_results.append({"label": det["label"], "status": "exception", "error": str(e)})

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_path = Path("results_analysis/uncertainty_decomposition.json")
    with open(out_path, "w") as fh:
        json.dump(all_results, fh, indent=2)
    log.info("Saved → %s", out_path)

    # ── Optional write-back ───────────────────────────────────────────────────
    if args.update_quant:
        quant_path = Path("results_analysis/quantification.json")
        quant = json.load(open(quant_path))
        by_scene = {r.get("scene_id"): r for r in quant if r.get("scene_id")}
        updated = 0
        for r in all_results:
            if r.get("status") != "ok":
                continue
            rec = by_scene.get(r["scene_id"])
            if rec is None:
                continue
            mc = r["uncertainty"]["combined_mc"]
            rec["uncertainty_decomposition"] = {
                "sigma_wind_pct":          r["uncertainty"]["wind"]["sigma_pct"],
                "sigma_coeff_pct":         r["uncertainty"]["coeff"]["sigma_pct"],
                "sigma_mask_pct":          r["uncertainty"]["mask"]["sigma_pct"],
                "sigma_background_pct":    r["uncertainty"]["background"]["sigma_pct"],
                "sigma_combined_mc_pct":   mc["sigma_pct_mc"],
                "sigma_combined_quad_pct": mc["sigma_pct_quadrature"],
                "ci_90_low_kgh":           mc["p5_kgh"],
                "ci_90_high_kgh":          mc["p95_kgh"],
                "n_mc_samples":            mc["n_samples"],
                "method":                  "bootstrap_jackknife_montecarlo",
                "reference":               "Varon 2021 AMT 14:2771-2785",
            }
            updated += 1
        json.dump(quant, open(quant_path, "w"), indent=2)
        log.info("quantification.json: wrote uncertainty_decomposition to %d records", updated)

    # ── Summary ───────────────────────────────────────────────────────────────
    ok  = [r for r in all_results if r.get("status") == "ok"]
    err = [r for r in all_results if r.get("status") != "ok"]
    print(f"\n{'─'*68}")
    print(f"  Complete: {len(ok)} detections  |  {len(err)} errors")
    if err:
        for e in err:
            print(f"  ✗ {e['label']}: {e.get('error', '?')}")
    print(f"  Output: {out_path}")


if __name__ == "__main__":
    main()
