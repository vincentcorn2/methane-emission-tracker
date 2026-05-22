"""
quantification_uncertainty.py
==============================
4-source uncertainty decomposition for confirmed CH4 detections.

For each detection in DETECTIONS, propagates four independent uncertainty
sources through the CEMF+IME flow-rate estimator:

  Source                        | 1σ (%)  | Method
  ─────────────────────────────────────────────────────────────────
  Wind speed (ERA5 → plume)     |  ±20%   | Literature prior (Varon 2021)
  Sensitivity coefficient        |  ±15%   | Literature prior (Varon 2021)
  Plume mask threshold           |  ±Z%    | Bootstrap over p* range
  Background annulus             |  ±W%    | Jackknife N/S/E/W quadrants
  Combined (Monte Carlo 10k)     |  ±tot%  | Independent sources assumed

Class hierarchy
---------------
BaseCemfQuantifier           Abstract base.  Provides the inline CEMF physics
                             (CemfFlowRate, CemfFlowWithBgMask,
                             MonteCarloCombined) as concrete methods, so all
                             subclasses share one canonical implementation.
                             Declares abstract _LoadData() for data I/O.

UncertaintyDecomposer        Full pipeline.  Implements _LoadData() using
(BaseCemfQuantifier)         memory-mapped npy + PIL/rasterio TIF crop loading.
                             Adds BootstrapMaskThreshold(), JackknifeBackground-
                             Annulus(), Run(), PrintSummary(), and SaveResults().

Usage
-----
    cd methane-api
    python scripts/quantification/quantification_uncertainty.py
    python scripts/quantification/quantification_uncertainty.py --update-quant
    python scripts/quantification/quantification_uncertainty.py --n-mc 20000
    python scripts/quantification/quantification_uncertainty.py --site neurath

Inputs  (per detection)
------
    data/npy_cache/<scene_id>.npy              12-band Sentinel-2 tile
    results_bitemporal/<site>/original_<scene_id>.tif   CH4Net probability TIF

Output
------
    results_analysis/uncertainty_decomposition.json

References
----------
Varon et al. (2021) "High-frequency monitoring of anomalous methane point
    sources with multispectral Sentinel-2 satellite observations." AMT 14,
    2771-2785.  Table 2: wind ±20%, sensitivity ±15%.
Lakshminarayanan et al. (2017) NeurIPS: epistemic/aleatoric uncertainty.
"""

import argparse
import json
import logging
import sys
import warnings
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Optional imports (PIL / rasterio for TIF loading)
# ---------------------------------------------------------------------------

try:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None          # suppress decompression-bomb warning
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

try:
    import rasterio
    _RASTERIO_AVAILABLE = True
except ImportError:
    _RASTERIO_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("quantification_uncertainty")

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------

ROOT        = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT / "results_analysis"
OUT_PATH    = RESULTS_DIR / "uncertainty_decomposition.json"

# Physical constants (mirrors src/quantification/cemf.py)
DRY_AIR_COLUMN    = 2.1e25      # molecules m⁻²
MOL_WEIGHT_CH4    = 0.016       # kg mol⁻¹
AVOGADRO          = 6.022e23
PIXEL_AREA_20M    = 400.0       # m² at 20 m native SWIR resolution
PIXEL_SIZE_20M    = 20.0        # m

SENSITIVITY_COEFF = 4e-7        # reflectance (ppb·m)⁻¹, Varon 2021 §2.2

# Band indices in 12-band .npy stack (B01 B02 … B8A B09 B11 B12)
IDX_B11 = 10
IDX_B12 = 11

CROP_PX = 500                   # 10 m pixels → 5 km × 5 km crop around facility

# Literature uncertainty priors (Varon 2021 AMT Table 2)
SIGMA_WIND_PCT  = 20.0          # ERA5 10 m → plume-layer transport
SIGMA_COEFF_PCT = 15.0          # CEMF sensitivity coefficient

WIND_FALLBACK_MS = 3.5          # climatological fallback m s⁻¹

# Mask bootstrap parameters
BOOTSTRAP_THRESHOLD_STEPS = 50
BOOTSTRAP_LO_FACTOR       = 0.3
BOOTSTRAP_HI_FACTOR       = 10.0
BOOTSTRAP_HI_CAP          = 0.50

# Monte Carlo default
N_MC_DEFAULT = 10_000
MC_SEED      = 42

# ---------------------------------------------------------------------------
# Detection registry
# ---------------------------------------------------------------------------

DETECTIONS: list[dict] = [
    {
        "site":              "neurath",
        "label":             "Neurath 2024-08-29",
        "scene_id":          "S2B_MSIL1C_20240829T103629_N0511_R008_T32ULB_20240829T124434",
        "acquisition_ts":    "2024-08-29T10:36:29Z",
        "lat": 51.038, "lon": 6.616,
        "site_row": 4324, "site_col": 3286,
        "prob_thresh":       0.0007,                # 5× ctrl_mu = 0.000149
        "tif": "results_bitemporal/neurath/"
               "original_S2B_MSIL1C_20240829T103629_N0511_R008_T32ULB_20240829T124434.tif",
        "npy": "data/npy_cache/"
               "S2B_MSIL1C_20240829T103629_N0511_R008_T32ULB_20240829T124434.npy",
        "wind_ms":           2.6612,
        "wind_source":       "ERA5_reanalysis",
        "reported_flow_kgh": 85.0,
    },
    {
        "site":              "belchatow",
        "label":             "Bełchatów 2024-07-10",
        "scene_id":          "S2A_MSIL1C_20240710T095031_N0510_R079_T34UCB_20240710T133148",
        "acquisition_ts":    "2024-07-10T09:50:31Z",
        "lat": 51.242, "lon": 19.275,   # mine centroid (ClimateTrace asset 16168)
        "site_row": 1949, "site_col": 8356,
        "prob_thresh":       0.002,                 # 5× ctrl_mu = 0.000499
        "tif": "results_bitemporal/belchatow/"
               "original_S2A_MSIL1C_20240710T095031_N0510_R079_T34UCB_20240710T133148.tif",
        "npy": "data/npy_cache/"
               "S2A_MSIL1C_20240710T095031_N0510_R079_T34UCB_20240710T133148.npy",
        "wind_ms":           3.3183,
        "wind_source":       "ERA5_reanalysis",
        "reported_flow_kgh": 1071.0,
    },
    {
        "site":              "neurath",
        "label":             "Neurath 2024-06-25",
        "scene_id":          "S2A_MSIL1C_20240625T103631_N0510_R008_T32ULB_20240625T142035",
        "acquisition_ts":    "2024-06-25T10:36:31Z",
        "lat": 51.038, "lon": 6.616,
        "site_row": 4324, "site_col": 3286,
        "prob_thresh":       0.016,                 # 3× ctrl_mean = 0.00526
        "tif": "results_bitemporal/neurath/"
               "original_S2A_MSIL1C_20240625T103631_N0510_R008_T32ULB_20240625T142035.tif",
        "npy": "data/npy_cache/"
               "S2A_MSIL1C_20240625T103631_N0510_R008_T32ULB_20240625T142035.npy",
        "wind_ms":           None,                  # fetch ERA5 at runtime
        "wind_source":       None,
        "reported_flow_kgh": 338.0,
    },
    {
        "site":              "belchatow",
        "label":             "Bełchatów 2024-08-24",
        "scene_id":          "S2B_MSIL1C_20240824T094549_N0511_R079_T34UCB_20240824T115611",
        "acquisition_ts":    "2024-08-24T09:45:49Z",
        "lat": 51.242, "lon": 19.275,   # mine centroid (ClimateTrace asset 16168)
        "site_row": 1949, "site_col": 8356,
        "prob_thresh":       0.0002,                # 5× ctrl_mean = 3.9e-05
        "tif": "results_bitemporal/belchatow/"
               "original_S2B_MSIL1C_20240824T094549_N0511_R079_T34UCB_20240824T115611.tif",
        "npy": "data/npy_cache/"
               "S2B_MSIL1C_20240824T094549_N0511_R079_T34UCB_20240824T115611.npy",
        "wind_ms":           2.19,
        "wind_source":       "ERA5_reanalysis",
        "reported_flow_kgh": 426.0,
    },
    {
        "site":              "turkmenistan",
        "label":             "Turkmenistan 2021-01-01 (L2A, pipeline validation)",
        "scene_id":          "S2B_T40SBE_20210101_Turkmenistan",
        "acquisition_ts":    "2021-01-01T07:43:00Z",
        "lat": 35.173, "lon": 54.754,
        "site_row": 5240, "site_col": 4774,
        "prob_thresh":       0.001,
        "tif": "results_bitemporal/turkmenistan/"
               "original_S2B_T40SBE_20210101_Turkmenistan.tif",
        "npy": "data/npy_cache/S2B_T40SBE_20210101_Turkmenistan.npy",
        "wind_ms":           1.58,
        "wind_source":       "ERA5_reanalysis",
        "reported_flow_kgh": 225.0,
    },
]

# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseCemfQuantifier(ABC):
    """
    Abstract base class for CEMF-based CH4 flow-rate quantifiers.

    Provides the inline CEMF physics as concrete methods so all subclasses
    share a single canonical implementation.  Subclasses must implement
    _LoadData() to supply the (b11_10m, b12_10m, prob_10m) arrays for a
    given detection record.

    CEMF formulation (Varon et al. 2021, §2.2)
    -------------------------------------------
    The cross-sectional flux method estimates the column-integrated mass
    flux across the plume cross-section:

        ΔXCH4 = clip( (ΔR_B12 - 0.5·ΔR_B11) / (μ_B11 · ε), 0 )
        mass   = Σ_plume  ΔXCH4 · 1e-9 · N_dry · M_CH4 / N_A · A_px
        Q      = mass / (L_plume / wind_speed)  [kg s⁻¹] × 3600  [kg h⁻¹]

    where ε = SENSITIVITY_COEFF = 4e-7 reflectance (ppb·m)⁻¹.
    """

    @abstractmethod
    def _LoadData(self, det: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load B11, B12, and CH4Net probability arrays for a detection.

        Parameters
        ----------
        det : dict
            One entry from DETECTIONS.

        Return
        ------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            (b11_10m, b12_10m, prob_10m) — float32 [0, 1], shape (CROP_PX, CROP_PX).

        Raises
        ------
        FileNotFoundError if the npy or TIF path does not exist.
        """

    # ---------------------------------------------------------------- physics

    @staticmethod
    def CemfFlowRate(
        b11: np.ndarray,
        b12: np.ndarray,
        mask_20m: np.ndarray,
        wind_ms: float,
        sensitivity: float = SENSITIVITY_COEFF,
    ) -> dict:
        """
        Compute CEMF-derived CH4 flow rate for a given plume mask.

        Parameters
        ----------
        b11, b12 : np.ndarray
            Float32 reflectance arrays [0, 1] at 20 m resolution.
        mask_20m : np.ndarray
            Binary (0/1) plume mask at 20 m.
        wind_ms : float
            Wind speed in m s⁻¹.
        sensitivity : float, optional
            CEMF sensitivity coefficient ε (default: 4e-7).

        Return
        ------
        dict  Keys: flow_kgh, total_mass_kg, plume_length_m,
              n_plume_pixels, valid (bool), warning (str or None).
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
        dxch4 = np.clip(
            (d_b12 - 0.5 * d_b11) / (mu_b11 * sensitivity), 0, None
        )

        mass_per_pixel = (
            dxch4[mask_bool] * 1e-9
            * DRY_AIR_COLUMN * MOL_WEIGHT_CH4 / AVOGADRO
            * PIXEL_AREA_20M
        )
        total_mass = float(mass_per_pixel.sum())

        rows, cols     = np.where(mask_bool)
        plume_length_m = max(
            float(max(rows.max() - rows.min(), cols.max() - cols.min())) * PIXEL_SIZE_20M,
            PIXEL_SIZE_20M,
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

    @staticmethod
    def CemfFlowWithBgMask(
        b11: np.ndarray,
        b12: np.ndarray,
        mask_20m: np.ndarray,
        bg_bool: np.ndarray,
        wind_ms: float,
        sensitivity: float = SENSITIVITY_COEFF,
    ) -> dict:
        """
        CEMF flow rate with an explicit background boolean mask.
        Used by the jackknife to supply custom background regions.

        Parameters
        ----------
        b11, b12  : np.ndarray  Float32 reflectance [0, 1] at 20 m.
        mask_20m  : np.ndarray  Binary plume mask at 20 m.
        bg_bool   : np.ndarray  Explicit background boolean mask.
        wind_ms   : float       Wind speed m s⁻¹.
        sensitivity : float     CEMF coefficient ε.

        Return
        ------
        dict  Keys: flow_kgh, total_mass_kg, valid (bool), warning.
        """
        if bg_bool.sum() < 50:
            return {"flow_kgh": 0.0, "valid": False, "warning": "too_few_bg_pixels"}

        mu_b11 = float(b11[bg_bool].mean())
        mu_b12 = float(b12[bg_bool].mean())
        if mu_b11 < 1e-6:
            return {"flow_kgh": 0.0, "valid": False, "warning": "near_zero_b11"}

        d_b11 = b11 - mu_b11
        d_b12 = b12 - mu_b12
        dxch4 = np.clip(
            (d_b12 - 0.5 * d_b11) / (mu_b11 * sensitivity), 0, None
        )
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
            PIXEL_SIZE_20M,
        )
        flow_kgh = round(
            total_mass / max(plume_length_m / max(wind_ms, 0.5), 1e-6) * 3600.0, 4
        )
        return {
            "flow_kgh":      flow_kgh,
            "total_mass_kg": total_mass,
            "valid":         True,
            "warning":       None,
        }

    @staticmethod
    def MonteCarloCombined(
        base_flow_kgh: float,
        sigma_wind_pct: float,
        sigma_coeff_pct: float,
        sigma_mask_pct: float,
        sigma_bg_pct: float,
        n_samples: int = N_MC_DEFAULT,
        seed: int = MC_SEED,
    ) -> dict:
        """
        10 k-sample Monte Carlo propagation of four independent uncertainty
        sources through Q = Q₀ × f_wind × f_coeff × f_mask × f_bg,
        where each fᵢ ~ N(1, σᵢ/100).  NaN inputs contribute zero variance.

        Parameters
        ----------
        base_flow_kgh   : float  Canonical flow rate (kg h⁻¹).
        sigma_wind_pct  : float  1σ as % for wind uncertainty.
        sigma_coeff_pct : float  1σ as % for sensitivity coefficient.
        sigma_mask_pct  : float  1σ as % for mask threshold (NaN → 0).
        sigma_bg_pct    : float  1σ as % for background annulus (NaN → 0).
        n_samples       : int    MC sample count (default: 10 000).
        seed            : int    RNG seed.

        Return
        ------
        dict  n_samples, mean_kgh, std_kgh, p5_kgh, p95_kgh,
              sigma_pct_mc, sigma_pct_quadrature.
        """
        rng = np.random.default_rng(seed)

        def _safe(v: float) -> float:
            return 0.0 if (v is None or np.isnan(v)) else v / 100.0

        s_w = _safe(sigma_wind_pct)
        s_c = _safe(sigma_coeff_pct)
        s_m = _safe(sigma_mask_pct)
        s_b = _safe(sigma_bg_pct)

        f_w = np.clip(rng.normal(1.0, s_w, n_samples), 0.01, None)
        f_c = np.clip(rng.normal(1.0, s_c, n_samples), 0.01, None)
        f_m = np.clip(rng.normal(1.0, s_m, n_samples), 0.01, None)
        f_b = np.clip(rng.normal(1.0, s_b, n_samples), 0.01, None)

        samples   = base_flow_kgh * f_w * f_c * f_m * f_b
        mean_s    = float(np.mean(samples))
        std_s     = float(np.std(samples, ddof=1))
        p5, p95   = float(np.percentile(samples, 5)), float(np.percentile(samples, 95))
        sigma_pct = round(100.0 * std_s / mean_s, 1) if mean_s > 0 else float("nan")
        quad_pct  = round(100.0 * np.sqrt(s_w**2 + s_c**2 + s_m**2 + s_b**2), 1)

        return {
            "n_samples":            n_samples,
            "mean_kgh":             round(mean_s, 2),
            "std_kgh":              round(std_s, 2),
            "p5_kgh":               round(p5, 2),
            "p95_kgh":              round(p95, 2),
            "sigma_pct_mc":         sigma_pct,
            "sigma_pct_quadrature": quad_pct,
        }


# ---------------------------------------------------------------------------
# UncertaintyDecomposer
# ---------------------------------------------------------------------------


class UncertaintyDecomposer(BaseCemfQuantifier):
    """
    Full 4-source uncertainty decomposition pipeline.

    Implements _LoadData() with memory-mapped npy + PIL/rasterio TIF cropping.
    Adds the two data-driven uncertainty sources:

    - BootstrapMaskThreshold(): sweeps p* over [0.3×p*, min(10×p*, 0.50)]
      (50 log-spaced steps) and reports σ/mean as % uncertainty.
    - JackknifeBackgroundAnnulus(): drops each of the N/S/E/W quadrants and
      recomputes CEMF; σ over the 4 runs = background uncertainty.

    Inputs (per detection, from DETECTIONS registry)
    -------------------------------------------------
    data/npy_cache/<scene_id>.npy
    results_bitemporal/<site>/original_<scene_id>.tif

    Output
    ------
    results_analysis/uncertainty_decomposition.json
    """

    def __init__(
        self,
        detections: list[dict] = DETECTIONS,
        n_mc: int = N_MC_DEFAULT,
        out_path: Path = OUT_PATH,
    ) -> None:
        """
        Parameters
        ----------
        detections : list[dict], optional
            Detection registry (default: module-level DETECTIONS).
        n_mc : int, optional
            Monte Carlo sample count (default: 10 000).
        out_path : Path, optional
            Output JSON path.
        """
        self._detections = detections
        self._n_mc       = n_mc
        self._out_path   = out_path

    # --------------------------------------------------------------- data I/O

    def _LoadData(
        self, det: dict
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Memory-map the npy tile and PIL/rasterio-crop the TIF to a
        CROP_PX × CROP_PX window centred on (site_row, site_col).

        Parameters
        ----------
        det : dict  One entry from DETECTIONS.

        Return
        ------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            (b11_10m, b12_10m, prob_10m) — float32 [0, 1], (CROP_PX, CROP_PX).

        Raises
        ------
        FileNotFoundError  if npy or TIF path is missing.
        ImportError        if neither PIL nor rasterio is available.
        """
        npy_path = ROOT / det["npy"]
        tif_path = ROOT / det["tif"]

        for p in (npy_path, tif_path):
            if not p.exists():
                raise FileNotFoundError(f"Required data file not found: {p}")

        row, col = det["site_row"], det["site_col"]
        half     = CROP_PX // 2
        r0, r1   = row - half, row + half
        c0, c1   = col - half, col + half

        # Memory-mapped npy — reads only the crop slice
        arr     = np.load(npy_path, mmap_mode="r")
        b11_10m = arr[r0:r1, c0:c1, IDX_B11].astype(np.float32) / 255.0
        b12_10m = arr[r0:r1, c0:c1, IDX_B12].astype(np.float32) / 255.0
        log.info("  B11 crop %s  range=[%.4f, %.4f]",
                 b11_10m.shape, b11_10m.min(), b11_10m.max())
        log.info("  B12 crop %s  range=[%.4f, %.4f]",
                 b12_10m.shape, b12_10m.min(), b12_10m.max())

        # TIF probability map
        if _PIL_AVAILABLE:
            img      = Image.open(tif_path)
            prob_10m = np.array(img.crop((c0, r0, c1, r1))).astype(np.float32)
        elif _RASTERIO_AVAILABLE:
            with rasterio.open(tif_path) as src:
                window   = rasterio.windows.Window(c0, r0, CROP_PX, CROP_PX)
                prob_10m = src.read(1, window=window).astype(np.float32)
        else:
            raise ImportError(
                "Install Pillow (pip install Pillow) or rasterio to read TIF files."
            )

        if prob_10m.max() > 1.0:
            prob_10m = prob_10m / prob_10m.max()
        log.info("  Prob crop %s  range=[%.6f, %.6f]",
                 prob_10m.shape, prob_10m.min(), prob_10m.max())
        return b11_10m, b12_10m, prob_10m

    # ------------------------------------------------------- Source 3: bootstrap

    def BootstrapMaskThreshold(
        self,
        b11: np.ndarray,
        b12: np.ndarray,
        prob_10m: np.ndarray,
        wind_ms: float,
        canonical_thresh: float,
        n_steps: int = BOOTSTRAP_THRESHOLD_STEPS,
    ) -> dict:
        """
        Bootstrap flow rate over log-spaced probability thresholds.

        Sweeps p* from BOOTSTRAP_LO_FACTOR × p_canon to
        min(BOOTSTRAP_HI_FACTOR × p_canon, BOOTSTRAP_HI_CAP),
        computing CEMF at each step.  Reports σ/mean as % uncertainty.

        Parameters
        ----------
        b11, b12         : np.ndarray  Float32 reflectance [0, 1] at 10 m.
        prob_10m         : np.ndarray  CH4Net probability map at 10 m.
        wind_ms          : float       Wind speed m s⁻¹.
        canonical_thresh : float       Canonical p* for this detection.
        n_steps          : int         Bootstrap steps (default: 50).

        Return
        ------
        dict  sigma_pct, n_valid, threshold_range, mean_flow_kgh,
              std_flow_kgh, flows_summary.
        """
        lo = BOOTSTRAP_LO_FACTOR * canonical_thresh
        hi = min(BOOTSTRAP_HI_FACTOR * canonical_thresh, BOOTSTRAP_HI_CAP)
        thresholds = np.geomspace(lo, hi, n_steps)

        flows = []
        for p_star in thresholds:
            mask_10m = (prob_10m >= p_star).astype(np.uint8)
            mask_20m = mask_10m[::2, ::2]
            if mask_20m.sum() < 4:
                continue
            result = self.CemfFlowRate(b11[::2, ::2], b12[::2, ::2], mask_20m, wind_ms)
            if result["valid"] and result["flow_kgh"] > 0:
                flows.append({
                    "p_star":   float(p_star),
                    "flow_kgh": result["flow_kgh"],
                    "n_px":     result["n_plume_pixels"],
                })

        if len(flows) < 4:
            log.warning("  Mask bootstrap: only %d valid steps — σ unreliable", len(flows))
            return {
                "sigma_pct":  float("nan"),
                "n_valid":    len(flows),
                "note":       "insufficient_range",
                "flows_summary": flows,
            }

        flow_vals = [x["flow_kgh"] for x in flows]
        mean_flow = float(np.mean(flow_vals))
        std_flow  = float(np.std(flow_vals, ddof=1))
        sigma_pct = round(100.0 * std_flow / mean_flow, 1) if mean_flow > 0 else float("nan")
        log.info("  Mask bootstrap: n=%d  mean=%.1f  std=%.1f  σ=%.1f%%",
                 len(flows), mean_flow, std_flow, sigma_pct)

        key_idxs     = {0, len(flows)//4, len(flows)//2, 3*len(flows)//4, len(flows)-1}
        flows_summary = [
            {k: round(v, 5) if isinstance(v, float) else v for k, v in flows[i].items()}
            for i in sorted(key_idxs)
        ]
        return {
            "sigma_pct":        sigma_pct,
            "n_valid":          len(flows),
            "threshold_range":  [round(float(lo), 6), round(float(hi), 6)],
            "canonical_thresh": canonical_thresh,
            "mean_flow_kgh":    round(mean_flow, 2),
            "std_flow_kgh":     round(std_flow, 2),
            "flows_summary":    flows_summary,
        }

    # ------------------------------------------------- Source 4: BG jackknife

    def JackknifeBackgroundAnnulus(
        self,
        b11_20m: np.ndarray,
        b12_20m: np.ndarray,
        mask_20m: np.ndarray,
        wind_ms: float,
    ) -> dict:
        """
        Jackknife the background annulus by dropping each N/S/E/W quadrant.

        For each of the four quadrant-drop runs, CEMF is computed using only
        the remaining three quadrants as background.  σ over the 4 valid
        runs is the background uncertainty.

        Parameters
        ----------
        b11_20m, b12_20m : np.ndarray  Float32 reflectance [0, 1] at 20 m.
        mask_20m         : np.ndarray  Binary plume mask at 20 m.
        wind_ms          : float       Wind speed m s⁻¹.

        Return
        ------
        dict  sigma_pct, n_valid, mean_flow_kgh, std_flow_kgh,
              per_quadrant_kgh.
        """
        plume_bool = mask_20m.astype(bool)
        bg_bool    = ~plume_bool

        if plume_bool.sum() < 4:
            return {"sigma_pct": float("nan"), "note": "empty_plume"}

        rows_p, cols_p = np.where(plume_bool)
        c_row = float(rows_p.mean())
        c_col = float(cols_p.mean())

        H, W = mask_20m.shape
        ri = np.arange(H, dtype=float).reshape(-1, 1) * np.ones((1, W))
        ci = np.ones((H, 1), dtype=float) * np.arange(W, dtype=float).reshape(1, -1)

        quadrants = {
            "N": bg_bool & (ri < c_row),
            "S": bg_bool & (ri >= c_row),
            "W": bg_bool & (ci < c_col),
            "E": bg_bool & (ci >= c_col),
        }

        flows: dict[str, float] = {}
        for drop_q in list(quadrants.keys()):
            bg_kept = np.zeros((H, W), dtype=bool)
            for q, qm in quadrants.items():
                if q != drop_q:
                    bg_kept |= qm.astype(bool)
            result = self.CemfFlowWithBgMask(b11_20m, b12_20m, mask_20m, bg_kept, wind_ms)
            if result["valid"] and result["flow_kgh"] > 0:
                flows[drop_q] = result["flow_kgh"]
                log.info("  BG jackknife (drop %s): %.1f kg/h", drop_q, result["flow_kgh"])
            else:
                log.warning("  BG jackknife (drop %s): invalid (%s)",
                            drop_q, result.get("warning", "?"))

        if len(flows) < 2:
            return {
                "sigma_pct": float("nan"),
                "n_valid":   len(flows),
                "flows":     flows,
                "note":      "too_few_valid_jackknife_runs",
            }

        flow_vals = list(flows.values())
        mean_flow = float(np.mean(flow_vals))
        std_flow  = float(np.std(flow_vals, ddof=1))
        sigma_pct = round(100.0 * std_flow / mean_flow, 1) if mean_flow > 0 else float("nan")
        log.info("  BG jackknife: n=%d  mean=%.1f  std=%.1f  σ=%.1f%%",
                 len(flow_vals), mean_flow, std_flow, sigma_pct)
        return {
            "sigma_pct":        sigma_pct,
            "n_valid":          len(flows),
            "mean_flow_kgh":    round(mean_flow, 2),
            "std_flow_kgh":     round(std_flow, 2),
            "per_quadrant_kgh": {q: round(v, 2) for q, v in flows.items()},
        }

    # --------------------------------------------------------- wind helper

    def _FetchEra5Wind(
        self, lat: float, lon: float, ts: str
    ) -> tuple[float, str]:
        """
        Retrieve ERA5 10 m wind via ERA5Client, falling back to the
        climatological default on any failure.

        Parameters
        ----------
        lat, lon : float  Site coordinates.
        ts       : str    ISO 8601 acquisition timestamp.

        Return
        ------
        tuple[float, str]  (wind_ms, wind_source).
        """
        try:
            sys.path.insert(0, str(ROOT))
            from src.ingestion.era5_client import ERA5Client, FALLBACK_WIND_SPEED
            result   = ERA5Client().get_wind(lat, lon, ts)
            wind_ms  = float(result["wind_speed_ms"])
            source   = result.get("wind_source", "ERA5_reanalysis")
            if wind_ms == FALLBACK_WIND_SPEED and "climatological" in source:
                log.warning("  ERA5 returned fallback %.1f m/s", wind_ms)
            else:
                log.info("  ERA5 wind: %.3f m/s", wind_ms)
            return wind_ms, source
        except Exception as exc:
            log.warning("  ERA5 unavailable (%s) → %.1f m/s fallback",
                        type(exc).__name__, WIND_FALLBACK_MS)
            return WIND_FALLBACK_MS, f"climatological_fallback_{WIND_FALLBACK_MS}ms"

    # --------------------------------------------------------- one detection

    def _RunOneDetection(self, det: dict) -> dict:
        """
        Execute the full 4-source decomposition for a single detection.

        Parameters
        ----------
        det : dict  One entry from DETECTIONS.

        Return
        ------
        dict  Full per-detection result, with status='ok' or status='error'.
        """
        label = det["label"]
        log.info("=" * 60)
        log.info("Processing: %s", label)

        # -- Resolve wind
        wind_ms     = det.get("wind_ms")
        wind_source = det.get("wind_source") or "unknown"
        if wind_ms is None:
            log.info("  Fetching ERA5 wind for %s %s...",
                     det["site"], det["acquisition_ts"][:10])
            wind_ms, wind_source = self._FetchEra5Wind(
                det["lat"], det["lon"], det["acquisition_ts"]
            )

        log.info("  Wind: %.4f m/s  source=%s", wind_ms, wind_source)

        # -- Load crops
        try:
            b11_10m, b12_10m, prob_10m = self._LoadData(det)
        except (FileNotFoundError, ImportError) as exc:
            log.warning("  Skipping %s: %s", label, exc)
            return {"label": label, "status": "error", "error": str(exc)}
        except Exception as exc:
            log.exception("  Unexpected load error for %s", label)
            return {"label": label, "status": "error", "error": str(exc)}

        b11_20m = b11_10m[::2, ::2]
        b12_20m = b12_10m[::2, ::2]

        # -- Canonical base flow
        mask_10m    = (prob_10m >= det["prob_thresh"]).astype(np.uint8)
        mask_20m    = mask_10m[::2, ::2]
        base_result = self.CemfFlowRate(b11_20m, b12_20m, mask_20m, wind_ms)
        base_flow   = base_result["flow_kgh"]
        log.info("  Canonical CEMF: %.1f kg/h  (n_px=%d, thresh=%.4g)",
                 base_flow, base_result["n_plume_pixels"], det["prob_thresh"])
        log.info("  Reported flow:  %.1f kg/h", det["reported_flow_kgh"])

        # -- Source 1: Wind (literature prior)
        unc_wind = {
            "sigma_pct": SIGMA_WIND_PCT,
            "method":    "literature_prior",
            "reference": "Varon 2021 AMT 14:2771-2785 Table 2",
            "note":      "ERA5 10 m → plume-layer transport uncertainty",
        }

        # -- Source 2: Sensitivity coefficient (literature prior)
        unc_coeff = {
            "sigma_pct": SIGMA_COEFF_PCT,
            "method":    "literature_prior",
            "reference": "Varon 2021 AMT 14:2771-2785 Table 2",
            "note":      "CEMF sensitivity ε = 4e-7 reflectance (ppb·m)⁻¹ ± 15%",
        }

        # -- Source 3: Mask threshold bootstrap
        log.info("  [3/4] Mask threshold bootstrap (%d log-spaced thresholds)...",
                 BOOTSTRAP_THRESHOLD_STEPS)
        unc_mask = self.BootstrapMaskThreshold(
            b11_10m, b12_10m, prob_10m, wind_ms,
            canonical_thresh=det["prob_thresh"],
        )

        # -- Source 4: Background annulus jackknife
        log.info("  [4/4] Background annulus jackknife (N/S/E/W)...")
        unc_bg = self.JackknifeBackgroundAnnulus(b11_20m, b12_20m, mask_20m, wind_ms)

        # -- Combined Monte Carlo
        log.info("  Monte Carlo combination (%d samples)...", self._n_mc)
        ref_flow = det["reported_flow_kgh"] or base_flow
        unc_mc   = self.MonteCarloCombined(
            base_flow_kgh   = ref_flow,
            sigma_wind_pct  = unc_wind["sigma_pct"],
            sigma_coeff_pct = unc_coeff["sigma_pct"],
            sigma_mask_pct  = unc_mask["sigma_pct"],
            sigma_bg_pct    = unc_bg["sigma_pct"],
            n_samples       = self._n_mc,
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

    # --------------------------------------------------------------- Run

    def Run(self, site_filter: str | None = None) -> list[dict]:
        """
        Process all detections (or a single site) in DETECTIONS.

        Parameters
        ----------
        site_filter : str, optional
            If given, only process detections whose ``site`` matches this
            string (e.g. ``"neurath"`` or ``"belchatow"``).

        Return
        ------
        list[dict]  Per-detection results (status='ok' or 'error').
        """
        targets = self._detections
        if site_filter:
            targets = [d for d in targets if d["site"] == site_filter]

        results = []
        for det in targets:
            try:
                results.append(self._RunOneDetection(det))
            except Exception as exc:
                log.exception("Unexpected error for %s", det["label"])
                results.append({
                    "label":  det["label"],
                    "status": "exception",
                    "error":  str(exc),
                })
        return results

    def PrintSummary(self, results: list[dict]) -> None:
        """
        Print a formatted uncertainty budget table for each successful detection.

        Parameters
        ----------
        results : list[dict]  Payload from Run().
        """
        sep = "=" * 68
        for r in results:
            if r.get("status") != "ok":
                print(f"\n  ✗ {r['label']}: {r.get('error', '?')}")
                continue
            unc = r["uncertainty"]
            mc  = unc["combined_mc"]
            print(f"\n{sep}")
            print(f"  {r['label']}")
            print(f"  Reported Q̂: {r['reported_flow_kgh']:.1f} kg/h   "
                  f"ERA5: {r['wind_ms']:.3f} m/s ({r['wind_source']})")
            print(sep)
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
                 f"Bootstrap n={unc['mask'].get('n_valid', '?')} "
                 f"range={unc['mask'].get('threshold_range', ['?', '?'])}"),
                ("Background annulus",
                 unc["background"]["sigma_pct"],
                 f"Jackknife N/S/E/W n={unc['background'].get('n_valid', '?')}"),
            ]
            for src, sigma, method in rows:
                if sigma is None or (isinstance(sigma, float) and np.isnan(sigma)):
                    sigma_str = "   n/a"
                else:
                    sigma_str = f"{sigma:>6.1f}%"
                print(f"  {src:<33}  {sigma_str}  {method}")
            print("  " + "-" * 63)
            print(f"  {'Combined (MC, 10k)':<33}  {mc['sigma_pct_mc']:>6.1f}%  "
                  "independent sources")
            print(f"  {'  quadrature √Σσ²':<33}  {mc['sigma_pct_quadrature']:>6.1f}%")
            print(f"  {'  90% CI (5th–95th pct)':<33}  "
                  f"[{mc['p5_kgh']:.1f}, {mc['p95_kgh']:.1f}] kg/h")
            lo30 = round(r["reported_flow_kgh"] * 0.70, 1)
            hi30 = round(r["reported_flow_kgh"] * 1.30, 1)
            diff = mc["sigma_pct_mc"] - 30.0
            flag = (
                "⚠  WIDER than canonical"
                if diff > 5
                else ("✓  consistent with ±30%" if abs(diff) <= 5 else "✓  narrower than ±30%")
            )
            print(f"\n  Canonical ±30%: [{lo30}, {hi30}] kg/h")
            print(f"  MC 90% CI:      [{mc['p5_kgh']:.1f}, {mc['p95_kgh']:.1f}] kg/h")
            print(f"  Combined σ vs ±30%: {flag} ({mc['sigma_pct_mc']:.1f}% vs 30%)")
            print(sep)

        ok_n  = sum(1 for r in results if r.get("status") == "ok")
        err_n = len(results) - ok_n
        print(f"\n  Complete: {ok_n} detections  |  {err_n} errors")

    def SaveResults(
        self,
        results: list[dict],
        path: Path | None = None,
        update_quant: bool = False,
    ) -> None:
        """
        Write results to JSON and optionally merge back into quantification.json.

        Parameters
        ----------
        results      : list[dict]  Payload from Run().
        path         : Path, optional  Override output path.
        update_quant : bool, optional  If True, write uncertainty fields
                       back to results_analysis/quantification.json.
        """
        out = path or self._out_path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        log.info("Saved → %s", out)

        if not update_quant:
            return

        quant_path = RESULTS_DIR / "quantification.json"
        if not quant_path.exists():
            log.warning("quantification.json not found; skipping write-back.")
            return

        quant    = json.loads(quant_path.read_text())
        by_scene = {r.get("scene_id"): r for r in quant if r.get("scene_id")}
        updated  = 0
        for r in results:
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

        quant_path.write_text(json.dumps(quant, indent=2))
        log.info("quantification.json: uncertainty_decomposition written to %d records",
                 updated)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Command-line interface for the uncertainty decomposition pipeline.

    Usage
    -----
        cd methane-api
        python scripts/quantification/quantification_uncertainty.py
        python scripts/quantification/quantification_uncertainty.py --update-quant
        python scripts/quantification/quantification_uncertainty.py --n-mc 20000
        python scripts/quantification/quantification_uncertainty.py --site neurath
    """
    parser = argparse.ArgumentParser(
        description="CH4Net WS1 — 4-source uncertainty decomposition"
    )
    parser.add_argument(
        "--n-mc", type=int, default=N_MC_DEFAULT,
        help=f"Monte Carlo sample count (default: {N_MC_DEFAULT})",
    )
    parser.add_argument(
        "--update-quant", action="store_true",
        help="Write uncertainty fields back to quantification.json",
    )
    parser.add_argument(
        "--site", default=None,
        help="Only process this site (e.g. neurath, belchatow)",
    )
    args = parser.parse_args()

    decomposer = UncertaintyDecomposer(n_mc=args.n_mc)
    results    = decomposer.Run(site_filter=args.site)
    decomposer.PrintSummary(results)
    decomposer.SaveResults(results, update_quant=args.update_quant)


if __name__ == "__main__":
    main()
