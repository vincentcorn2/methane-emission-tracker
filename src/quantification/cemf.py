import numpy as np
from dataclasses import dataclass
from typing import Optional

# Constants
DRY_AIR_COLUMN = 2.1e25        # molecules/m^2
MOL_WEIGHT_CH4 = 0.016         # kg/mol
AVOGADRO = 6.022e23
PIXEL_AREA_20M = 400.0         # m^2 at native 20m SWIR resolution

@dataclass
class CEMFResult:
    dxch4_map: np.ndarray          # per-pixel enhancement (ppb*m)
    plume_mask: np.ndarray         # binary mask at 20m
    total_mass_kg: float           # integrated column mass over plume
    scene_id: str
    timestamp: str
    pixel_area_m2: float = PIXEL_AREA_20M
    n_plume_pixels: int = 0
    retrieval_valid: bool = True
    warning: Optional[str] = None

def run_cemf(
    b11: np.ndarray,
    b12: np.ndarray,
    mask: np.ndarray,
    scene_id: str,
    timestamp: str
) -> CEMFResult:
    """
    Translate Sentinel-2 B11/B12 TOA reflectance into per-pixel
    methane column enhancement (dXCH4) using a scene-derived
    background matched filter.

    Args:
        b11: Band 11 TOA reflectance at native 20m resolution
        b12: Band 12 TOA reflectance at native 20m resolution
        mask: CH4Net binary mask downsampled to 20m (1=plume, 0=background)
        scene_id: Sentinel-2 product ID string
        timestamp: Scene acquisition timestamp (ISO format)

    Returns:
        CEMFResult with per-pixel dXCH4 map and integrated mass
    """
    mask_bool = mask.astype(bool)
    background = ~mask_bool

    # Need enough background pixels for stable reference
    if background.sum() < 100:
        return CEMFResult(
            dxch4_map=np.zeros_like(b11),
            plume_mask=mask,
            total_mass_kg=0.0,
            scene_id=scene_id,
            timestamp=timestamp,
            n_plume_pixels=int(mask_bool.sum()),
            retrieval_valid=False,
            warning="Insufficient background pixels for stable retrieval"
        )

    # ── Varon et al. (2021) MBSP column retrieval ─────────────────────────────
    # Eq. (3): ΔR_MBSP = (c · R12 − R11) / R11
    # where c is the band-scaling factor fit by zero-intercept least squares on
    # background pixels only: minimise ||R11 − c·R12||²  →  c = Σ(R11·R12)/Σ(R12²)
    #
    # Methane absorbs in B12 but not B11, so over a plume pixel c·R12 < R11 and
    # ΔR_MBSP < 0.  We negate before dividing by the sensitivity so dXCH4 > 0.
    #
    # Sensitivity: ~4e-7 reflectance per ppb·m  (Varon 2021, calibrated for S2A/B
    # near background column Ω₀ ≈ 1875 ppb; within ±15% across typical scenes).
    # Using the full in-scene CH4Net mask as the background selector keeps this
    # single-pass (MBSP) — no temporal differencing, safe for continuous emitters.

    bg_b11 = b11[background].astype(np.float64)
    bg_b12 = b12[background].astype(np.float64)

    mu_b11 = float(bg_b11.mean())
    if mu_b11 < 1e-6:
        return CEMFResult(
            dxch4_map=np.zeros_like(b11),
            plume_mask=mask,
            total_mass_kg=0.0,
            scene_id=scene_id,
            timestamp=timestamp,
            retrieval_valid=False,
            warning="Near-zero B11 background — possible bad scene"
        )

    b12_sq_sum = float(np.dot(bg_b12, bg_b12))
    if b12_sq_sum < 1e-12:
        return CEMFResult(
            dxch4_map=np.zeros_like(b11),
            plume_mask=mask,
            total_mass_kg=0.0,
            scene_id=scene_id,
            timestamp=timestamp,
            retrieval_valid=False,
            warning="Near-zero B12 background — possible bad scene"
        )

    # Scene-derived band-scaling factor (replaces hardcoded 0.5)
    c = float(np.dot(bg_b11, bg_b12)) / b12_sq_sum

    # Fractional reflectance anomaly per Varon Eq. (3)
    delta_R = (c * b12.astype(np.float64) - b11.astype(np.float64)) / b11.astype(np.float64)

    # Convert to column enhancement: ΔR is negative over plume → negate for positive dXCH4
    SENSITIVITY = 4e-7   # reflectance per ppb·m (Varon 2021 Sec. 2.2)
    dxch4 = np.clip(-delta_R / SENSITIVITY, 0, None).astype(np.float32)

    # Integrate mass over plume pixels
    # dXCH4 (ppb*m) -> kg per pixel
    mass_per_pixel = (
        dxch4[mask_bool] * 1e-9
        * DRY_AIR_COLUMN
        * MOL_WEIGHT_CH4
        / AVOGADRO
        * PIXEL_AREA_20M
    )
    total_mass = float(mass_per_pixel.sum())

    return CEMFResult(
        dxch4_map=dxch4,
        plume_mask=mask,
        total_mass_kg=total_mass,
        scene_id=scene_id,
        timestamp=timestamp,
        n_plume_pixels=int(mask_bool.sum()),
        retrieval_valid=True
    )


def downsample_mask(mask_10m: np.ndarray) -> np.ndarray:
    """Downsample CH4Net 10m mask to 20m for SWIR band alignment."""
    return mask_10m[::2, ::2]
