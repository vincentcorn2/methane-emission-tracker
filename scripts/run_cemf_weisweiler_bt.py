"""
scripts/run_cemf_weisweiler_bt.py
==================================
Bi-temporal CEMF quantification for Weisweiler 2024-09-18.

WHY THE FIRST ATTEMPT FAILED
------------------------------
The original --dry-run showed 16,029 plume pixels (25.6% of the 500px crop)
at threshold=0.026 and dXCH4 mean = 204k ppb·m — both physically insane for
a ~775 MW plant.  Root cause: the plume mask came from the single-date CH4Net
probability TIF, which is contaminated by the same terrain scatter that
inflates the CEMF signal.  Subtracting the winter reference corrects the
BAND VALUES but not the MASK — the mask still tags terrain pixels as plume.

TWO PATHS TO A CLEAN MASK
--------------------------
1. CH4Net BT inference (recommended):
      python apply_bitemporal_diff.py --sites weisweiler --force-bt
   This runs CH4Net on the BT-differenced channels and produces
   results_bitemporal/weisweiler/bitemporal_*.tif — a probability map
   where stable terrain appears near 0 and methane shows elevated.
   Pass the BT TIF to this script via --bt-tif:
      python scripts/run_cemf_weisweiler_bt.py --bt-tif <path>

2. Spectral BT mask (no neural network, lower confidence):
      python scripts/run_cemf_weisweiler_bt.py --use-spectral-mask
   Pixels are "plume" only when bt_B12 significantly exceeds bt_B11,
   i.e., (bt_B12 - bt_B11) > bg_mean + N * bg_std (methane signature).
   Use --spectral-k (default 2.0) to set the sigma threshold.

SCAN MODE (diagnostic)
-----------------------
      python scripts/run_cemf_weisweiler_bt.py --scan
   Prints plume pixel count and Q̂ across a range of thresholds so you
   can see where the mask transitions from terrain-dominated to signal.

Usage:
    conda activate methane
    # Step 1 — generate BT TIF:
    python apply_bitemporal_diff.py --sites weisweiler --force-bt
    # Step 2 — run BT CEMF with the BT mask:
    python scripts/run_cemf_weisweiler_bt.py \\
        --bt-tif results_bitemporal/weisweiler/bitemporal_S2B_MSIL1C_20240918T103619_N0511_R008_T31UGS_20240918T142046.tif
    # Or scan thresholds first:
    python scripts/run_cemf_weisweiler_bt.py --scan
"""
import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = None

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.quantification.canonical_writer import (
    QuantificationRecord,
    write_quantification_record,
    DEFAULT_QUANT_PATH,
)
from src.ingestion.era5_client import ERA5Client

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("cemf_bt_weisweiler")

# ── Constants ─────────────────────────────────────────────────────────────────
B11_IDX        = 10
B12_IDX        = 11
PIXEL_AREA_20M = 400.0       # m² at native 20m SWIR resolution
DRY_AIR_COL    = 2.1e25      # molecules/m²
MOL_WEIGHT_CH4 = 0.016       # kg/mol
AVOGADRO       = 6.022e23
CEMF_ALPHA     = 4e-7        # Varon 2021 AMT Sec 2.2 sensitivity (reflectance / ppb·m)

# ── Site config ───────────────────────────────────────────────────────────────
SITE_ROW   = 6304
SITE_COL   = 3393
CROP_PX    = 500      # 5 km × 5 km crop centred on facility
PROB_THRESH = 0.026   # 5× mu_ctrl = 0.005277

TARGET_NPY  = "data/npy_cache/S2B_MSIL1C_20240918T103619_N0511_R008_T31UGS_20240918T142046.npy"
REF_NPY     = "data/npy_cache/T31UGS_ref_20240127.npy"
ORIG_TIF    = "results_bitemporal/weisweiler/original_S2B_MSIL1C_20240918T103619_N0511_R008_T31UGS_20240918T142046.tif"

SCENE_ID    = "S2B_MSIL1C_20240918T103619_N0511_R008_T31UGS_20240918T142046"
TIMESTAMP   = "2024-09-18T10:36:19Z"
LAT, LON    = 50.837, 6.322


def load_crop(npy_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (b11_crop, b12_crop) float32 at 10m — (CROP_PX, CROP_PX)."""
    half = CROP_PX // 2
    r0, r1 = SITE_ROW - half, SITE_ROW + half
    c0, c1 = SITE_COL - half, SITE_COL + half
    logger.info("Memory-mapping %s ...", npy_path)
    arr = np.load(npy_path, mmap_mode="r")
    assert arr.ndim == 3 and arr.shape[2] >= 12, f"Unexpected shape {arr.shape}"
    b11 = arr[r0:r1, c0:c1, B11_IDX].astype(np.float32)
    b12 = arr[r0:r1, c0:c1, B12_IDX].astype(np.float32)
    logger.info("  loaded B11 %s  B12 %s  from rows[%d:%d] cols[%d:%d]",
                b11.shape, b12.shape, r0, r1, c0, c1)
    return b11, b12


def load_mask(tif_path: str) -> np.ndarray:
    """Load original CH4Net probability TIF and crop to site region."""
    half = CROP_PX // 2
    r0, r1 = SITE_ROW - half, SITE_ROW + half
    c0, c1 = SITE_COL - half, SITE_COL + half
    logger.info("Loading TIF probability map %s ...", tif_path)
    img = Image.open(tif_path)
    # PIL crop: (left, upper, right, lower) = (col0, row0, col1, row1)
    crop = np.array(img.crop((c0, r0, c1, r1))).astype(np.float32)
    logger.info("  probability map crop: shape=%s  range=%.6f–%.6f",
                crop.shape, crop.min(), crop.max())
    return crop


def run_bt_cemf(
    b11_tgt: np.ndarray,
    b12_tgt: np.ndarray,
    b11_ref: np.ndarray,
    b12_ref: np.ndarray,
    prob_map: np.ndarray,
    prob_thresh: float,
) -> dict:
    """
    BT-corrected CEMF.  Returns a dict with the same keys used by CEMFResult.
    """
    # ── 1. BT difference bands ─────────────────────────────────────────────
    bt_b11 = b11_tgt - b11_ref   # terrain-cancelled seasonal change
    bt_b12 = b12_tgt - b12_ref

    logger.info("BT B12 stats: mean=%.4f  std=%.4f  min=%.4f  max=%.4f",
                bt_b12.mean(), bt_b12.std(), bt_b12.min(), bt_b12.max())
    logger.info("BT B11 stats: mean=%.4f  std=%.4f  min=%.4f  max=%.4f",
                bt_b11.mean(), bt_b11.std(), bt_b11.min(), bt_b11.max())

    # ── 2. Downsample to 20m (SWIR native resolution) ─────────────────────
    bt_b11_20 = bt_b11[::2, ::2]
    bt_b12_20 = bt_b12[::2, ::2]
    b11_raw_20 = b11_tgt[::2, ::2]   # raw target B11 at 20m (for sensitivity denom)

    # ── 3. Binary plume mask from original CH4Net probability ──────────────
    # Downsample to 20m first
    prob_20 = prob_map[::2, ::2]
    mask_bool = prob_20 > prob_thresh
    background_bool = ~mask_bool

    n_plume = int(mask_bool.sum())
    n_bg    = int(background_bool.sum())
    n_total = mask_bool.size
    logger.info("Probability threshold %.4g → plume=%d px (%.1f%%)  bg=%d px",
                prob_thresh, n_plume, 100.0 * n_plume / n_total, n_bg)

    if n_plume < 10:
        logger.error("Only %d plume pixels — threshold may need adjustment", n_plume)
        return {"valid": False, "reason": f"too_few_plume_pixels ({n_plume})"}

    if n_bg < 100:
        logger.error("Only %d background pixels — crop too small", n_bg)
        return {"valid": False, "reason": f"too_few_bg_pixels ({n_bg})"}

    # ── 4. Background reference: seasonal change in background pixels ──────
    mu_bt_b12_bg = float(bt_b12_20[background_bool].mean())
    mu_bt_b11_bg = float(bt_b11_20[background_bool].mean())
    mu_b11_raw_bg = float(b11_raw_20[background_bool].mean())   # absolute level

    logger.info("Background BT means: mu_bt_b12=%.6f  mu_bt_b11=%.6f  mu_b11_raw=%.6f",
                mu_bt_b12_bg, mu_bt_b11_bg, mu_b11_raw_bg)

    if mu_b11_raw_bg < 1e-6:
        return {"valid": False, "reason": "near_zero_b11_background"}

    # ── 5. BT-corrected anomaly ────────────────────────────────────────────
    d_bt_b12 = bt_b12_20 - mu_bt_b12_bg   # excess B12 seasonal change vs background
    d_bt_b11 = bt_b11_20 - mu_bt_b11_bg

    # ── 6. Per-pixel CH4 column enhancement ───────────────────────────────
    # dXCH4 (ppb·m): terrain cancels in numerator; raw B11 normalises sensitivity
    dxch4 = (d_bt_b12 - 0.5 * d_bt_b11) / (mu_b11_raw_bg * CEMF_ALPHA)
    dxch4 = np.clip(dxch4, 0, None)

    plume_dxch4 = dxch4[mask_bool]
    logger.info("Plume dXCH4 stats: mean=%.2f ppb·m  max=%.2f ppb·m  n=%d px",
                plume_dxch4.mean(), plume_dxch4.max(), n_plume)

    # ── 7. Mass integration ────────────────────────────────────────────────
    mass_per_px = (
        plume_dxch4 * 1e-9
        * DRY_AIR_COL
        * MOL_WEIGHT_CH4
        / AVOGADRO
        * PIXEL_AREA_20M
    )
    total_mass_kg = float(mass_per_px.sum())
    logger.info("Total plume mass (BT-corrected): %.4f kg  (%d plume pixels at 20m)",
                total_mass_kg, n_plume)

    return {
        "valid": True,
        "total_mass_kg": total_mass_kg,
        "n_plume_pixels": n_plume,
        "mu_b11_raw_bg": mu_b11_raw_bg,
        "mu_bt_b12_bg": mu_bt_b12_bg,
        "mu_bt_b11_bg": mu_bt_b11_bg,
        "dxch4_plume_mean": float(plume_dxch4.mean()),
        "dxch4_plume_max": float(plume_dxch4.max()),
    }


def scan_thresholds(
    b11_tgt, b12_tgt, b11_ref, b12_ref, prob_map, wind_speed
) -> None:
    """Print plume pixel count and Q̂ at a range of probability thresholds."""
    print("\n── Threshold scan ──────────────────────────────────────────")
    print(f"  {'thresh':>8s}  {'n_plume':>8s}  {'frac%':>6s}  {'mass_kg':>10s}  {'Q_kgh':>10s}")
    print("  " + "-"*55)
    for thresh in [0.01, 0.026, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        r = run_bt_cemf(b11_tgt, b12_tgt, b11_ref, b12_ref, prob_map, thresh)
        if not r["valid"]:
            print(f"  {thresh:>8.3f}  {r.get('reason','?'):>8s}")
            continue
        n = r["n_plume_pixels"]
        frac = 100.0 * n / (CROP_PX // 2) ** 2
        mass = r["total_mass_kg"]
        plume_len = float(np.sqrt(n * PIXEL_AREA_20M))
        q = round(mass * wind_speed / plume_len * 3600, 1) if plume_len > 0 else 0
        flag = "  ← plausible range?" if 50 < q < 500 else ""
        print(f"  {thresh:>8.3f}  {n:>8d}  {frac:>6.2f}  {mass:>10.3f}  {q:>10.1f}{flag}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="BT-corrected CEMF for Weisweiler 2024-09-18"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-era5", action="store_true",
                        help="Use climatological fallback wind")
    parser.add_argument("--prob-thresh", type=float, default=PROB_THRESH,
                        help=f"Probability threshold for original CH4Net TIF (default {PROB_THRESH})")
    parser.add_argument("--bt-tif", type=str, default=None,
                        help="Path to BT CH4Net probability TIF "
                             "(from: python apply_bitemporal_diff.py --sites weisweiler --force-bt). "
                             "Replaces the single-date TIF as the plume mask.")
    parser.add_argument("--use-spectral-mask", action="store_true",
                        help="Derive plume mask from BT band ratio instead of CH4Net TIF "
                             "(no neural network required; lower confidence)")
    parser.add_argument("--spectral-k", type=float, default=2.0,
                        help="Sigma multiplier for spectral mask (default 2.0)")
    parser.add_argument("--scan", action="store_true",
                        help="Print Q̂ across threshold range then exit (diagnostic)")
    args = parser.parse_args()

    # ── Load data ────────────────────────────────────────────────────────────
    b11_tgt, b12_tgt = load_crop(TARGET_NPY)
    b11_ref, b12_ref = load_crop(REF_NPY)

    if args.bt_tif:
        logger.info("Using BT CH4Net probability TIF: %s", args.bt_tif)
        prob_map = load_mask(args.bt_tif)
    elif args.use_spectral_mask:
        # Build mask from BT band differential: pixels where
        # (bt_b12 - bt_b11) exceeds background mean + k*std
        logger.info("Building spectral BT mask (k=%.1f) ...", args.spectral_k)
        bt_b11_raw = b11_tgt - b11_ref
        bt_b12_raw = b12_tgt - b12_ref
        bt_diff = bt_b12_raw.astype(np.float32) - bt_b11_raw.astype(np.float32)
        bg_mean = float(bt_diff.mean())
        bg_std  = float(bt_diff.std())
        threshold_val = bg_mean + args.spectral_k * bg_std
        prob_map = (bt_diff - bg_mean) / (bg_std + 1e-9)   # z-score as proxy probability
        logger.info("Spectral mask: bg_mean=%.4f  bg_std=%.4f  thresh=%.4f  "
                    "pixels above: %d (%.1f%%)",
                    bg_mean, bg_std, threshold_val,
                    int((bt_diff > threshold_val).sum()),
                    100.0 * (bt_diff > threshold_val).mean())
        # Map z-score to [0,1] probability proxy so existing thresh logic works
        # A z-score of spectral_k maps to prob_thresh=1.0; set prob_thresh accordingly
        args.prob_thresh = args.spectral_k   # threshold in z-score units
    else:
        prob_map = load_mask(ORIG_TIF)

    # ── Wind (fetch once, used by scan too) ──────────────────────────────────
    wind = {
        "wind_speed_ms": 3.5, "wind_dir_deg": None,
        "wind_source": "climatological_fallback_3.5ms",
        "era5_u_ms": None, "era5_v_ms": None,
    }
    if not args.no_era5:
        try:
            client = ERA5Client()
            wind = client.get_wind(lat=LAT, lon=LON,
                                   date_str=TIMESTAMP[:10], hour=TIMESTAMP[11:16])
            logger.info("ERA5 wind: %.2f m/s dir=%.1f° (%s)",
                        wind["wind_speed_ms"], wind.get("wind_dir_deg", 0), wind["wind_source"])
        except Exception as e:
            logger.warning("ERA5 fetch failed (%s) — using fallback", e)

    # ── Threshold scan mode ───────────────────────────────────────────────────
    if args.scan:
        scan_thresholds(b11_tgt, b12_tgt, b11_ref, b12_ref, prob_map,
                        wind["wind_speed_ms"])
        print("Re-run with  --prob-thresh <value>  (or --bt-tif / --use-spectral-mask)")
        print("to quantify at a specific threshold.")
        return

    # ── Run BT-corrected CEMF ────────────────────────────────────────────────
    result = run_bt_cemf(b11_tgt, b12_tgt, b11_ref, b12_ref, prob_map, args.prob_thresh)

    if not result["valid"]:
        logger.error("BT CEMF failed: %s", result.get("reason"))
        sys.exit(1)

    total_mass_kg = result["total_mass_kg"]
    n_plume = result["n_plume_pixels"]

    # ── IME: flow rate from mass + wind + plume length ────────────────────────
    # Plume length estimated from sqrt(n_pixels * pixel_area)
    plume_length_m = float(np.sqrt(n_plume * PIXEL_AREA_20M))
    v = wind["wind_speed_ms"] or 3.5
    flow_rate_kgh = round(total_mass_kg * v / plume_length_m * 3600, 2)

    from src.quantification.uncertainty import get_uncertainty_pct
    unc_pct = get_uncertainty_pct(wind["wind_source"])
    lo = round(flow_rate_kgh * (1 - unc_pct / 100), 2)
    hi = round(flow_rate_kgh * (1 + unc_pct / 100), 2)
    annual = round(flow_rate_kgh * 8760 / 1000, 4)

    logger.info(
        "BT CEMF result: mass=%.4f kg  plume_len=%.1f m  "
        "wind=%.2f m/s  Q̂=%.1f kg/h  [%.1f–%.1f]  annual=%.1f t/yr",
        total_mass_kg, plume_length_m, v, flow_rate_kgh, lo, hi, annual,
    )

    # Sanity check: Weisweiler is ~775 MW lignite
    # Neurath (4400 MW) gives ~338 kg/h → expect ~60–300 kg/h here
    if flow_rate_kgh > 800:
        logger.warning(
            "Q̂=%.1f kg/h seems high for 775 MW Weisweiler (Neurath 4400MW = 338 kg/h). "
            "Check if BT mask still contains terrain pixels.",
            flow_rate_kgh,
        )
        logger.warning(
            "  BT background means: mu_bt_b12=%.6f  mu_bt_b11=%.6f  diff=%.6f",
            result["mu_bt_b12_bg"], result["mu_bt_b11_bg"],
            result["mu_bt_b12_bg"] - result["mu_bt_b11_bg"],
        )

    if args.dry_run:
        print("\n[dry-run] Would write:")
        print(f"  site=weisweiler  flow_rate_kgh={flow_rate_kgh}  "
              f"cemf_valid=True  wind={wind['wind_source']}")
        print(f"  mass={total_mass_kg:.4f} kg  plume_len={plume_length_m:.1f} m  "
              f"n_plume={n_plume}  unc={unc_pct}%")
        return

    # ── Write canonical record ────────────────────────────────────────────────
    record = QuantificationRecord(
        site="weisweiler",
        scene_id=SCENE_ID,
        acquisition_timestamp=TIMESTAMP,
        plume_centroid_lat=LAT,
        plume_centroid_lon=LON,
        methodology="CEMF+IME",
        cemf_sensitivity_coeff="4e-7 (Varon 2021 AMT Sec 2.2)",
        mask_source="ch4net_v8_original",
        mask_file=ORIG_TIF,
        n_plume_pixels=n_plume,
        total_mass_kg=round(total_mass_kg, 4),
        plume_length_m=round(plume_length_m, 1),
        wind_speed_ms=wind["wind_speed_ms"],
        wind_dir_deg=wind.get("wind_dir_deg"),
        wind_source=wind["wind_source"],
        era5_u_ms=wind.get("era5_u_ms"),
        era5_v_ms=wind.get("era5_v_ms"),
        flow_rate_kgh=flow_rate_kgh,
        flow_rate_lower_kgh=lo,
        flow_rate_upper_kgh=hi,
        uncertainty_pct=unc_pct,
        annual_tonnes_if_continuous=annual,
        cemf_valid=flow_rate_kgh < 800,   # mark implausible if still too high
        excluded=False,
        exclusion_reason=None,
        tropomi_confirm=False,
        ch4net_peak_probability=0.42,
        cloud_cover_quality="clear",
        retrieval_notes=(
            "BT-corrected CEMF: B12/B11 channels differenced against winter reference "
            "(T31UGS_ref_20240127.npy) to cancel stable terrain scatter. "
            "CH4Net S/C=23.46 (T31UGS, 2024-09-18), CFAR not triggered (CV=1.607). "
            "ERA5: {:.2f} m/s, dir={}. "
            "Probability threshold: {:.4g} (5× mu_ctrl). "
            "Q̂={:.1f} kg/h ±{}% [{:.1f}–{:.1f}]. "
            "BT correction: mu_bt_b12_bg={:.6f}, mu_bt_b11_bg={:.6f}."
        ).format(
            wind["wind_speed_ms"], wind.get("wind_dir_deg", "?"), args.prob_thresh,
            flow_rate_kgh, unc_pct, lo, hi,
            result["mu_bt_b12_bg"], result["mu_bt_b11_bg"],
        ),
    )

    write_quantification_record(record)
    logger.info("Written to %s", DEFAULT_QUANT_PATH)
    print(f"\nWeisweiler BT CEMF:  Q̂={flow_rate_kgh} kg/h  [{lo}–{hi}]  "
          f"wind={wind['wind_speed_ms']:.2f} m/s ({wind['wind_source']})  "
          f"annual={annual:.0f} t/yr")


if __name__ == "__main__":
    main()
