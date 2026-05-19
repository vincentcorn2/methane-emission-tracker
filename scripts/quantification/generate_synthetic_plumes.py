#!/usr/bin/env python3
"""
generate_synthetic_plumes.py — Physics-informed synthetic methane plume augmentation.

Problem
-------
CH4Net v2 has 13.5M parameters but only 21 European training crops. All fine-tuning
attempts (v1-v4) fail: either catastrophic overfitting or the all-zero plateau.
Synthetic augmentation is the standard approach in the methane detection literature
(Vaughan et al. 2024, AMT) to overcome data scarcity.

Method
------
Takes each real negative crop (verified non-emitter, real European terrain) and
injects a physics-informed synthetic methane absorption signature:

  1. Methane absorbs at 2190nm (B12, channel index 11).
  2. Methane is transparent at 1610nm (B11, channel index 10).
  3. The plume is modelled as a 2D Gaussian with random center, sigma, and intensity.
  4. B12 is attenuated: B12_plume = B12_bg * (1 - alpha * plume_mask)
     where alpha ~ [0.03, 0.12] simulates column densities from weak to strong plumes.
  5. All other bands are unchanged.
  6. The plume_mask itself becomes the training label.

This creates realistic positive training instances where:
  - The background terrain is genuinely European (real Sentinel-2 data)
  - The spectral signature is physically correct (B12 absorbs, B11 doesn't)
  - The spatial morphology is plausible (Gaussian approximation of point-source plume)

Output
------
  data/crops/synthetic/
    synth_{source_site}_{idx}.npy           — 200x200x12 uint8 array
    synth_{source_site}_{idx}_label.json    — metadata + label_value=1 + split=train

Usage
-----
  conda activate methane
  python generate_synthetic_plumes.py                    # 10 plumes per negative (default)
  python generate_synthetic_plumes.py --per-crop 20      # 20 plumes per negative
  python generate_synthetic_plumes.py --clear             # delete existing synthetics first
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

CROPS_DIR    = Path('data/crops')
SYNTH_DIR    = CROPS_DIR / 'synthetic'
NEG_DIR      = CROPS_DIR / 'negative'

B12_IDX      = 11   # 2190nm — methane absorption band
B11_IDX      = 10   # 1610nm — reference band (transparent to CH4)

# Plume simulation parameters (physically motivated ranges)
# v8: back to v6 range (10-25%). v7 at 5-15% was worse — weaker synthetic plumes
# caused the model to calibrate its threshold too low, destroying Weisweiler.
# 10-25% alpha matches Weisweiler's actual industrial B12 contrast level.
ALPHA_MIN    = 0.10  # weak plume: ~10% B12 attenuation at plume centre
ALPHA_MAX    = 0.25  # strong plume: ~25% B12 attenuation at plume centre
SIGMA_MIN    = 15    # small plume: ~150m at 10m/px
SIGMA_MAX    = 50    # large plume: ~500m
CENTRE_PAD   = 30    # keep plume centre at least 30px from edge


def generate_plume(bg_arr, rng):
    """
    Inject a synthetic methane plume into a background crop.

    Parameters
    ----------
    bg_arr : np.ndarray (200, 200, 12) uint8
        Background Sentinel-2 crop (real negative terrain).
    rng : np.random.Generator
        Random number generator for reproducibility.

    Returns
    -------
    plume_arr : np.ndarray (200, 200, 12) uint8
        Modified crop with synthetic methane absorption in B12.
    plume_mask : np.ndarray (200, 200) float32
        Normalised plume mask [0, 1] for training label.
    params : dict
        Plume generation parameters for metadata.
    """
    H, W = bg_arr.shape[:2]

    # Random plume parameters
    cy    = rng.integers(CENTRE_PAD, H - CENTRE_PAD)
    cx    = rng.integers(CENTRE_PAD, W - CENTRE_PAD)
    sigma = rng.uniform(SIGMA_MIN, SIGMA_MAX)
    alpha = rng.uniform(ALPHA_MIN, ALPHA_MAX)

    # Optional: slight ellipticity + rotation to mimic wind-dispersed plumes
    aspect = rng.uniform(0.7, 1.4)      # ellipse aspect ratio
    angle  = rng.uniform(0, np.pi)      # rotation angle

    # Build 2D Gaussian plume mask
    y, x = np.ogrid[:H, :W]
    dy = y - cy
    dx = x - cx

    # Rotate coordinates
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    dy_r = dy * cos_a - dx * sin_a
    dx_r = dy * sin_a + dx * cos_a

    # Elliptical Gaussian
    dist2 = (dy_r / sigma) ** 2 + (dx_r / (sigma * aspect)) ** 2
    plume_mask = np.exp(-0.5 * dist2).astype(np.float32)

    # Apply methane absorption to B12 only
    # Physics: transmittance T = 1 - alpha * plume_mask
    # B12_observed = B12_background * T
    result = bg_arr.copy()
    b12    = result[:, :, B12_IDX].astype(np.float32)
    transmittance = 1.0 - alpha * plume_mask
    b12_plume     = np.clip(b12 * transmittance, 0, 255).astype(np.uint8)
    result[:, :, B12_IDX] = b12_plume

    # Verify B11 unchanged (sanity check)
    assert np.array_equal(result[:, :, B11_IDX], bg_arr[:, :, B11_IDX])

    params = {
        'centre_y': int(cy),
        'centre_x': int(cx),
        'sigma': float(sigma),
        'alpha': float(alpha),
        'aspect': float(aspect),
        'angle_rad': float(angle),
        'b12_mean_attenuation_dn': float((b12 - b12_plume.astype(np.float32)).mean()),
        'b12_max_attenuation_dn': float((b12 - b12_plume.astype(np.float32)).max()),
    }

    return result, plume_mask, params


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic methane plumes on negative background crops')
    parser.add_argument('--per-crop', type=int, default=10,
                        help='Number of synthetic plumes per negative crop (default: 10)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--clear', action='store_true',
                        help='Delete existing synthetic crops before generating')
    args = parser.parse_args()

    # Setup
    SYNTH_DIR.mkdir(parents=True, exist_ok=True)

    if args.clear:
        for f in SYNTH_DIR.glob('*'):
            f.unlink()
        log.info('Cleared existing synthetic crops')

    rng = np.random.default_rng(args.seed)

    # Load all negative crops
    neg_npys = sorted(NEG_DIR.glob('*.npy'))
    neg_npys = [p for p in neg_npys if '_label' not in p.stem]
    log.info('Found %d negative background crops', len(neg_npys))

    if not neg_npys:
        log.error('No negative crops found in %s. Run extract_training_crops.py first.', NEG_DIR)
        sys.exit(1)

    total_generated = 0
    total_skipped   = 0

    for npy_path in neg_npys:
        site_name = npy_path.stem  # e.g. "belchatow_T34UCB_20240824"

        # Load background
        bg = np.load(npy_path)
        if bg.shape[0] != 200 or bg.shape[1] != 200:
            log.warning('Skipping non-200x200 crop: %s  shape=%s', npy_path.name, bg.shape)
            continue

        # Check B12 has enough signal to attenuate
        b12_mean = bg[:, :, B12_IDX].mean()
        if b12_mean < 5:
            log.warning('Skipping %s — B12 mean too low (%.1f DN), attenuation would be invisible',
                        site_name, b12_mean)
            continue

        log.info('')
        log.info('Source: %-35s  B12_mean=%.1f DN', site_name, b12_mean)

        for i in range(args.per_crop):
            out_stem = f'synth_{site_name}_{i:03d}'
            out_npy  = SYNTH_DIR / f'{out_stem}.npy'
            out_json = SYNTH_DIR / f'{out_stem}_label.json'

            if out_npy.exists():
                total_skipped += 1
                continue

            plume_arr, plume_mask, params = generate_plume(bg, rng)

            # Save crop
            np.save(out_npy, plume_arr)

            # Save label (matches extract_training_crops.py format)
            label = {
                'site': f'synth_{site_name}',
                'source_site': site_name,
                'label_value': 1,
                'split': 'train',
                'synthetic': True,
                'plume_params': params,
                'plume_centre_y': params['centre_y'],
                'plume_centre_x': params['centre_x'],
                'enhancement_ppb': round(params['alpha'] * 200, 1),  # rough ppb estimate
                'extracted': datetime.now(timezone.utc).isoformat(),
            }
            with open(out_json, 'w') as f:
                json.dump(label, f, indent=2)

            total_generated += 1

        log.info('  Generated %d synthetic plumes (B12 attenuation range: %.1f–%.1f DN)',
                 args.per_crop, b12_mean * ALPHA_MIN, b12_mean * ALPHA_MAX)

    log.info('')
    log.info('=' * 60)
    log.info('SYNTHETIC PLUME GENERATION COMPLETE')
    log.info('=' * 60)
    log.info('  New synthetic crops:  %d', total_generated)
    log.info('  Skipped (existing):   %d', total_skipped)
    log.info('  Source negatives:     %d', len(neg_npys))
    log.info('  Output directory:     %s', SYNTH_DIR)
    log.info('')
    log.info('  Next: run approach_c_retrain.py to train with augmented dataset.')
    log.info('  The training script will auto-discover synthetic crops in data/crops/synthetic/')


if __name__ == '__main__':
    main()
