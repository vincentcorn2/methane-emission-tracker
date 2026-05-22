"""
scripts/synthetic_only_validation.py
======================================
Empirical defence of the synthetic-plume training strategy.

What this answers
-----------------
Ali Hirsa's question: "Is the model just learning the synthesis artifact instead
of methane?" If a model trained ONLY on synthetic plumes (no real positives)
still detects the 14 real positive crops at test time, the synthetic
distribution is generalising to real plumes — synthetic data is doing physical
work, not just memorisation. If it fails to detect real plumes, synthetic data
is overfit to the generation artifact, and that needs to be in limitations.

How it works
------------
1. Move the 14 real positive label files out of view (rename .json → .json.hidden).
2. Re-run approach_c_retrain.py — the dataset now contains 51 synthetic + 22
   real negatives. Outputs weights to weights/synthetic_only_v1.pth.
3. Restore the real positive labels.
4. Load synthetic_only_v1.pth and the production european_model_v8.pth.
5. Run inference on the 14 real positive crops with both models. Compute the
   detection-mask plume probability and the synthetic-only / v8 agreement
   on each crop.

Output
------
results_analysis/synthetic_only_validation.json
results_analysis/synthetic_only_validation.md

Usage
-----
  conda activate methane
  python scripts/synthetic_only_validation.py --dry-run   # preview what will be touched
  python scripts/synthetic_only_validation.py             # full run, ~1 hour wall-clock
  python scripts/synthetic_only_validation.py --skip-train --weights weights/synthetic_only_v1.pth
      # if training already done; just run the comparison

CAUTION
-------
This script renames files in data/crops/positive/. If it crashes between step 1
and step 3, run --restore to put the labels back. The script also creates a
.synthetic_only_validation.lock sentinel file while real positives are hidden;
if the lock is present at startup, the script will refuse to proceed.
"""
import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Ensure we can import the production detector
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.detection.ch4net_model import CH4NetDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("synth_validation")

# ── Config ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
POSITIVES_DIR = ROOT / "data" / "crops" / "positive"
OUT_DIR = ROOT / "results_analysis"
LOCK_FILE = ROOT / ".synthetic_only_validation.lock"

V8_WEIGHTS = ROOT / "weights" / "european_model_v8.pth"
SYNTHETIC_ONLY_WEIGHTS = ROOT / "weights" / "synthetic_only_v1.pth"

RETRAIN_SCRIPT = ROOT / "approach_c_retrain.py"

# Detection threshold on the per-crop mean probability. We use a much lower
# threshold than the production 0.18 because we are comparing two models'
# RELATIVE behaviour on the same crops, not making a production detection call.
PROB_DETECT_THRESHOLD = 0.10
PROB_STRONG_THRESHOLD = 0.30


def hide_real_positives() -> list[tuple[Path, Path]]:
    """Rename every .json label file in data/crops/positive/ to add .hidden suffix.
    Returns list of (original, hidden) pairs for restoration."""
    pairs = []
    for label in sorted(POSITIVES_DIR.glob("*_label.json")):
        hidden = label.with_suffix(".json.hidden")
        if hidden.exists():
            log.warning("Skipping %s — already hidden", label.name)
            continue
        label.rename(hidden)
        pairs.append((label, hidden))
    log.info("Hid %d real positive label files", len(pairs))
    return pairs


def restore_real_positives() -> int:
    """Undo hide_real_positives(). Returns number of files restored."""
    restored = 0
    for hidden in sorted(POSITIVES_DIR.glob("*_label.json.hidden")):
        original = hidden.with_suffix("")          # strip .hidden
        # `.with_suffix("")` removes the last suffix only; on a multi-suffix file
        # like "foo_label.json.hidden", that yields "foo_label.json". Good.
        if original.exists():
            log.warning("Skipping %s — original already exists at %s", hidden, original)
            continue
        hidden.rename(original)
        restored += 1
    log.info("Restored %d real positive label files", restored)
    return restored


def run_retrain():
    """Call approach_c_retrain.py with synthetic-only weights output."""
    cmd = [
        sys.executable, str(RETRAIN_SCRIPT),
        "--weights-out", str(SYNTHETIC_ONLY_WEIGHTS),
    ]
    log.info("Running retrain: %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(ROOT))
    if proc.returncode != 0:
        raise RuntimeError(f"Retrain failed with exit code {proc.returncode}")


def load_positive_crop(npy_path: Path) -> np.ndarray:
    """Load a 12-band uint8 crop and return a 160x160 center crop ready for inference.

    Real positive crops in data/crops/positive/ are not all 200x200 — several
    are rectangular (e.g. 200x141 for de_bad_lauchstaedt). U-Net requires
    H and W divisible by 16 (2^4 max-pool stages). Strategy: pad shorter
    dimension to at least 160 with edge-replicated pixels (visually neutral),
    then center-crop to 160x160.
    """
    arr = np.load(npy_path)
    H, W, C = arr.shape
    if C != 12:
        raise ValueError(f"Expected 12-band crop, got {arr.shape} for {npy_path}")

    # Pad to at least 160x160 if either dim is short
    pad_h = max(0, 160 - H)
    pad_w = max(0, 160 - W)
    if pad_h > 0 or pad_w > 0:
        before_h, after_h = pad_h // 2, pad_h - pad_h // 2
        before_w, after_w = pad_w // 2, pad_w - pad_w // 2
        arr = np.pad(arr,
                     ((before_h, after_h), (before_w, after_w), (0, 0)),
                     mode="edge")
        H, W = arr.shape[:2]

    # Center-crop to exactly 160x160
    r0 = (H - 160) // 2
    c0 = (W - 160) // 2
    return arr[r0:r0 + 160, c0:c0 + 160, :]


def evaluate_on_real_positives(weights_path: Path, label_pairs: list):
    """Run inference on each real positive crop using the given weights.
    Returns list of dicts with per-crop detection statistics."""
    detector = CH4NetDetector(str(weights_path))
    results = []

    # Even after restoration we need to find the originals. The hidden→original
    # rename happens before evaluation, so just glob the label JSONs.
    real_labels = sorted(POSITIVES_DIR.glob("*_label.json"))
    log.info("Evaluating %s on %d real positive crops", weights_path.name, len(real_labels))

    for label_path in real_labels:
        crop_path = label_path.with_name(label_path.stem.replace("_label", "") + ".npy")
        if not crop_path.exists():
            log.warning("Missing crop for label %s", label_path.name)
            continue

        meta = json.loads(label_path.read_text())
        crop = load_positive_crop(crop_path)
        det = detector.detect(crop)
        prob = det.probability_map

        results.append({
            "crop":              crop_path.name,
            "site":              meta.get("site", crop_path.stem.split("_T")[0]),
            "tropomi_enh_ppb":   meta.get("tropomi_enhancement_ppb"),
            "prob_mean":         float(prob.mean()),
            "prob_max":          float(prob.max()),
            "prob_peak_p95":     float(np.quantile(prob, 0.95)),
            "n_above_0_10":      int((prob > PROB_DETECT_THRESHOLD).sum()),
            "n_above_0_30":      int((prob > PROB_STRONG_THRESHOLD).sum()),
            "detected_at_010":   bool((prob > PROB_DETECT_THRESHOLD).sum() >= 115),
            "detected_at_030":   bool((prob > PROB_STRONG_THRESHOLD).sum() >= 115),
        })

    return results


def build_report(v8_results, synth_results):
    md = []
    md.append("# Synthetic-only retraining validation\n")
    md.append("**Question:** is CH4Net v8's performance driven by the 51 synthetic plume "
              "injections, or by the 14 real positive crops? If a model trained on synthetic "
              "data alone still detects the 14 real plumes, the synthetic distribution is "
              "generalising — which validates the augmentation strategy.\n")

    md.append("## Per-crop comparison\n")
    md.append("| Crop | TROPOMI ΔXCH₄ (ppb) | v8 prob mean | v8 prob max | "
              "synth-only prob mean | synth-only prob max | "
              "v8 detect | synth-only detect |")
    md.append("|---|---|---|---|---|---|---|---|")
    pairs = list(zip(v8_results, synth_results))

    agree_pos = 0
    v8_det = 0
    synth_det = 0
    for v8, sy in pairs:
        agree = v8["detected_at_010"] and sy["detected_at_010"]
        if agree:
            agree_pos += 1
        if v8["detected_at_010"]:
            v8_det += 1
        if sy["detected_at_010"]:
            synth_det += 1
        md.append(f"| {v8['crop']} | {v8.get('tropomi_enh_ppb')} | "
                  f"{v8['prob_mean']:.3f} | {v8['prob_max']:.3f} | "
                  f"{sy['prob_mean']:.3f} | {sy['prob_max']:.3f} | "
                  f"{'✓' if v8['detected_at_010'] else '·'} | "
                  f"{'✓' if sy['detected_at_010'] else '·'} |")

    md.append("")
    n = len(pairs)
    md.append("## Summary statistics\n")
    md.append(f"- v8 detections on real positives: **{v8_det}/{n}** "
              f"({100*v8_det/n:.0f}%)")
    md.append(f"- Synthetic-only detections on real positives: **{synth_det}/{n}** "
              f"({100*synth_det/n:.0f}%)")
    md.append(f"- Both models agree (both detect): **{agree_pos}/{n}** "
              f"({100*agree_pos/n:.0f}%)")
    md.append("")
    md.append("## Interpretation\n")
    rate = synth_det / n if n else 0
    if rate >= 0.7:
        md.append("**Synthetic plume training generalises to real plumes** "
                  f"({100*rate:.0f}% detection rate). The Gaussian B12-attenuation "
                  "augmentation is producing samples close enough to the real-plume "
                  "manifold that the model can recognise real plumes it has never seen. "
                  "This is the answer to the 'is the model just learning the synthesis "
                  "artifact' question — it is not.\n")
    elif rate >= 0.4:
        md.append("**Synthetic plume training partially generalises** "
                  f"({100*rate:.0f}% detection rate). The augmentation is doing some "
                  "physical work but the production model is leaning on the 14 real "
                  "positive crops for the harder cases. This should be acknowledged in "
                  "limitations.\n")
    else:
        md.append("**Synthetic plume training does not generalise on its own** "
                  f"({100*rate:.0f}% detection rate). The model trained without real "
                  "positives fails to detect most real plumes, indicating the synthetic "
                  "distribution is too narrow. This is a limitation to report explicitly "
                  "and the synthetic generation procedure should be redesigned before "
                  "deployment.\n")

    return "\n".join(md)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview what will be touched; do not modify files or train")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip retraining; evaluate existing synthetic-only weights")
    parser.add_argument("--weights", type=Path, default=SYNTHETIC_ONLY_WEIGHTS,
                        help="Path to synthetic-only weights (output, or input if --skip-train)")
    parser.add_argument("--restore", action="store_true",
                        help="Restore hidden real positive labels (use if a previous run crashed)")
    args = parser.parse_args()

    if args.restore:
        restored = restore_real_positives()
        if LOCK_FILE.exists():
            LOCK_FILE.unlink()
        log.info("Restore complete (%d files). Exit.", restored)
        return

    if LOCK_FILE.exists():
        log.error("Lock file exists at %s. A previous run did not clean up. "
                  "Run with --restore to recover.", LOCK_FILE)
        sys.exit(1)

    real_labels = list(POSITIVES_DIR.glob("*_label.json"))
    if args.dry_run:
        log.info("DRY RUN")
        log.info("Would hide %d real positive labels in %s", len(real_labels), POSITIVES_DIR)
        log.info("Would retrain via: %s --weights-out %s", RETRAIN_SCRIPT, args.weights)
        log.info("Would evaluate v8 and synthetic-only on the %d real positive crops",
                 len(real_labels))
        log.info("Would write report to %s", OUT_DIR / "synthetic_only_validation.md")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_train:
        LOCK_FILE.write_text("synthetic-only retrain in progress")
        pairs = []
        try:
            pairs = hide_real_positives()
            run_retrain()
        finally:
            restore_real_positives()
            if LOCK_FILE.exists():
                LOCK_FILE.unlink()

    log.info("Evaluating v8 on real positives ...")
    v8_results = evaluate_on_real_positives(V8_WEIGHTS, [])

    log.info("Evaluating synthetic-only on real positives ...")
    synth_results = evaluate_on_real_positives(args.weights, [])

    out_obj = {
        "v8_results": v8_results,
        "synthetic_only_results": synth_results,
        "v8_weights": str(V8_WEIGHTS),
        "synthetic_only_weights": str(args.weights),
    }
    (OUT_DIR / "synthetic_only_validation.json").write_text(json.dumps(out_obj, indent=2))

    report = build_report(v8_results, synth_results)
    (OUT_DIR / "synthetic_only_validation.md").write_text(report)
    print(report)

    log.info("Done.")
    log.info("  results_analysis/synthetic_only_validation.md")
    log.info("  results_analysis/synthetic_only_validation.json")


if __name__ == "__main__":
    main()
