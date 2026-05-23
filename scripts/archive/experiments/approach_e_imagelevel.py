#!/usr/bin/env python3
"""
approach_e_imagelevel.py — Image-level classification loss for CH4Net fine-tuning.

Why this is different from approach_c_retrain.py
-------------------------------------------------
approach_c uses pixel-level BCEWithLogitsLoss on sparse Gaussian plume masks.
This has two problems with our dataset:
  1. Pixel imbalance: plume pixels are <5% of each crop → model finds the
     all-zero attractor (predict background everywhere, loss is small).
  2. Wrong objective: we care about S/C = mean_prob(site) / mean_prob(control),
     not per-pixel accuracy. Pixel-level loss doesn't directly optimise this.

This script replaces the loss with image-level BCE:
  - Compute mean sigmoid output over the centre 80×80 pixels of each crop.
  - Compare this scalar to the image-level label (1=emitter, 0=clean terrain).
  - Minimise BCE between these scalars across the batch.

This directly optimises what the S/C metric measures, cannot be minimised by
predicting all-zeros (a positive crop at mean=0 incurs full BCE penalty), and
is tractable with 45 training images. Recommended by Gemini deep research
for few-shot remote sensing adaptation with sparse foreground signals.

Usage
-----
  python approach_e_imagelevel.py                          # default settings
  python approach_e_imagelevel.py --epochs 120
  python approach_e_imagelevel.py --dry-run
  python approach_e_imagelevel.py --weights-in weights/european_model.pth  # warm-start from v8
"""

import os
import sys
import json
import logging
import argparse
import re
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    print("ERROR: PyTorch not found. Run: conda activate methane")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent))
from src.detection.ch4net_model import Unet

# ── Config ─────────────────────────────────────────────────────────────────────
CROP_OUT      = 160     # random crop size extracted from 200×200 input
CENTRE_RADIUS = 40      # half-width of centre window used in loss & S/C proxy
                        # → 80×80 px window centred on crop = 800m × 800m at 10m/px
BATCH_SIZE    = 4
EPOCHS        = 120     # more headroom — image-level loss converges differently
LR            = 5e-5
LR_STEP       = 10
LR_DECAY      = 0.95
WEIGHT_DECAY  = 1e-4    # same soft regularizer as approach_c
PATIENCE      = 25      # slightly more patience for noisier image-level signal
CROPS_DIR     = Path("data/crops")
WEIGHTS_IN    = "weights/best_model.pth"    # start from global base
WEIGHTS_OUT   = "weights/european_model_imagelevel.pth"
LOG_FILE      = "results_analysis/retrain_imagelevel.log"

# Sample-level positive weight. With 4:1 pos/neg training ratio, upweight
# negatives by 4× so each class contributes equally to the loss gradient.
# (Unlike pixel-level BCE where pos_weight=15 was needed for sparse plumes,
# here the imbalance is mild and sample weighting is sufficient.)
NEG_WEIGHT    = 4.0     # loss weight applied to negative samples

# ── Logging ────────────────────────────────────────────────────────────────────
Path("results_analysis").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE),
    ],
)
log = logging.getLogger(__name__)


# ── Dataset ────────────────────────────────────────────────────────────────────

class MethaneDataset(Dataset):
    """
    Yields (image, label_float) tuples.

    image       : FloatTensor (12, CROP_OUT, CROP_OUT)  normalised [0,1]
    label_float : FloatTensor scalar  0.0=negative  1.0=positive

    No pixel masks are generated — this dataset is for image-level classification.
    Augmentation: random 160×160 crop + horizontal/vertical flip + transpose.
    """

    def __init__(self, crops_dir, split="train", augment=True):
        self.split   = split
        self.augment = augment and (split == "train")
        self.samples = []

        for label_path in sorted(Path(crops_dir).glob("**/*_label.json")):
            with open(label_path) as f:
                meta = json.load(f)
            if meta.get("split") != split:
                continue
            npy_path = Path(str(label_path).replace("_label.json", ".npy"))
            if not npy_path.exists():
                continue
            arr = np.load(npy_path, mmap_mode="r")
            if arr.shape[0] != 200 or arr.shape[1] != 200:
                continue
            self.samples.append({
                "npy":   str(npy_path),
                "label": float(meta["label_value"]),
                "site":  meta.get("site", "?"),
            })

        n_pos = sum(1 for s in self.samples if s["label"] == 1.0)
        n_neg = len(self.samples) - n_pos
        log.info("Dataset split=%-5s  total=%d  pos=%d  neg=%d",
                 split, len(self.samples), n_pos, n_neg)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        arr = np.load(s["npy"]).astype(np.float32) / 255.0   # (200, 200, 12)

        # Random 160×160 crop
        pad = 200 - CROP_OUT
        if self.augment:
            r0 = int(np.random.randint(0, pad + 1))
            c0 = int(np.random.randint(0, pad + 1))
        else:
            r0 = c0 = pad // 2

        crop = arr[r0:r0 + CROP_OUT, c0:c0 + CROP_OUT, :]   # (160, 160, 12)

        # Augmentation
        if self.augment:
            if np.random.rand() > 0.5:
                crop = crop[:, ::-1, :].copy()
            if np.random.rand() > 0.5:
                crop = crop[::-1, :, :].copy()
            if np.random.rand() > 0.5:
                crop = np.transpose(crop, (1, 0, 2)).copy()

        img   = torch.from_numpy(crop.transpose(2, 0, 1))          # (12, H, W)
        label = torch.tensor(s["label"], dtype=torch.float32)       # scalar
        return img, label


# ── Model ──────────────────────────────────────────────────────────────────────

def load_model(weights_path, device):
    sd = torch.load(weights_path, map_location=device)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]

    div_factor = 8
    if "out.weight" in sd:
        in_ch      = sd["out.weight"].shape[1]
        div_factor = max(1, 128 // in_ch)
    log.info("Checkpoint div_factor=%d  (out_channels=%d)", div_factor, 128 // div_factor)

    model = Unet(in_channels=12, out_channels=1,
                 div_factor=div_factor, prob_output=False)

    remapped = {}
    for k, v in sd.items():
        k = re.sub(r'^inc\.(\d)',           r'inc.net.\1',        k)
        k = re.sub(r'^(down\d)\.1\.(\d)',   r'\1.net.1.net.\2',   k)
        k = re.sub(r'^(up\d\.conv)\.(\d)',  r'\1.net.\2',         k)
        remapped[k] = v

    model.load_state_dict(remapped, strict=True)
    model.to(device)
    return model


# ── Loss ───────────────────────────────────────────────────────────────────────

def image_level_loss(logits, labels, neg_weight=NEG_WEIGHT, radius=CENTRE_RADIUS):
    """
    Image-level BCE on the mean sigmoid output over the centre window.

    logits : (B, 1, H, W) raw model output
    labels : (B,) float  0.0 or 1.0

    Steps:
      1. Sigmoid → probabilities
      2. Average over centre RADIUS×RADIUS window → scalar per image
      3. BCE vs image-level label
      4. Upweight negative samples by neg_weight to balance 4:1 pos/neg ratio
    """
    probs = torch.sigmoid(logits)                                   # (B, 1, H, W)
    c = logits.shape[2] // 2
    r = radius
    centre_mean = probs[:, 0, c-r:c+r, c-r:c+r].mean(dim=(1, 2))  # (B,)

    # Per-sample weights: negatives upweighted to equalise class contribution
    weights = torch.ones_like(labels)
    weights[labels == 0.0] = neg_weight

    loss = F.binary_cross_entropy(centre_mean, labels,
                                  weight=weights, reduction="mean")
    return loss, centre_mean


# ── Training ───────────────────────────────────────────────────────────────────

def run_epoch(model, loader, optimiser, device, train=True):
    model.train() if train else model.eval()
    losses, correct, total = [], 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs   = imgs.to(device)
            labels = labels.to(device)

            logits = model(imgs).permute(0, 3, 1, 2)   # (B, 1, H, W)
            loss, centre_mean = image_level_loss(logits, labels)

            if train:
                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimiser.step()

            losses.append(loss.item())

            preds   = (centre_mean > 0.5).float()
            correct += (preds == labels).sum().item()
            total   += len(labels)

    return float(np.mean(losses)), correct / max(total, 1)


def train(args):
    device = torch.device(
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )
    log.info("Device: %s", device)

    train_ds = MethaneDataset(CROPS_DIR, split="train", augment=True)
    val_ds   = MethaneDataset(CROPS_DIR, split="val",   augment=False)

    if len(train_ds) == 0:
        log.error("No training crops found. Run extract_training_crops.py first.")
        sys.exit(1)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0, drop_last=False)

    log.info("Loading weights: %s", args.weights_in)
    model = load_model(args.weights_in, device)
    log.info("No encoder freezing — full fine-tuning with weight_decay=%.0e", WEIGHT_DECAY)

    optimiser = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimiser, step_size=LR_STEP, gamma=LR_DECAY)

    epochs         = 1 if args.dry_run else args.epochs
    best_val_loss  = float("inf")
    patience_count = 0
    history        = []

    log.info("=" * 62)
    log.info("  European fine-tuning (image-level loss) — %d train / %d val  |  %d epochs",
             len(train_ds), len(val_ds), epochs)
    log.info("  lr=%.1e  weight_decay=%.0e  neg_weight=%.1f  patience=%d",
             args.lr, WEIGHT_DECAY, NEG_WEIGHT, PATIENCE)
    log.info("  centre_radius=%dpx → %dx%d window in %dx%d crop",
             CENTRE_RADIUS, 2*CENTRE_RADIUS, 2*CENTRE_RADIUS, CROP_OUT, CROP_OUT)
    log.info("=" * 62)

    for epoch in range(1, epochs + 1):
        t_loss, t_acc = run_epoch(model, train_loader, optimiser, device, train=True)
        v_loss, v_acc = run_epoch(model, val_loader,   optimiser, device, train=False)
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]

        log.info("Ep %3d/%d  train=%.4f (acc=%.0f%%)  val=%.4f (acc=%.0f%%)  lr=%.1e",
                 epoch, epochs, t_loss, t_acc*100, v_loss, v_acc*100, lr_now)
        history.append({"epoch": epoch, "train_loss": t_loss, "train_acc": t_acc,
                        "val_loss": v_loss, "val_acc": v_acc})

        if not args.dry_run:
            if v_loss < best_val_loss:
                best_val_loss  = v_loss
                patience_count = 0
                torch.save(model.state_dict(), args.weights_out)
                log.info("  -> Saved best model (val_loss=%.4f)", v_loss)
            else:
                patience_count += 1
                if patience_count >= PATIENCE:
                    log.info("Early stop — no val improvement for %d epochs.", PATIENCE)
                    break

    log.info("=" * 62)
    if args.dry_run:
        log.info("Dry run complete — no weights saved.")
    else:
        log.info("Done. Best val loss: %.4f", best_val_loss)
        log.info("Weights: %s", args.weights_out)
        hist_path = "results_analysis/retrain_imagelevel_history.json"
        with open(hist_path, "w") as f:
            json.dump({"config": vars(args), "history": history}, f, indent=2)
        log.info("History: %s", hist_path)


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Fine-tune CH4Net with image-level classification loss")
    p.add_argument("--weights-in",  default=WEIGHTS_IN)
    p.add_argument("--weights-out", default=WEIGHTS_OUT)
    p.add_argument("--epochs",      type=int,   default=EPOCHS)
    p.add_argument("--lr",          type=float, default=LR)
    p.add_argument("--batch-size",  type=int,   default=BATCH_SIZE)
    p.add_argument("--dry-run",     action="store_true",
                   help="Run 1 epoch without saving (smoke test)")
    args = p.parse_args()

    if not CROPS_DIR.exists():
        log.error("crops dir not found: %s", CROPS_DIR)
        sys.exit(1)

    train(args)
