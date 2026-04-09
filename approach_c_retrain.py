#!/usr/bin/env python3
"""
approach_c_retrain.py — Fine-tune CH4Net v2 on European TROPOMI-confirmed sites.

Training data
-------------
  Positives : data/crops/positive/  (TROPOMI-confirmed, label_value=1)
  Negatives : data/crops/negative/  (JRC/survey confirmed non-emitters, label_value=0)

Each crop is a 200x200x12 uint8 .npy file (Sentinel-2 L1C, all 12 bands).
Non-200x200 crops (edge tiles) are silently skipped.

Model
-----
  Architecture : CH4Net U-Net (div_factor auto-detected from checkpoint)
  Starting weights : weights/best_model.pth  (with flat->net key remap)
  Output weights   : weights/european_model.pth

Target masks
------------
  Positive crop : soft Gaussian disk centred on the site pixel (100,100 in
                  200x200 space, shifted by random crop offset).
  Negative crop : all-zero mask.

Loss : BCEWithLogitsLoss (model runs with prob_output=False -> raw logits).
       Positive pixels are up-weighted by POS_WEIGHT to handle sparse plumes.

Usage
-----
  python approach_c_retrain.py                      # default settings
  python approach_c_retrain.py --epochs 100 --lr 5e-5
  python approach_c_retrain.py --dry-run            # 1 epoch, no save
  python approach_c_retrain.py --weights-in weights/european_model.pth  # resume
"""

import os
import sys
import json
import glob
import logging
import argparse
import re
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    print("ERROR: PyTorch not found. Run: conda activate methane")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent))
from src.detection.ch4net_model import Unet

# ── Config ─────────────────────────────────────────────────────────────────────
CROP_OUT    = 160       # random crop size extracted from 200x200 input
BATCH_SIZE  = 4
EPOCHS      = 60
LR          = 1e-4
LR_STEP     = 10        # decay LR every N epochs
LR_DECAY    = 0.95
PLUME_SIGMA = 30        # Gaussian sigma for positive target mask (pixels in crop space)
POS_WEIGHT  = 5.0       # BCEWithLogitsLoss pos_weight — upweights plume pixels
PATIENCE    = 15        # early stopping: epochs without val improvement
CROPS_DIR   = Path("data/crops")
WEIGHTS_IN  = "weights/best_model.pth"
WEIGHTS_OUT = "weights/european_model.pth"
LOG_FILE    = "results_analysis/retrain.log"

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

def _gaussian_mask(H, W, cy, cx, sigma):
    """Soft Gaussian disk, peak 1.0 at (cy, cx)."""
    y, x = np.ogrid[:H, :W]
    dist2 = (y - cy) ** 2 + (x - cx) ** 2
    return np.exp(-dist2 / (2 * sigma ** 2)).astype(np.float32)


class MethaneDataset(Dataset):
    """
    Yields (image, mask, label_int) tuples.

    image : FloatTensor (12, CROP_OUT, CROP_OUT)  normalised [0,1]
    mask  : FloatTensor ( 1, CROP_OUT, CROP_OUT)  target segmentation mask
    label : int  0=negative  1=positive
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
                log.debug("Skipping edge crop %s  shape=%s", npy_path.name, arr.shape)
                continue
            self.samples.append({
                "npy":   str(npy_path),
                "label": int(meta["label_value"]),
                "site":  meta.get("site", "?"),
            })

        n_pos = sum(s["label"] for s in self.samples)
        n_neg = len(self.samples) - n_pos
        log.info("Dataset split=%-5s  total=%d  pos=%d  neg=%d",
                 split, len(self.samples), n_pos, n_neg)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        arr = np.load(s["npy"]).astype(np.float32) / 255.0   # (200, 200, 12)

        # Random 160x160 crop (40px pad on each side)
        pad = 200 - CROP_OUT
        if self.augment:
            r0 = int(np.random.randint(0, pad + 1))
            c0 = int(np.random.randint(0, pad + 1))
        else:
            r0 = c0 = pad // 2   # deterministic centre crop for val/test

        crop = arr[r0:r0 + CROP_OUT, c0:c0 + CROP_OUT, :]   # (160, 160, 12)

        # Target mask: site centre is always at pixel (100, 100) in 200x200 space
        if s["label"] == 1:
            cy = max(0, min(CROP_OUT - 1, 100 - r0))
            cx = max(0, min(CROP_OUT - 1, 100 - c0))
            mask = _gaussian_mask(CROP_OUT, CROP_OUT, cy, cx, PLUME_SIGMA)
        else:
            mask = np.zeros((CROP_OUT, CROP_OUT), dtype=np.float32)

        # Augmentation (train only)
        if self.augment:
            if np.random.rand() > 0.5:
                crop = crop[:, ::-1, :].copy()
                mask = mask[:, ::-1].copy()
            if np.random.rand() > 0.5:
                crop = crop[::-1, :, :].copy()
                mask = mask[::-1, :].copy()
            if np.random.rand() > 0.5:
                crop = np.transpose(crop, (1, 0, 2)).copy()
                mask = mask.T.copy()

        img = torch.from_numpy(crop.transpose(2, 0, 1))  # (12, H, W)
        msk = torch.from_numpy(mask).unsqueeze(0)         # ( 1, H, W)
        return img, msk, s["label"]


# ── Model ──────────────────────────────────────────────────────────────────────

def load_model(weights_path, device, prob_output=False):
    """
    Load CH4Net with automatic div_factor detection and key remapping.
    prob_output=False -> raw logits (use for training with BCEWithLogitsLoss).
    """
    sd = torch.load(weights_path, map_location=device)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]

    # Detect div_factor from out.weight shape [1, 128//d, 1, 1]
    div_factor = 8
    if "out.weight" in sd:
        in_ch      = sd["out.weight"].shape[1]
        div_factor = max(1, 128 // in_ch)
    log.info("Checkpoint div_factor=%d  (out_channels=%d)", div_factor, 128 // div_factor)

    model = Unet(in_channels=12, out_channels=1,
                 div_factor=div_factor, prob_output=prob_output)

    # Remap flat-Sequential -> .net-wrapped keys (idempotent for already-new keys)
    remapped = {}
    for k, v in sd.items():
        k = re.sub(r'^inc\.(\d)',           r'inc.net.\1',        k)
        k = re.sub(r'^(down\d)\.1\.(\d)',   r'\1.net.1.net.\2',   k)
        k = re.sub(r'^(up\d\.conv)\.(\d)',  r'\1.net.\2',         k)
        remapped[k] = v

    model.load_state_dict(remapped, strict=True)
    model.to(device)
    return model


# ── Training ───────────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimiser, device, train=True):
    model.train() if train else model.eval()
    losses, correct, total = [], 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, masks, labels in loader:
            imgs   = imgs.to(device)
            masks  = masks.to(device)
            labels = torch.tensor(labels, dtype=torch.long, device=device)

            logits = model(imgs).permute(0, 3, 1, 2)  # (B,1,H,W)
            loss   = criterion(logits, masks)

            if train:
                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimiser.step()

            losses.append(loss.item())

            # Image-level accuracy: compare mean prob in centre 40x40 vs 0.5
            probs  = torch.sigmoid(logits)
            c      = CROP_OUT // 2
            centre = probs[:, :, c-20:c+20, c-20:c+20].mean(dim=(1, 2, 3))
            preds  = (centre > 0.5).long()
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

    # Data
    train_ds = MethaneDataset(CROPS_DIR, split="train", augment=True)
    val_ds   = MethaneDataset(CROPS_DIR, split="val",   augment=False)

    if len(train_ds) == 0:
        log.error("No training crops found in %s. Run extract_training_crops.py first.", CROPS_DIR)
        sys.exit(1)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0, drop_last=False)

    # Model
    log.info("Loading base weights: %s", args.weights_in)
    model = load_model(args.weights_in, device, prob_output=False)

    # Loss and optimiser
    pos_w     = torch.tensor([POS_WEIGHT], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimiser, step_size=LR_STEP, gamma=LR_DECAY)

    epochs         = 1 if args.dry_run else args.epochs
    best_val_loss  = float("inf")
    patience_count = 0
    history        = []

    log.info("=" * 62)
    log.info("  European fine-tuning — %d train / %d val  |  %d epochs",
             len(train_ds), len(val_ds), epochs)
    log.info("=" * 62)

    for epoch in range(1, epochs + 1):
        t_loss, t_acc = run_epoch(model, train_loader, criterion, optimiser, device, train=True)
        v_loss, v_acc = run_epoch(model, val_loader,   criterion, optimiser, device, train=False)
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
        hist_path = "results_analysis/retrain_history.json"
        with open(hist_path, "w") as f:
            json.dump({"config": vars(args), "history": history}, f, indent=2)
        log.info("History: %s", hist_path)


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Fine-tune CH4Net on European training data")
    p.add_argument("--weights-in",  default=WEIGHTS_IN)
    p.add_argument("--weights-out", default=WEIGHTS_OUT)
    p.add_argument("--epochs",      type=int,   default=EPOCHS)
    p.add_argument("--lr",          type=float, default=LR)
    p.add_argument("--batch-size",  type=int,   default=BATCH_SIZE)
    p.add_argument("--dry-run",     action="store_true",
                   help="Run 1 epoch without saving weights (smoke test)")
    args = p.parse_args()

    if not CROPS_DIR.exists():
        log.error("No crops directory found: %s", CROPS_DIR)
        sys.exit(1)

    train(args)
