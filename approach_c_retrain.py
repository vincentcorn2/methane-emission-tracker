"""
Approach C: Retrain CH4Net with the correct architecture (div_factor=8, ~1.7M params)
using the official Vaughan et al. 2024 dataset (av555/ch4net from HuggingFace).

Fixes vs. current broken weights:
  1. div_factor=8  (~1.7M params vs broken 13.5M — matches paper architecture)
  2. Uses official train/val splits from the downloaded dataset
  3. Correct data format: s2 uint8 /255, label float64 binary mask
  4. BCE loss reduction='mean' (correct default)

Data format (confirmed from inspection):
  s2:    (217, 180, 12) uint8  range [8,255]  → divide by 255 → (12,217,180) float32
  label: (217, 180)    float64 binary 0/1    → (1,217,180) float32
  mbmp:  (217, 180, 4) uint8  RGBA visualisation — NOT used for training

Dataset size:
  train: 8,255 samples (includes positives + hard negatives)
  val:     255 samples
  test:  2,473 samples

Run with:
  conda activate methane
  python approach_c_retrain.py
  python approach_c_retrain.py --epochs 100 --batch_size 32   # more thorough

Outputs:
  weights/ch4net_div8_retrained.pth   ← best val-F1 checkpoint
"""

import argparse, os, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir",    default=os.path.expanduser("~/Downloads/ch4net_official"))
parser.add_argument("--out_weights", default="weights/ch4net_div8_retrained.pth")
parser.add_argument("--epochs",      type=int,   default=50)
parser.add_argument("--batch_size",  type=int,   default=16)
parser.add_argument("--lr",          type=float, default=1e-3)
parser.add_argument("--div_factor",  type=int,   default=8,
    help="Paper uses 8 (~1.7M params). Broken weights used 1 (~13.5M).")
parser.add_argument("--threshold",   type=float, default=0.5,
    help="Probability threshold for F1 during validation (calibrate after training).")
args = parser.parse_args()

torch.manual_seed(42)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Device: {device}")

# ── Fixed crop size ───────────────────────────────────────────────────────────
# The dataset has variable-size crops (e.g. 217×181, 227×166, etc.).
# We center-crop (or zero-pad if smaller) to a fixed size for batch collation.
# 160×160 fits within the minimum dimension observed (166) and is 10×16
# (cleanly divisible by 2^4=16 for the 4 U-Net downsampling stages).
CROP_H, CROP_W = 160, 160

def _center_crop_pad(arr, target_h, target_w):
    """Crop or zero-pad a (C, H, W) array to exactly (C, target_h, target_w)."""
    _, h, w = arr.shape
    # Height: crop or pad
    if h >= target_h:
        start = (h - target_h) // 2
        arr = arr[:, start:start + target_h, :]
    else:
        pad = target_h - h
        arr = np.pad(arr, [(0, 0), (pad // 2, pad - pad // 2), (0, 0)])
    # Width: crop or pad
    _, h2, w2 = arr.shape
    if w2 >= target_w:
        start = (w2 - target_w) // 2
        arr = arr[:, :, start:start + target_w]
    else:
        pad = target_w - w2
        arr = np.pad(arr, [(0, 0), (0, 0), (pad // 2, pad - pad // 2)])
    return arr

# ── Dataset ───────────────────────────────────────────────────────────────────
class CH4NetDataset(Dataset):
    """
    Loads (s2_image, label_mask) pairs from the official av555/ch4net dataset.

    Input:  s2    — variable-size (H,W,12) uint8  → normalised to (12,160,160) float32
    Target: label — variable-size (H,W)    float64 → (1,160,160) float32 binary

    NOTE: mbmp (H,W,4 uint8 RGBA) is NOT used for training — it is a
    visualisation artefact, not the ground-truth binary mask.
    Crops vary in size; _center_crop_pad standardises them to CROP_H×CROP_W.
    """
    def __init__(self, split_dir):
        self.s2_paths    = sorted(glob.glob(os.path.join(split_dir, "s2",    "*.npy")))
        self.label_paths = sorted(glob.glob(os.path.join(split_dir, "label", "*.npy")))
        assert len(self.s2_paths) == len(self.label_paths), (
            f"Mismatch: {len(self.s2_paths)} images vs {len(self.label_paths)} labels "
            f"in {split_dir}")
        assert len(self.s2_paths) > 0, f"No .npy files found in {split_dir}/s2/"
        # Count positives
        n_pos = sum(1 for p in self.label_paths
                    if np.load(p).max() > 0) if len(self.label_paths) < 500 else "?"
        print(f"  {os.path.basename(split_dir):6s}: {len(self.s2_paths):5d} samples  "
              f"({n_pos} positive)" if n_pos != "?" else
              f"  {os.path.basename(split_dir):6s}: {len(self.s2_paths):5d} samples")

    def __len__(self):
        return len(self.s2_paths)

    def __getitem__(self, idx):
        # Image: (H,W,12) uint8 → (12,H,W) float32 in [0,1]
        img = np.load(self.s2_paths[idx]).copy().astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)              # (H,W,C) → (C,H,W)
        img = np.clip(img, 0.0, 1.0)

        # Label: (H,W) float64 → (1,H,W) float32 binary
        lbl = np.load(self.label_paths[idx]).copy().astype(np.float32)
        lbl = (lbl > 0).astype(np.float32)
        lbl = lbl[np.newaxis, :, :]               # (H,W) → (1,H,W)

        # Standardise to fixed spatial size for batch collation
        img = _center_crop_pad(img, CROP_H, CROP_W)
        lbl = _center_crop_pad(lbl, CROP_H, CROP_W)

        return torch.from_numpy(img.copy()), torch.from_numpy(lbl.copy())


# ── U-Net (matches ch4net_model.py exactly, parametrised by div_factor) ──────
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(True),
        )
    def forward(self, x): return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))
    def forward(self, x): return self.net(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        dy, dx = x2.size(2)-x1.size(2), x2.size(3)-x1.size(3)
        x1 = F.pad(x1, [dx//2, dx-dx//2, dy//2, dy-dy//2])
        return self.conv(torch.cat([x2, x1], dim=1))

class Unet(nn.Module):
    def __init__(self, in_channels=12, div_factor=8):
        super().__init__()
        d = div_factor
        self.inc   = DoubleConv(in_channels, 64//d)
        self.down1 = Down(64//d,  128//d)
        self.down2 = Down(128//d, 256//d)
        self.down3 = Down(256//d, 512//d)
        self.down4 = Down(512//d, 512//d)
        self.up1   = Up(1024//d,  256//d)
        self.up2   = Up(512//d,   128//d)
        self.up3   = Up(256//d,    64//d)
        self.up4   = Up(128//d,   128//d)
        self.out   = nn.Conv2d(128//d, 1, kernel_size=1)
    def forward(self, x):
        x1=self.inc(x); x2=self.down1(x1); x3=self.down2(x2)
        x4=self.down3(x3); x5=self.down4(x4)
        x=self.up1(x5,x4); x=self.up2(x,x3); x=self.up3(x,x2); x=self.up4(x,x1)
        return self.out(x)   # raw logits

# ── Metrics ───────────────────────────────────────────────────────────────────
def pixel_f1(logits, targets, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    tgts  = (targets > 0.5).float()
    tp = (preds * tgts).sum()
    fp = (preds * (1-tgts)).sum()
    fn = ((1-preds) * tgts).sum()
    return (2*tp / (2*tp + fp + fn + 1e-8)).item()

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    data_dir = os.path.expanduser(args.data_dir)
    train_dir = os.path.join(data_dir, "train")
    val_dir   = os.path.join(data_dir, "val")

    for d in [train_dir, val_dir]:
        if not os.path.isdir(os.path.join(d, "s2")):
            raise FileNotFoundError(
                f"Expected {d}/s2/ — check --data_dir points to the downloaded dataset.\n"
                f"  Contents of {data_dir}: {os.listdir(data_dir)}")

    print("Loading datasets...")
    train_ds = CH4NetDataset(train_dir)
    val_ds   = CH4NetDataset(val_dir)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=False)

    model = Unet(in_channels=12, div_factor=args.div_factor).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: div_factor={args.div_factor}, params={n_params:,}")
    print(f"  Broken weights had div_factor=1 → {13_585_281:,} params (8× overfit)")
    print(f"  This model:      div_factor={args.div_factor} → {n_params:,} params\n")

    # Positive-class weight: dataset is imbalanced (7.3% positive images, even
    # fewer positive pixels within those images). Sample uniformly across the
    # full training set rather than just the first N files (which are mostly neg).
    print("Estimating class balance for loss weighting (sampling 300 uniform examples)...")
    import random
    sample_idx = random.sample(range(len(train_ds)), min(300, len(train_ds)))
    pos_px, total_px = 0, 0
    for i in sample_idx:
        _, lbl = train_ds[i]
        pos_px   += lbl.sum().item()
        total_px += lbl.numel()
    pos_frac = pos_px / total_px if total_px > 0 else 0.01
    # Clamp: don't let pos_weight go above 50 (unstable) or below 1
    pos_weight_val = min(50.0, max(1.0, (1 - pos_frac) / (pos_frac + 1e-8)))
    pos_weight = torch.tensor([pos_weight_val]).to(device)
    print(f"  Positive pixel fraction: {pos_frac:.5f}  →  pos_weight={pos_weight.item():.1f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")
    optimiser = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimiser, mode="max", factor=0.5, patience=5)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_weights)), exist_ok=True)
    best_val_f1 = 0.0

    print(f"\n{'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>10}  "
          f"{'Val F1@{:.2f}'.format(args.threshold):>12}  {'Saved?':>7}")
    print("─" * 60)

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimiser.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            optimiser.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_ds)

        # Validate
        model.eval()
        val_loss = val_f1 = 0.0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                logits = model(imgs)
                val_loss += criterion(logits, lbls).item() * imgs.size(0)
                val_f1   += pixel_f1(logits, lbls, args.threshold) * imgs.size(0)
        val_loss /= len(val_ds)
        val_f1   /= len(val_ds)

        scheduler.step(val_f1)
        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), args.out_weights)

        print(f"{epoch:>6}  {train_loss:>12.6f}  {val_loss:>10.6f}  "
              f"{val_f1:>12.4f}  {'  ✓ saved' if is_best else ''}")

    print(f"\n{'='*60}")
    print(f"Training complete. Best val F1: {best_val_f1:.4f}")
    print(f"Weights: {args.out_weights}")
    print(f"""
Next steps:
  1. Update src/detection/ch4net_model.py  →  change div_factor=1 to div_factor={args.div_factor}
  2. Update scripts/live_pipeline.py       →  weights path to '{args.out_weights}'
  3. Re-run approach_a_centered_crops.py to check if spatial specificity improves
  4. Re-run approach_b_rethreshold.py to check if emission/clean ratio flips > 1.0
""")


if __name__ == "__main__":
    main()
