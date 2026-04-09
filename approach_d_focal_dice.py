"""
Approach D: Retrain CH4Net with Focal+Dice composite loss + random augmentation.

Changes vs. approach_c_retrain.py:
  1. Loss: BCE+pos_weight → Focal+Dice composite (alpha=0.5 each, focal gamma=2.0)
     - DiceLoss directly optimises the F1-like overlap metric
     - FocalLoss down-weights easy negatives (the vast sea of non-plume pixels),
       letting the model focus gradient on hard positives
  2. Augmentation: center-crop-only → random crop + random H/V flip (train split only)
     - Paper uses random 160×160 crops from variable-size images
     - Flips add geometric invariance cheaply; methane plumes have no canonical orientation
  3. Everything else unchanged: architecture, div_factor=8, optimizer, scheduler, val loop

Recommended run (Colab T4, ~2h for 100 epochs):
  python approach_d_focal_dice.py --epochs 100 --batch_size 16

Compare against approach_c baseline:
  Best val F1 (approach_c): 0.2733
  Target: >0.30 with same 8,255 training samples
"""

import argparse, os, glob, random
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
parser.add_argument("--out_weights", default="weights/ch4net_div8_focal_dice.pth")
parser.add_argument("--epochs",      type=int,   default=50)
parser.add_argument("--batch_size",  type=int,   default=16)
parser.add_argument("--lr",          type=float, default=1e-3)
parser.add_argument("--div_factor",  type=int,   default=8)
parser.add_argument("--threshold",   type=float, default=0.5)
parser.add_argument("--dice_w",      type=float, default=0.5,  help="Weight for Dice term")
parser.add_argument("--focal_w",     type=float, default=0.5,  help="Weight for Focal term")
parser.add_argument("--focal_gamma", type=float, default=2.0,  help="Focal loss gamma")
parser.add_argument("--focal_alpha", type=float, default=0.25, help="Focal loss alpha (pos class)")
args = parser.parse_args()

torch.manual_seed(42)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Device: {device}")

CROP_H, CROP_W = 160, 160

# ── Spatial transforms (applied identically to image + label) ─────────────────
def random_crop(img, lbl, h, w):
    """Random crop from (C, H, W) arrays. Falls back to center crop if smaller."""
    _, ih, iw = img.shape
    if ih < h or iw < w:
        # Zero-pad to target size then return
        ph = max(0, h - ih); pw = max(0, w - iw)
        img = np.pad(img, [(0,0),(ph//2, ph-ph//2),(pw//2, pw-pw//2)])
        lbl = np.pad(lbl, [(0,0),(ph//2, ph-ph//2),(pw//2, pw-pw//2)])
        return img, lbl
    top  = random.randint(0, ih - h)
    left = random.randint(0, iw - w)
    return img[:, top:top+h, left:left+w], lbl[:, top:top+h, left:left+w]

def center_crop_pad(arr, target_h, target_w):
    """Center-crop or zero-pad (C, H, W) array to exactly (C, target_h, target_w)."""
    _, h, w = arr.shape
    if h >= target_h:
        s = (h - target_h) // 2
        arr = arr[:, s:s+target_h, :]
    else:
        p = target_h - h
        arr = np.pad(arr, [(0,0),(p//2, p-p//2),(0,0)])
    _, h2, w2 = arr.shape
    if w2 >= target_w:
        s = (w2 - target_w) // 2
        arr = arr[:, :, s:s+target_w]
    else:
        p = target_w - w2
        arr = np.pad(arr, [(0,0),(0,0),(p//2, p-p//2)])
    return arr

# ── Dataset ───────────────────────────────────────────────────────────────────
class CH4NetDataset(Dataset):
    def __init__(self, split_dir, augment=False):
        self.s2_paths    = sorted(glob.glob(os.path.join(split_dir, "s2",    "*.npy")))
        self.label_paths = sorted(glob.glob(os.path.join(split_dir, "label", "*.npy")))
        self.augment     = augment
        assert len(self.s2_paths) == len(self.label_paths)
        assert len(self.s2_paths) > 0, f"No .npy files in {split_dir}/s2/"
        n_pos = sum(1 for p in self.label_paths
                    if np.load(p).max() > 0) if len(self.label_paths) < 500 else "?"
        aug_str = " [augmented]" if augment else ""
        print(f"  {os.path.basename(split_dir):6s}: {len(self.s2_paths):5d} samples"
              + (f"  ({n_pos} positive)" if n_pos != "?" else "") + aug_str)

    def __len__(self):
        return len(self.s2_paths)

    def __getitem__(self, idx):
        img = np.load(self.s2_paths[idx]).copy().astype(np.float32) / 255.0
        img = np.clip(img.transpose(2, 0, 1), 0.0, 1.0)   # (C,H,W)

        lbl = np.load(self.label_paths[idx]).copy().astype(np.float32)
        lbl = (lbl > 0).astype(np.float32)[np.newaxis]     # (1,H,W)

        if self.augment:
            # Random 160×160 crop (same region for image and label)
            img, lbl = random_crop(img, lbl, CROP_H, CROP_W)
            # Random horizontal flip
            if random.random() > 0.5:
                img = img[:, :, ::-1].copy()
                lbl = lbl[:, :, ::-1].copy()
            # Random vertical flip
            if random.random() > 0.5:
                img = img[:, ::-1, :].copy()
                lbl = lbl[:, ::-1, :].copy()
        else:
            img = center_crop_pad(img, CROP_H, CROP_W)
            lbl = center_crop_pad(lbl, CROP_H, CROP_W)

        return torch.from_numpy(img.copy()), torch.from_numpy(lbl.copy())


# ── Loss functions ─────────────────────────────────────────────────────────────
class DiceLoss(nn.Module):
    """Soft Dice loss. Smooth=1 prevents divide-by-zero on empty label maps."""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs   = torch.sigmoid(logits)
        targets = (targets > 0.5).float()
        # Flatten spatial dims, compute per-sample then average
        probs   = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        intersection = (probs * targets).sum(dim=1)
        dice = (2. * intersection + self.smooth) / (
               probs.sum(dim=1) + targets.sum(dim=1) + self.smooth)
        return 1. - dice.mean()


class FocalLoss(nn.Module):
    """
    Sigmoid focal loss (Lin et al. 2017).
    gamma=2.0: standard; down-weights easy negatives exponentially
    alpha=0.25: up-weights the rare positive class
    """
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        targets = (targets > 0.5).float()
        bce     = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs   = torch.sigmoid(logits)
        pt      = targets * probs + (1. - targets) * (1. - probs)        # p_t
        alpha_t = targets * self.alpha + (1. - targets) * (1. - self.alpha)
        loss    = alpha_t * (1. - pt) ** self.gamma * bce
        return loss.mean()


class FocalDiceLoss(nn.Module):
    def __init__(self, dice_w=0.5, focal_w=0.5, focal_gamma=2.0, focal_alpha=0.25):
        super().__init__()
        self.dice_w  = dice_w
        self.focal_w = focal_w
        self.dice    = DiceLoss()
        self.focal   = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)

    def forward(self, logits, targets):
        return self.dice_w * self.dice(logits, targets) + \
               self.focal_w * self.focal(logits, targets)


# ── U-Net (identical to approach_c / ch4net_model.py) ────────────────────────
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
    data_dir  = os.path.expanduser(args.data_dir)
    train_dir = os.path.join(data_dir, "train")
    val_dir   = os.path.join(data_dir, "val")

    for d in [train_dir, val_dir]:
        if not os.path.isdir(os.path.join(d, "s2")):
            raise FileNotFoundError(
                f"Expected {d}/s2/ — check --data_dir. Contents: {os.listdir(data_dir)}")

    print("Loading datasets...")
    train_ds = CH4NetDataset(train_dir, augment=True)   # random crop + flips
    val_ds   = CH4NetDataset(val_dir,   augment=False)  # deterministic center crop

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=False)

    model    = Unet(in_channels=12, div_factor=args.div_factor).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: div_factor={args.div_factor}, params={n_params:,}")

    criterion = FocalDiceLoss(
        dice_w      = args.dice_w,
        focal_w     = args.focal_w,
        focal_gamma = args.focal_gamma,
        focal_alpha = args.focal_alpha,
    )
    print(f"Loss: Focal(gamma={args.focal_gamma}, alpha={args.focal_alpha}) × {args.focal_w}"
          f" + Dice × {args.dice_w}")

    optimiser = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimiser, mode="max", factor=0.5, patience=5)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_weights)), exist_ok=True)
    best_val_f1 = 0.0

    print(f"\n{'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>10}  "
          f"{'Val F1@{:.2f}'.format(args.threshold):>12}  {'Saved?':>7}")
    print("─" * 60)

    for epoch in range(1, args.epochs + 1):
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

        model.eval()
        val_loss = val_f1 = 0.0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                logits  = model(imgs)
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
Baseline (approach_c, BCE+pos_weight, center-crop): 0.2733
This run (approach_d, Focal+Dice, random crop+flip): {best_val_f1:.4f}

Next steps if F1 > 0.30:
  1. Update src/detection/ch4net_model.py weights path to '{args.out_weights}'
  2. Re-run overnight_validation.py to compare Approach A/B results
  3. Consider adding NDMI channel (B11-B12)/(B11+B12) as channel 13 next
""")


if __name__ == "__main__":
    main()
