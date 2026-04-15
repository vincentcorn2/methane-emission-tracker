"""
CH4Net: U-Net architecture for methane plume segmentation.

Extracted from the Colab notebook implementation and made production-ready.
Architecture matches the paper (Vaughan et al., 2024, AMT):
  - 4 encoder blocks, 4 decoder blocks, skip connections
  - Input: 12-band Sentinel-2 imagery (160x160 crop)
  - Output: pixel-wise probability mask

Key notes:
  1. div_factor=8 (paper architecture, ~214K params) — retrained on official
     Vaughan et al. dataset (av555/ch4net on HuggingFace, 8255 train samples)
  2. div_factor=1 (13.5M params) was the original broken config — massively
     overfit on 925 samples, acted as a terrain detector not a methane detector
  3. prob_output=False during training (BCEWithLogitsLoss expects logits)
  4. Separate inference method that applies sigmoid + threshold
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class DetectionResult:
    """Structured output from a single detection run.

    This is the internal representation before it hits the API schema.
    Contains everything needed for downstream quantification and
    entity resolution.
    """
    has_plume: bool
    confidence: float              # max probability in detected region
    plume_mask: np.ndarray         # binary mask at detection threshold
    probability_map: np.ndarray    # raw sigmoid output
    plume_area_pixels: int
    centroid_row: Optional[int] = None
    centroid_col: Optional[int] = None


# ── Sub-modules (must match approach_c_retrain.py exactly so weights load) ────

class _DoubleConv(nn.Module):
    """Two conv-BN-ReLU blocks. Saved as .net in the state_dict."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(True),
        )
    def forward(self, x): return self.net(x)


class _Down(nn.Module):
    """MaxPool2d + DoubleConv. Saved as .net in the state_dict."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), _DoubleConv(in_ch, out_ch))
    def forward(self, x): return self.net(x)


class _Up(nn.Module):
    """Bilinear upsample + skip-concat + DoubleConv."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = _DoubleConv(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        dy, dx = x2.size(2) - x1.size(2), x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class Unet(nn.Module):
    """
    U-Net for methane plume binary segmentation.

    Architecture exactly matches approach_c_retrain.py so that saved weights
    load without key remapping. Uses named sub-modules (_DoubleConv, _Down, _Up)
    which produce state_dict keys like 'inc.net.0.weight', 'down1.net.1.net.0.weight'.

    div_factor=8 → ~214K params (paper architecture, Vaughan et al. 2024 AMT).
    prob_output=True applies sigmoid for inference; False returns raw logits for training.
    """

    def __init__(
        self,
        in_channels: int = 12,
        out_channels: int = 1,
        div_factor: int = 8,
        prob_output: bool = True,
    ):
        super().__init__()
        self.prob_output = prob_output
        self.sigmoid = nn.Sigmoid()
        d = div_factor
        self.inc   = _DoubleConv(in_channels, 64 // d)
        self.down1 = _Down(64 // d,  128 // d)
        self.down2 = _Down(128 // d, 256 // d)
        self.down3 = _Down(256 // d, 512 // d)
        self.down4 = _Down(512 // d, 512 // d)
        self.up1   = _Up(1024 // d,  256 // d)
        self.up2   = _Up(512 // d,   128 // d)
        self.up3   = _Up(256 // d,    64 // d)
        self.up4   = _Up(128 // d,   128 // d)
        self.out   = nn.Conv2d(128 // d, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x  = self.up1(x5, x4)
        x  = self.up2(x,  x3)
        x  = self.up3(x,  x2)
        x  = self.up4(x,  x1)
        if self.prob_output:
            return self.sigmoid(self.out(x)).permute(0, 2, 3, 1)
        else:
            return self.out(x).permute(0, 2, 3, 1)


class CH4NetDetector:
    """
    High-level inference wrapper around the trained U-Net.

    Handles model loading, preprocessing, threshold application,
    and structured output generation.

    Usage:
        detector = CH4NetDetector("weights/best_model.pth")
        result = detector.detect(sentinel2_array)  # shape: (H, W, 12)
        if result.has_plume:
            print(f"Plume at pixel ({result.centroid_row}, {result.centroid_col})")
            print(f"Confidence: {result.confidence:.2f}")
    """

    def __init__(
        self,
        weights_path: str,
        threshold: float = 0.18,
        min_plume_pixels: int = 115,
        device: str = "auto",
    ):
        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.threshold = threshold
        self.min_plume_pixels = min_plume_pixels

        # Load model with prob_output=True for inference (applies sigmoid)
        # Note: approach_c_retrain.py saves raw state_dict (not wrapped in a dict)
        state_dict = torch.load(weights_path, map_location=self.device)
        # Handle both raw state_dict and wrapped {"model_state_dict": ...} formats
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        # Auto-detect div_factor from out.weight shape: out=Conv2d(128//d, 1, 1)
        # so out.weight has shape [1, 128//d, 1, 1] → d = 128 // in_channels
        _out_key = "out.weight"
        _div_factor = 8  # default (paper architecture, ~214K params)
        if _out_key in state_dict:
            _in_ch = state_dict[_out_key].shape[1]
            _div_factor = max(1, 128 // _in_ch)

        self.model = Unet(in_channels=12, out_channels=1, div_factor=_div_factor, prob_output=True)

        # Remap flat-Sequential keys to .net-wrapped keys unconditionally (idempotent).
        # Old format: inc.0.weight, down1.1.0.weight, up1.conv.0.weight
        # New format: inc.net.0.weight, down1.net.1.net.0.weight, up1.conv.net.0.weight
        import re
        remapped = {}
        for k, v in state_dict.items():
            k = re.sub(r'^inc\.(\d)', r'inc.net.\1', k)
            k = re.sub(r'^(down\d)\.1\.(\d)', r'\1.net.1.net.\2', k)
            k = re.sub(r'^(up\d\.conv)\.(\d)', r'\1.net.\2', k)
            remapped[k] = v
        self.model.load_state_dict(remapped)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def detect(self, s2_array: np.ndarray) -> DetectionResult:
        """
        Run detection on a single Sentinel-2 image patch.

        Args:
            s2_array: numpy array of shape (H, W, 12) — 12 Sentinel-2 bands,
                      values in [0, 255] range (raw .npy format from dataset)

        Returns:
            DetectionResult with plume mask, confidence, centroid, etc.
        """
        # Preprocess: normalize to [0, 1], reshape to (1, 12, H, W)
        tensor = (
            torch.from_numpy(s2_array)
            .float()
            .permute(2, 0, 1)
            .unsqueeze(0)
            / 255.0
        )
        tensor = tensor.to(self.device)

        # Inference
        prob_map = self.model(tensor)  # (1, H, W, 1)
        prob_map = prob_map.squeeze().cpu().numpy()  # (H, W)

        # Apply calibrated threshold
        binary_mask = (prob_map >= self.threshold).astype(np.uint8)

        # Check for contiguous plume region above minimum size
        plume_pixels = int(binary_mask.sum())
        has_plume = plume_pixels >= self.min_plume_pixels

        # Calculate centroid if plume detected
        centroid_row, centroid_col = None, None
        confidence = 0.0
        if has_plume:
            rows, cols = np.where(binary_mask > 0)
            centroid_row = int(rows.mean())
            centroid_col = int(cols.mean())
            confidence = float(prob_map[binary_mask > 0].max())

        return DetectionResult(
            has_plume=has_plume,
            confidence=confidence,
            plume_mask=binary_mask,
            probability_map=prob_map,
            plume_area_pixels=plume_pixels,
            centroid_row=centroid_row,
            centroid_col=centroid_col,
        )

    @torch.no_grad()
    def detect_batch(self, s2_arrays: list[np.ndarray], batch_size: int = 64) -> list[np.ndarray]:
        """
        Run detection on a batch of Sentinel-2 image patches to accelerate CPU/GPU inference.

        Args:
            s2_arrays: List of numpy arrays of shape (H, W, 12)
            batch_size: Number of patches to process simultaneously

        Returns:
            List of 2D probability maps (H, W)
        """
        all_prob_maps = []

        # Process in chunks of batch_size to maximize core usage without OOM
        for i in range(0, len(s2_arrays), batch_size):
            batch = s2_arrays[i:i + batch_size]

            # Stack into a single numpy array: (B, H, W, 12)
            batch_np = np.stack(batch)

            # Convert to tensor: (B, 12, H, W) and normalize to [0, 1]
            tensor = (
                torch.from_numpy(batch_np)
                .float()
                .permute(0, 3, 1, 2)
                / 255.0
            ).to(self.device)

            # Batched Inference
            prob_maps = self.model(tensor)  # output shape: (B, H, W, 1)
            # squeeze(-1): (B, H, W, 1) → (B, H, W)
            # reshape(-1, H, W): ensures correct shape even when B=1
            B, H, W, _ = prob_maps.shape
            prob_maps = prob_maps.squeeze(-1).reshape(B, H, W).cpu().numpy()

            all_prob_maps.extend(prob_maps)

        return all_prob_maps
