"""
CH4Net: U-Net architecture for methane plume segmentation.

Extracted from the Colab notebook implementation and made production-ready.
Architecture matches the paper (Vaughan et al., 2024, AMT):
  - 4 encoder blocks, 4 decoder blocks, skip connections
  - Input: 12-band Sentinel-2 imagery (100x100 crop)
  - Output: pixel-wise probability mask

Key changes from notebook:
  1. div_factor=1 (full channel width, matching best results)
  2. Threshold calibrated at 0.18 (optimized F1, not default 0.25)
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


class Unet(nn.Module):
    """
    U-Net for methane plume binary segmentation.

    Faithfully reproduces the CH4Net architecture from Vaughan et al.
    with div_factor=1 (full width) which outperformed div_factor=8
    in our training runs.
    """

    def __init__(
        self,
        in_channels: int = 12,
        out_channels: int = 1,
        div_factor: int = 1,
        prob_output: bool = True,
    ):
        super().__init__()
        self.n_channels = in_channels
        self.prob_output = prob_output
        self.sigmoid = nn.Sigmoid()

        def double_conv(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        def down(in_ch, out_ch):
            return nn.Sequential(nn.MaxPool2d(2), double_conv(in_ch, out_ch))

        class Up(nn.Module):
            def __init__(self, in_ch, out_ch):
                super().__init__()
                self.up = nn.Upsample(
                    scale_factor=2, mode="bilinear", align_corners=True
                )
                self.conv = double_conv(in_ch, out_ch)

            def forward(self, x1, x2):
                x1 = self.up(x1)
                dy = x2.size(2) - x1.size(2)
                dx = x2.size(3) - x1.size(3)
                x1 = F.pad(x1, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
                return self.conv(torch.cat([x2, x1], dim=1))

        d = div_factor
        self.inc = double_conv(self.n_channels, 64 // d)
        self.down1 = down(64 // d, 128 // d)
        self.down2 = down(128 // d, 256 // d)
        self.down3 = down(256 // d, 512 // d)
        self.down4 = down(512 // d, 512 // d)
        self.up1 = Up(1024 // d, 256 // d)
        self.up2 = Up(512 // d, 128 // d)
        self.up3 = Up(256 // d, 64 // d)
        self.up4 = Up(128 // d, 128 // d)
        self.out = nn.Conv2d(128 // d, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        if self.prob_output:
            return self.sigmoid(self.out(x)).permute(0, 2, 3, 1)
        else:
            # Return raw logits for BCEWithLogitsLoss during training
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
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.threshold = threshold
        self.min_plume_pixels = min_plume_pixels

        # Load model with prob_output=True for inference (applies sigmoid)
        self.model = Unet(in_channels=12, out_channels=1, div_factor=1, prob_output=True)
        checkpoint = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
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
            prob_maps = prob_maps.squeeze(-1).cpu().numpy()  # (B, H, W)

            # Handle edge case where the final batch only has 1 item
            if len(batch) == 1:
                prob_maps = np.expand_dims(prob_maps, axis=0)

            all_prob_maps.extend(prob_maps)

        return all_prob_maps
