"""
debug_batch_inference.py
────────────────────────
Isolates the detect_batch MPS bug by comparing single-tile vs batched
inference on the same patches.

Run with:
    conda activate methane && python debug_batch_inference.py
"""
import sys
import numpy as np
import torch
sys.path.insert(0, ".")
from src.detection.ch4net_model import CH4NetDetector

WEIGHTS   = "weights/best_model.pth"
NPY_PATH  = "data/npy_cache/S2B_MSIL1C_20240918T103619_N0511_R008_T31UGS_20240918T142046.npy"
TILE_SIZE = 100

print(f"Loading weights: {WEIGHTS}")
detector = CH4NetDetector(WEIGHTS)
print(f"Device: {detector.device}")

print(f"\nLoading npy (this takes ~30s)...")
scene = np.load(NPY_PATH, mmap_mode="r")
print(f"Scene shape: {scene.shape}")

# Extract a few tiles from the Weisweiler area (approx centre of tile)
# Weisweiler is near row ~5500, col ~300 in T31UGS
patches = []
for r in range(5400, 5800, 100):
    for c in range(200, 600, 100):
        patch = np.array(scene[r:r+TILE_SIZE, c:c+TILE_SIZE, :])
        if patch.shape == (TILE_SIZE, TILE_SIZE, 12):
            patches.append(patch)

print(f"\nExtracted {len(patches)} patches near Weisweiler")

# ── Single-tile inference ──────────────────────────────────────────────────
print("\n[A] Single-tile detect() loop:")
single_preds = []
for p in patches[:8]:
    result = detector.detect(p)
    single_preds.append(result.probability_map)
    print(f"  patch mean={result.probability_map.mean():.6f}  max={result.probability_map.max():.6f}")

# ── Batch inference, batch_size=1 ─────────────────────────────────────────
print("\n[B] detect_batch() with batch_size=1:")
batch1_preds = detector.detect_batch(patches[:8], batch_size=1)
for i, pred in enumerate(batch1_preds):
    print(f"  patch mean={pred.mean():.6f}  max={pred.max():.6f}  shape={pred.shape}")

# ── Batch inference, batch_size=8 ─────────────────────────────────────────
print("\n[C] detect_batch() with batch_size=8:")
batch8_preds = detector.detect_batch(patches[:8], batch_size=8)
for i, pred in enumerate(batch8_preds):
    print(f"  patch mean={pred.mean():.6f}  max={pred.max():.6f}  shape={pred.shape}")

# ── Compare A vs B vs C ───────────────────────────────────────────────────
print("\n[COMPARISON] max abs diff:")
for i in range(min(8, len(single_preds))):
    da = np.abs(single_preds[i] - batch1_preds[i]).max()
    db = np.abs(single_preds[i] - batch8_preds[i]).max()
    print(f"  patch {i}: single vs batch1={da:.6f}  single vs batch8={db:.6f}")

# ── Raw model output check ─────────────────────────────────────────────────
print("\n[D] Raw model output shape check (batch_size=8):")
detector.model.eval()
with torch.no_grad():
    batch_np = np.stack(patches[:8])
    tensor = (torch.from_numpy(batch_np).float().permute(0, 3, 1, 2) / 255.0).to(detector.device)
    raw = detector.model(tensor)
    print(f"  model output shape:        {tuple(raw.shape)}")
    print(f"  model output dtype:        {raw.dtype}")
    print(f"  model output mean:         {raw.mean().item():.6f}")
    print(f"  model output max:          {raw.max().item():.6f}")
    squeezed = raw.squeeze(-1)
    print(f"  after squeeze(-1) shape:   {tuple(squeezed.shape)}")
    contiguous = raw.squeeze(-1).contiguous()
    print(f"  contiguous mean:           {contiguous.mean().item():.6f}")
    on_cpu = contiguous.cpu().numpy()
    print(f"  .cpu().numpy() mean:       {on_cpu.mean():.6f}")
    print(f"  .cpu().numpy() max:        {on_cpu.max():.6f}")
