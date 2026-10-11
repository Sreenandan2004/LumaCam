# 🟢 Sreenandan (Me) — Task Handoff for AI Agent

> **Role**: Lead Integrator — GPU Owner & Core Architecture
> **Branch**: `feature/train-pipeline`
> **Hardware**: NVIDIA GTX 1650 (4GB) + Kaggle Free P100 GPU
> **Git Remote**: `https://github.com/Sreenandan2004/LumaCam.git`

---

## Project Context

**LumaCam** is a real-time low-light image enhancement mobile camera app. The core neural network is an **Exposure-Conditioned Attention U-Net GAN** that takes packed 4-channel Bayer raw sensor data and a user-controlled exposure scalar, and outputs an enhanced full-resolution sRGB image.

### Repository Location
```
D:\Development\LumaCam\
```

### Already Implemented (DO NOT modify these files)

These modules are complete and tested. Your new code depends on them:

1. **`model/networks/generator.py`** — `ExposureConditionedAttentionUNet`
   - Input: `(B, 4, H/2, W/2)` packed Bayer + `(B, 1)` exposure alpha
   - Output: `(B, 3, H, W)` sRGB in `[0, 1]`, optional attention maps
   - 4 encoder stages (32→64→128→256 channels), bottleneck (512ch) with FiLM, 4 decoder stages with Attention Gates, PixelShuffle(2) head
   - `forward(x, alpha_cond=None, return_attention_maps=False) -> Tuple[Tensor, Optional[List[Tensor]]]`

2. **`model/networks/discriminator.py`** — `PatchGANDiscriminator`
   - Input: `condition (B, C, H, W)` + `target (B, 3, H, W)` (auto-interpolates if sizes differ)
   - Output: `(B, 1, N, N)` patch logits
   - Default `in_channels=6` (3 condition + 3 target), but supports 7 (4 Bayer + 3 RGB)

3. **`model/networks/attention_gate.py`** — `AttentionGate`
   - `forward(x, g) -> (gated_x, alpha)` where alpha ∈ [0, 1]

4. **`model/networks/conditioning.py`** — `FiLMLayer`
   - `forward(feature_map, cond_vector) -> modulated_feature_map`
   - Zero-initialized for identity pass-through

5. **`model/datasets/raw_processing.py`**
   - `pack_raw_bayer_np(raw_image, black_level=512.0, white_level=16383.0)` → `(H/2, W/2, 4)`
   - `pack_raw_bayer_torch(raw_tensor)` → `(B, 4, H/2, W/2)`
   - `apply_exposure_amplification(packed_raw, ratio)` → amplified tensor

---

## Task 1: Composite Loss Function (P0 — Day 1-2)

### File to Create
```
model/losses/__init__.py
model/losses/composite_loss.py
```

### What to Build

Three loss classes and one composite wrapper:

#### 1. `MSSSIMLoss` — Multi-Scale Structural Similarity Loss

```python
class MSSSIMLoss(nn.Module):
    """
    Differentiable Multi-Scale SSIM loss.
    
    MS-SSIM computes structural similarity at 5 different scales (resolutions)
    by iteratively downsampling the image. This captures both fine-grained texture
    similarity and global luminance/contrast agreement.
    
    Loss = 1 - MS_SSIM(predicted, target)
    
    The 5-scale weights are from Wang et al. (2003):
        [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    """
    def __init__(self, data_range: float = 1.0, n_scales: int = 5, 
                 win_size: int = 11, win_sigma: float = 1.5):
        ...
    
    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predicted: (B, 3, H, W) generated sRGB image in [0, 1]
            target: (B, 3, H, W) ground truth sRGB image in [0, 1]
        Returns:
            Scalar loss = 1 - MS_SSIM(predicted, target)
        """
```

**Implementation approach**: Use Gaussian window convolution via `F.conv2d` with `groups=C`. At each scale, compute luminance (l), contrast (c), structure (s) components of SSIM. Multiply contrast*structure across scales 1..M-1, then multiply luminance*contrast*structure at scale M. Take weighted product across scales.

Alternatively, use `torchmetrics.image.MultiScaleStructuralSimilarityIndexMeasure` internally and wrap it as a loss (1 - metric_value).

#### 2. `AdversarialLoss` — Conditional GAN Loss

```python
class AdversarialLoss(nn.Module):
    """
    Standard conditional GAN losses using BCEWithLogitsLoss.
    The PatchGAN discriminator outputs (B, 1, N, N) logits — one real/fake
    probability per 70x70 receptive field patch.
    """
    def __init__(self):
        self.criterion = nn.BCEWithLogitsLoss()
    
    def generator_loss(self, fake_pred: torch.Tensor) -> torch.Tensor:
        """Generator wants discriminator to think fakes are real.
        Args: fake_pred: (B, 1, N, N) discriminator output on generated images
        Returns: scalar loss
        """
        real_labels = torch.ones_like(fake_pred)
        return self.criterion(fake_pred, real_labels)
    
    def discriminator_loss(self, real_pred: torch.Tensor, 
                           fake_pred: torch.Tensor) -> torch.Tensor:
        """Discriminator wants to correctly classify real vs fake.
        Args:
            real_pred: (B, 1, N, N) discriminator output on real images
            fake_pred: (B, 1, N, N) discriminator output on generated images
        Returns: scalar loss = (loss_real + loss_fake) / 2
        """
```

#### 3. `CompositeLoss` — Combined Training Objective

```python
class CompositeLoss(nn.Module):
    """
    Total generator loss combining pixel L1, MS-SSIM, and adversarial components.
    L_total = λ_adv * L_adv + λ_l1 * L_l1 + λ_ms * L_ms_ssim
    """
    def __init__(self, lambda_l1: float = 100.0, lambda_ms_ssim: float = 10.0, 
                 lambda_adv: float = 1.0):
        ...
    
    def forward(self, predicted: torch.Tensor, target: torch.Tensor,
                disc_fake_pred: torch.Tensor) -> dict:
        """
        Args:
            predicted: (B, 3, H, W) generator output
            target: (B, 3, H, W) ground truth
            disc_fake_pred: (B, 1, N, N) discriminator's judgment of predicted
        Returns:
            dict with keys: 'total', 'l1', 'ms_ssim', 'adversarial' (all scalars)
        """
```

### `__init__.py`
```python
from .composite_loss import MSSSIMLoss, AdversarialLoss, CompositeLoss
```

---

## Task 3: Forward Pass Unit Test (P0 — Day 3-4)

### File to Create
```
tests/__init__.py
tests/unit/__init__.py
tests/unit/test_model_forward.py
```

### What to Test

```python
import unittest
import torch
from model.networks.generator import ExposureConditionedAttentionUNet
from model.networks.discriminator import PatchGANDiscriminator
from model.losses.composite_loss import CompositeLoss, AdversarialLoss

class TestGeneratorForward(unittest.TestCase):
    """Validate generator tensor shapes through all stages."""
    
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gen = ExposureConditionedAttentionUNet().to(self.device)
        self.B, self.H, self.W = 2, 256, 256  # packed Bayer spatial dims
    
    def test_output_shape(self):
        """Generator (B,4,256,256) + (B,1) → (B,3,512,512)"""
        x = torch.randn(self.B, 4, self.H, self.W, device=self.device)
        alpha = torch.tensor([[100.0], [250.0]], device=self.device)
        out, attn = self.gen(x, alpha)
        self.assertEqual(out.shape, (self.B, 3, self.H * 2, self.W * 2))
    
    def test_output_range(self):
        """Output should be in [0, 1] due to Sigmoid head."""
        x = torch.randn(2, 4, 256, 256, device=self.device)
        out, _ = self.gen(x)
        self.assertTrue(out.min() >= 0.0)
        self.assertTrue(out.max() <= 1.0)
    
    def test_attention_maps(self):
        """Attention maps should be in [0, 1] and have correct shapes."""
        x = torch.randn(2, 4, 256, 256, device=self.device)
        out, attn_maps = self.gen(x, return_attention_maps=True)
        self.assertIsNotNone(attn_maps)
        self.assertEqual(len(attn_maps), 4)
        for a in attn_maps:
            self.assertTrue(a.min() >= 0.0)
            self.assertTrue(a.max() <= 1.0)
    
    def test_default_alpha(self):
        """Forward pass works with alpha_cond=None (defaults to 1.0)."""
        x = torch.randn(2, 4, 256, 256, device=self.device)
        out, _ = self.gen(x, alpha_cond=None)
        self.assertEqual(out.shape, (2, 3, 512, 512))

class TestDiscriminatorForward(unittest.TestCase):
    """Validate PatchGAN discriminator output shape."""
    
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # in_channels=7: 4 (packed Bayer upsampled) + 3 (sRGB target)
        self.disc = PatchGANDiscriminator(in_channels=7).to(self.device)
    
    def test_patch_output_shape(self):
        """Disc with 512x512 input → (B, 1, 61, 61) patch map."""
        condition = torch.randn(2, 4, 256, 256, device=self.device)  # will be interpolated
        target = torch.randn(2, 3, 512, 512, device=self.device)
        out = self.disc(condition, target)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 1)

class TestGradientFlow(unittest.TestCase):
    """Verify backward pass produces non-zero, finite gradients."""
    
    def test_generator_gradients(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        gen = ExposureConditionedAttentionUNet().to(device)
        x = torch.randn(1, 4, 256, 256, device=device)
        alpha = torch.tensor([[100.0]], device=device)
        out, _ = gen(x, alpha)
        loss = out.mean()
        loss.backward()
        # Check FiLM layer gets gradients
        film_grad = gen.film_cond.mlp[0].weight.grad
        self.assertIsNotNone(film_grad)
        self.assertFalse(torch.isnan(film_grad).any())
        self.assertFalse(torch.isinf(film_grad).any())
        self.assertTrue(film_grad.abs().sum() > 0)

class TestCompositeLoss(unittest.TestCase):
    """Verify loss computation produces valid scalars."""
    
    def test_loss_values(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        loss_fn = CompositeLoss().to(device)
        pred = torch.rand(2, 3, 512, 512, device=device)
        target = torch.rand(2, 3, 512, 512, device=device)
        fake_pred = torch.randn(2, 1, 61, 61, device=device)
        losses = loss_fn(pred, target, fake_pred)
        for key in ['total', 'l1', 'ms_ssim', 'adversarial']:
            self.assertIn(key, losses)
            self.assertFalse(torch.isnan(losses[key]))
            self.assertFalse(torch.isinf(losses[key]))
```

### How to Run
```powershell
python -m pytest tests/unit/test_model_forward.py -v
```

---

## Task 4: Training & Validation Loop (P0 — Day 5-7)

### Files to Create
```
model/configs/train_config.yaml
model/train.py
```

### `train_config.yaml`
```yaml
# LumaCam Training Configuration
data:
  data_root: "path/to/SID/Sony"
  train_list: "Sony_train_list.txt"
  val_list: "Sony_val_list.txt"
  patch_size: 512
  num_workers: 4

model:
  in_channels: 4
  out_channels: 3
  base_channels: 32
  cond_dim: 1

training:
  epochs: 200
  batch_size: 4
  lr_generator: 1.0e-4
  lr_discriminator: 1.0e-4
  beta1: 0.5
  beta2: 0.999
  lambda_l1: 100.0
  lambda_ms_ssim: 10.0
  lambda_adv: 1.0
  use_amp: true
  
  # Discriminator/Generator training ratio
  d_steps_per_g_step: 1
  
checkpointing:
  save_dir: "checkpoints/"
  save_every_n_epochs: 10
  save_best: true

logging:
  log_every_n_steps: 50
  val_every_n_epochs: 5
```

### `train.py` Architecture

```python
"""
Main training script for LumaCam GAN.

Usage:
    python model/train.py --config model/configs/train_config.yaml
    python model/train.py --config model/configs/train_config.yaml --dry-run --epochs 1 --batch-size 1

Key design decisions:
- Alternating D/G training (standard GAN protocol)
- PyTorch AMP (torch.amp) for mixed precision on limited 4GB VRAM
- Best checkpoint saved by validation PSNR
- Supports DummySIDDataset for dry-run testing without real data
"""

# Core training loop structure:
# 1. Parse config YAML + CLI overrides
# 2. Initialize Generator, Discriminator, CompositeLoss, AdversarialLoss
# 3. Adam optimizers for G and D with separate LR
# 4. GradScaler for AMP
# 5. DataLoader from SIDSonyDataset (or DummySIDDataset if --dry-run)
# 6. For each epoch:
#    a. TRAIN:
#       - For each batch (input, target, alpha):
#         - D step: real_pred = D(input, target), fake = G(input, alpha),
#                   fake_pred = D(input, fake.detach()), d_loss = adv.discriminator_loss(...)
#         - G step: fake = G(input, alpha), fake_pred = D(input, fake),
#                   g_losses = composite_loss(fake, target, fake_pred)
#    b. VALIDATE (every N epochs):
#       - Compute PSNR, SSIM on validation set
#       - Save best checkpoint if PSNR improves
#    c. Log metrics to console
```

---

## Task 5: Kaggle Training Notebook (P0 — Day 8-9)

### File to Create
```
notebooks/train_lumacam_kaggle.ipynb
```

### Key Sections
1. GPU detection (`torch.cuda.get_device_name()`)
2. Mount Kaggle dataset (upload SID Sony subset as Kaggle Dataset)
3. Install dependencies (`pip install rawpy lpips torchmetrics`)
4. Import from `model/` package
5. Training loop (copy from `train.py` or import directly)
6. Visualization: dark input vs enhanced output vs ground truth (side-by-side)
7. Attention gate heatmap overlay visualization
8. Download best checkpoint

---

## Task 7: INT8 Quantization (P1 — Day 10-13)

### File to Create
```
edge_opt/quantize_int8.py
```

### What to Build
- Depends on: Lakshmi's ONNX export (Task 6) producing `edge_opt/models/lumacam_fp32.onnx`
- Convert ONNX → TFLite FP32 → TFLite INT8 using `ai-edge-litert` or `tensorflow` converter
- Build a representative calibration dataset: 100 dark Bayer patches from SID Sony
- Measure PSNR drop between FP32 and INT8 outputs (target: < 0.5 dB)
- Output: `edge_opt/models/lumacam_fp32.tflite` and `edge_opt/models/lumacam_int8.tflite`

---

## Verification Checklist

- [x] `python -m pytest tests/unit/test_model_forward.py -v` — all 10 tests pass (3.75s)
- [x] `python model/train.py --config model/configs/train_config.yaml --dry-run --epochs 1 --batch-size 1` — completes without error (4.45s / epoch on GTX 1650)
- [ ] Training loss decreases over first 100 iterations on GTX 1650
- [ ] Kaggle notebook runs end-to-end on free P100
- [ ] INT8 PSNR drop < 0.5 dB vs FP32
