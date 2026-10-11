# 🟠 Lakshmi (Sreelakshmi A) — Task Handoff for AI Agent

> **Role**: Data Pipeline, ONNX Model Exporter & Color Science
> **Branch**: `feature/dataset-sid`
> **Hardware**: Laptop (CPU is fine; tests run on dummy tensors or small sample raw files)
> **AI Tool**: Use **Claude Code** or any AI coding agent with this document as context
> **Git Remote**: `https://github.com/Sreenandan2004/LumaCam.git`

---

## Project Context

**LumaCam** is a real-time low-light image enhancement mobile camera app. The core neural network is an **Exposure-Conditioned Attention U-Net GAN**.

**Your job**:
1. Implement the **SID Sony Dataset Loader** (`model/datasets/sid_dataset.py`) + dummy generator for tests.
2. Build the **ONNX Exporter** (`model/export_onnx.py`) to convert trained PyTorch models to standard ONNX.
3. Build the **Color Science Post-Processing Module** (`model/postprocess/color_science.py`).
4. Maintain **Documentation** (`Progress.md` and Phase deliverables).

> [!IMPORTANT]
> **Status as of 2026-10-12** (details in [`CHANGELOG.md`](CHANGELOG.md)):
> 1. ✅ **Task 2:** implemented by Nandan, then corrected by Lakshmi:
>    - list parsing, `data_root`, official split and input clipping fixed;
>    - the loader returns a `(input, target, alpha)` **tuple**, not the dict sketched below.
> 2. ✅ **Task 6:** `model/export_onnx.py` is done (parity 8.9e-08 vs PyTorch).
> 3. ✅ **Task 13:** `model/postprocess/color_science.py` is done.
>    The spec's extra `I^(1/2.2)` gamma below is **superseded**: the model already outputs gamma-encoded sRGB.
> 4. 🔄 **Docs:** [`docs/api_specifications.md`](docs/api_specifications.md) added. `base_paper_notes.md` and
>    `phase1_deliverables.md` are still to do.
>
> The sections below are kept as the original spec. Where they differ, the code and `docs/` win.

### Repository Setup
```bash
# Clone and switch to your branch
git clone https://github.com/Sreenandan2004/LumaCam.git
cd LumaCam
git checkout feature/dataset-sid
```

### Pre-existing Modules (Reference Only — Do Not Break)
- `model/datasets/raw_processing.py`: Has `pack_raw_bayer_np(raw_image)` and `pack_raw_bayer_torch()`. **Use this function directly**; do not rewrite Bayer packing logic!
- `model/networks/generator.py`: `ExposureConditionedAttentionUNet` expects:
  - Input: `(B, 4, H/2, W/2)`
  - Exposure Alpha: `(B, 1)`
  - Output: `(B, 3, H, W)`

---

## Task 2: See-in-the-Dark (SID) Dataset Loader (P0 — Week 1, Day 1-4)

### Files to Create
```
model/datasets/__init__.py
model/datasets/sid_dataset.py
```

### What to Build in `model/datasets/sid_dataset.py`

#### 1. Real Dataset: `SIDSonyDataset`
- Reads pair lists like `Sony_train_list.txt`:
  ```text
  ./Sony/short/00001_00_0.1s.ARW ./Sony/long/00001_00_30s.ARW ISO200 F8
  ```
- Parses short and long exposure times via regex (`(\d+\.?\d*)s`) and computes ratio:
  $$\alpha = \frac{\text{exposure\_time}_{long}}{\text{exposure\_time}_{short}}$$
- Unpacks raw with `rawpy.imread(short_path).raw_image_visible.copy()`
- Packs into Bayer 4-channel tensor: `pack_raw_bayer_np(...)` $\rightarrow (H/2, W/2, 4)$
- Scales packed raw by $\alpha$
- Ground truth target $y$:
  ```python
  gt = rawpy.imread(long_path).postprocess(
      use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16
  )  # (H, W, 3) uint16 -> convert to float32 / 65535.0 in [0, 1]
  ```
- Random crop of synchronized patches: crop raw patch size $(P/2, P/2, 4)$ matching sRGB patch $(P, P, 3)$ (default $P = 512$).
- Augmentations: Synchronized random horizontal/vertical flips.

#### 2. Synthetic Dummy Dataset: `DummySIDDataset`
Allows Nandan and Veni to run training dry-runs and unit tests **without needing the 25GB Sony dataset**:
```python
import torch
from torch.utils.data import Dataset
import random

class DummySIDDataset(Dataset):
    def __init__(self, num_samples: int = 50, patch_size: int = 512):
        self.num_samples = num_samples
        self.patch_size = patch_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int):
        h, w = self.patch_size // 2, self.patch_size // 2
        return {
            'input': torch.rand(4, h, w, dtype=torch.float32),
            'target': torch.rand(3, h * 2, w * 2, dtype=torch.float32),
            'alpha': torch.tensor([float(random.choice([100.0, 250.0, 300.0]))], dtype=torch.float32),
            'filename': f'dummy_{index:05d}',
        }
```

---

## Task 6: ONNX Model Exporter (P1 — Week 1/2, Day 5-8)

### File to Create
```
model/export_onnx.py
```

### Specifications
Converts the trained `ExposureConditionedAttentionUNet` into an ONNX computational graph (`.onnx`), enabling edge deployment via LiteRT.

```python
"""
LumaCam ONNX Exporter
=====================
Exports ExposureConditionedAttentionUNet to ONNX format with dual input signatures.

Usage:
    python model/export_onnx.py --checkpoint checkpoints/best_generator.pth \
                                --output edge_opt/models/lumacam_fp32.onnx \
                                --height 256 --width 256 --test
"""

import argparse
import os
import torch
import onnx
import onnxruntime as ort
import numpy as np

from model.networks.generator import ExposureConditionedAttentionUNet

def export_to_onnx(
    model: torch.nn.Module,
    output_path: str,
    input_size: tuple = (256, 256),
    dynamic_axes: bool = True,
    opset_version: int = 17,
):
    """
    Args:
        model: PyTorch generator in eval mode
        output_path: Path to write .onnx file
        input_size: (H_half, W_half) for packed Bayer (e.g. 256x256 -> outputs 512x512)
        dynamic_axes: If True, allows variable batch and spatial dimensions
        opset_version: ONNX opset (17+ required for stable PixelShuffle & LeakyReLU)
    """
    model.eval()
    dummy_bayer = torch.randn(1, 4, input_size[0], input_size[1], dtype=torch.float32)
    dummy_alpha = torch.tensor([[100.0]], dtype=torch.float32)

    input_names = ["input_bayer", "alpha_exposure"]
    output_names = ["enhanced_rgb"]

    dyn_axes_config = None
    if dynamic_axes:
        dyn_axes_config = {
            "input_bayer": {0: "batch_size", 2: "height", 3: "width"},
            "alpha_exposure": {0: "batch_size"},
            "enhanced_rgb": {0: "batch_size", 2: "out_height", 3: "out_width"},
        }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    torch.onnx.export(
        model,
        (dummy_bayer, dummy_alpha),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dyn_axes_config,
    )
    print(f"✅ Exported ONNX model to {output_path}")

    # Check graph validity
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX graph check passed!")

def verify_numerical_parity(model: torch.nn.Module, onnx_path: str, input_size=(256, 256)):
    """Verifies that PyTorch and ONNX Runtime outputs match within 1e-4."""
    model.eval()
    bayer_np = np.random.randn(1, 4, *input_size).astype(np.float32)
    alpha_np = np.array([[150.0]], dtype=np.float32)

    with torch.no_grad():
        torch_out, _ = model(torch.from_numpy(bayer_np), torch.from_numpy(alpha_np))
        torch_out_np = torch_out.cpu().numpy()

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_inputs = {"input_bayer": bayer_np, "alpha_exposure": alpha_np}
    ort_out = session.run(["enhanced_rgb"], ort_inputs)[0]

    max_diff = np.max(np.abs(torch_out_np - ort_out))
    print(f"Max absolute difference between PyTorch & ONNX: {max_diff:.6f}")
    assert max_diff < 1e-4, f"Numerical divergence too high: {max_diff}"
    print("✅ Numerical verification passed!")
```

---

## Task 13: Color Science Post-Processing (P1 — Week 2, Day 8-11)

### Files to Create
```
model/postprocess/__init__.py
model/postprocess/color_science.py
```

### What to Build in `color_science.py`
Raw model outputs in extreme low-light can look slightly flat or desaturated. Implement fast NumPy/PyTorch utilities:

1. **Auto White Balance (Gray World Algorithm)**:
   - Computes channel means $(\mu_R, \mu_G, \mu_B)$.
   - Target gray average: $\mu_{gray} = \frac{\mu_R + \mu_G + \mu_B}{3}$.
   - Gains: $k_R = \mu_{gray}/\mu_R, k_G = \mu_{gray}/\mu_G, k_B = \mu_{gray}/\mu_B$.
2. **Filmic Tone Curve / Gamma Correction**:
   - $I_{gamma} = I^{1/\gamma}$ (default $\gamma = 2.2$).
3. **Saturation Boost**:
   - Convert RGB $\rightarrow$ HSV, multiply $S$ channel by boost factor (e.g. $1.15$), convert back to RGB.
4. **Composite Pipeline**:
   - `apply_color_pipeline(img_rgb, awb=True, gamma=2.2, sat_boost=1.15)`

---

## Git Workflow for Lakshmi

```bash
# Pull latest changes daily
git pull origin feature/dataset-sid

# After creating/editing files:
git add model/datasets/ model/export_onnx.py model/postprocess/ docs/
git commit -m "feat(datasets): implement SID Sony dataset loader with dummy fallback"
git push origin feature/dataset-sid

# Create Pull Request on GitHub to 'dev' branch:
# Base: dev  <--  Compare: feature/dataset-sid
# Assign Sreenandan as reviewer
```

---

## Verification Checklist

- [x] `from model.datasets.sid_dataset import DummySIDDataset; d = DummySIDDataset(); print(d[0][0].shape)` prints `torch.Size([4, 256, 256])` (the loader returns a tuple).
- [x] `python model/export_onnx.py --random-init --test` (or `--checkpoint checkpoints/lumacam_best.pt --test`) passes ONNX validity checks and numerical parity.
- [x] `model/postprocess/color_science.py` enhances colors without clipping or double-gamma artifacts (`python -m pytest tests/unit/test_color_science.py`).
