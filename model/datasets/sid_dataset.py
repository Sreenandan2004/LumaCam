"""
LumaCam See-in-the-Dark (SID) Dataset Loader & Synthetic Pipeline
=================================================================
Implements PyTorch Dataset classes for training the Exposure-Conditioned Attention U-Net GAN:
1. `SIDSonyDataset`: Loads paired Sony Alpha 7S II (.ARW) raw sensor files (Chen et al., 2018).
2. `DummySIDDataset`: Fast synthetic data generator for offline testing, dry-runs, and CI.

Background (Computational Photography):
1. CFA Phase Alignment:
   Bayer raw data contains repeating 2x2 color filter quads (RGGB). When extracting random spatial
   crops from raw images, crop origin (top, left) coordinates MUST be even integers (crop_y % 2 == 0,
   crop_x % 2 == 0). An odd offset shifts the Bayer phase (e.g. Red becomes Green), completely
   scrambling channel interpretations!
2. Exposure Amplification Factor (alpha):
   In the SID dataset, dark images are captured with short exposure (e.g., 0.1s or 0.04s) and
   ground truth is captured with long exposure (e.g., 10s or 30s). The physical amplification ratio:
       alpha = exposure_time_long / exposure_time_short (often 100x to 300x)
   LumaCam feeds alpha both as an analytical input multiplier and as a conditioning vector to FiLM.
"""

import os
import re
from pathlib import Path
from typing import Tuple, List, Optional, Union
import numpy as np
import torch
from torch.utils.data import Dataset

from .raw_processing import pack_raw_bayer_np


def _parse_exposure_time(filename: str) -> float:
    """
    Extracts exposure time in seconds from Sony SID filenames.
    Example: '00001_00_0.1s.ARW' -> 0.1, '10003_00_10s.ARW' -> 10.0
    """
    stem = Path(filename).stem
    match = re.search(r"(\d+(?:\.\d+)?)s$", stem)
    if match:
        return float(match.group(1))
    # Fallback to default
    return 1.0


class SIDSonyDataset(Dataset):
    """
    Dataset loader for the See-in-the-Dark (SID) Sony dataset.

    Each item yields:
        - input_bayer: Amplified packed Bayer tensor (4, H/2, W/2) in [0, 1].
        - target_rgb: Ground-truth sRGB tensor (3, H, W) in [0, 1].
        - alpha: Exposure amplification scalar tensor (1,).
    """

    def __init__(
        self,
        data_root: Union[str, Path],
        list_file: Optional[Union[str, Path]] = None,
        patch_size: int = 512,
        is_train: bool = True,
        black_level: float = 512.0,
        white_level: float = 16383.0,
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            data_root: Root directory containing 'Sony/short/' and 'Sony/long/'.
            list_file: Path to text file listing short and long file paths.
            patch_size: Height/width of cropped training patch (must be divisible by 2).
            is_train: If True, performs random crops and augmentations; otherwise center crop.
            black_level: Sony Alpha 7S II black pedestal (512).
            white_level: Sony Alpha 7S II 14-bit ceiling (16383).
            max_samples: Optional limit on dataset size for fast validation/debugging.
        """
        super().__init__()
        self.data_root = Path(data_root)
        self.patch_size = patch_size
        self.is_train = is_train
        self.black_level = black_level
        self.white_level = white_level

        if patch_size % 2 != 0:
            raise ValueError(f"patch_size must be even to preserve Bayer 2x2 phase, got {patch_size}")

        self.file_pairs = self._load_file_pairs(list_file)
        if max_samples is not None:
            self.file_pairs = self.file_pairs[:max_samples]

    def _load_file_pairs(
        self, list_file: Optional[Union[str, Path]]
    ) -> List[Tuple[Path, Path, float, float]]:
        """Parses image pairs and exposure metadata."""
        pairs = []
        if list_file and Path(list_file).exists():
            with open(list_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        short_p = self.data_root / parts[0].lstrip("./")
                        long_p = self.data_root / parts[1].lstrip("./")
                        short_exp = float(parts[2]) if len(parts) >= 3 else _parse_exposure_time(parts[0])
                        long_exp = float(parts[3]) if len(parts) >= 4 else _parse_exposure_time(parts[1])
                        pairs.append((short_p, long_p, short_exp, long_exp))
        else:
            # Automatic directory scanning fallback
            short_dir = self.data_root / "short"
            long_dir = self.data_root / "long"
            if short_dir.is_dir() and long_dir.is_dir():
                short_files = sorted(list(short_dir.glob("*.ARW")) + list(short_dir.glob("*.arw")))
                for s_path in short_files:
                    # Sony SID format: ID_scene_exposure.ARW e.g., 00001_00_0.1s.ARW
                    prefix = "_".join(s_path.stem.split("_")[:2])
                    long_matches = list(long_dir.glob(f"{prefix}_*s.ARW")) + list(long_dir.glob(f"{prefix}_*s.arw"))
                    if long_matches:
                        l_path = long_matches[0]
                        s_exp = _parse_exposure_time(s_path.name)
                        l_exp = _parse_exposure_time(l_path.name)
                        pairs.append((s_path, l_path, s_exp, l_exp))
        return pairs

    def __len__(self) -> int:
        return len(self.file_pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        try:
            import rawpy
        except ImportError as e:
            raise ImportError(
                "rawpy is required to read .ARW raw files. Install with: pip install rawpy"
            ) from e

        short_path, long_path, short_exp, long_exp = self.file_pairs[idx]
        ratio = long_exp / max(short_exp, 1e-4)

        # 1. Read Short Exposure Raw Sensor Data
        with rawpy.imread(str(short_path)) as raw:
            raw_visible = raw.raw_image_visible.astype(np.float32)  # (H, W)

        # 2. Read Long Exposure Ground Truth and Demosaic to sRGB
        with rawpy.imread(str(long_path)) as raw_gt:
            # Convert ground truth raw to 8-bit sRGB with camera white balance
            gt_rgb = raw_gt.postprocess(
                use_camera_wb=True,
                half_size=False,
                no_auto_bright=True,
                output_bps=8,
            ).astype(np.float32) / 255.0  # (H, W, 3) in [0, 1]

        h, w = raw_visible.shape
        p = self.patch_size

        # 3. Spatial Crop (Origin must be even to preserve RGGB Bayer phase)
        if self.is_train and h > p and w > p:
            max_y = (h - p) // 2
            max_x = (w - p) // 2
            top = np.random.randint(0, max_y + 1) * 2
            left = np.random.randint(0, max_x + 1) * 2
        else:
            top = ((h - p) // 4) * 2
            left = ((w - p) // 4) * 2

        raw_patch = raw_visible[top : top + p, left : left + p]
        gt_patch = gt_rgb[top : top + p, left : left + p, :]

        # 4. Pack Bayer (p, p) -> (p/2, p/2, 4)
        packed_bayer = pack_raw_bayer_np(
            raw_patch,
            black_level=self.black_level,
            white_level=self.white_level,
            bayer_pattern="RGGB",
        )

        # Apply analytical exposure ratio amplification
        amplified_bayer = packed_bayer * ratio

        # 5. Data Augmentations (Performed consistently across Bayer & RGB)
        if self.is_train:
            if np.random.rand() > 0.5:  # Horizontal flip
                amplified_bayer = np.flip(amplified_bayer, axis=1)
                gt_patch = np.flip(gt_patch, axis=1)
            if np.random.rand() > 0.5:  # Vertical flip
                amplified_bayer = np.flip(amplified_bayer, axis=0)
                gt_patch = np.flip(gt_patch, axis=0)
            if np.random.rand() > 0.5:  # Transpose 90 deg
                amplified_bayer = np.transpose(amplified_bayer, (1, 0, 2))
                gt_patch = np.transpose(gt_patch, (1, 0, 2))

        # Convert to PyTorch format: (4, H/2, W/2) and (3, H, W)
        input_tensor = torch.from_numpy(amplified_bayer.transpose((2, 0, 1)).copy()).float()
        target_tensor = torch.from_numpy(gt_patch.transpose((2, 0, 1)).copy()).float()
        alpha_tensor = torch.tensor([ratio], dtype=torch.float32)

        return input_tensor, target_tensor, alpha_tensor


class DummySIDDataset(Dataset):
    """
    Synthetic Low-Light Dataset for offline testing, CI verification, and dry-runs.
    Generates realistic synthetic underexposed Bayer pairs with Poisson-Gaussian noise.
    """

    def __init__(
        self,
        num_samples: int = 16,
        patch_size: int = 512,
        seed: int = 42,
    ):
        """
        Args:
            num_samples: Number of synthetic pairs to simulate.
            patch_size: Spatial resolution of full RGB image (default: 512).
            seed: Random seed for deterministic reproducibility.
        """
        super().__init__()
        self.num_samples = num_samples
        self.patch_size = patch_size
        self.bayer_h = patch_size // 2
        self.bayer_w = patch_size // 2
        self.rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Generate clean synthetic ground truth (textured patterns & gradients)
        grid_y, grid_x = np.meshgrid(
            np.linspace(0, 1, self.patch_size),
            np.linspace(0, 1, self.patch_size),
            indexing="ij",
        )
        freq = (idx % 4 + 1) * 2.0
        r_chan = 0.5 + 0.4 * np.sin(freq * np.pi * grid_x) * np.cos(freq * np.pi * grid_y)
        g_chan = 0.5 + 0.4 * np.sin((freq + 1) * np.pi * grid_x)
        b_chan = 0.5 + 0.4 * np.cos((freq + 2) * np.pi * grid_y)
        clean_rgb = np.stack([r_chan, g_chan, b_chan], axis=0).astype(np.float32)  # (3, H, W)
        clean_rgb = np.clip(clean_rgb, 0.0, 1.0)

        # Realistic exposure amplification factor (e.g. 100x to 300x)
        ratio = float(100.0 + (idx % 5) * 50.0)

        # Downsample clean RGB to Bayer quad channels (4, H/2, W/2)
        r_quad = clean_rgb[0, 0::2, 0::2]
        g1_quad = clean_rgb[1, 0::2, 1::2]
        b_quad = clean_rgb[2, 1::2, 1::2]
        g2_quad = clean_rgb[1, 1::2, 0::2]
        bayer_clean = np.stack([r_quad, g1_quad, b_quad, g2_quad], axis=0)  # (4, H/2, W/2)

        # Simulate severe underexposure: divide by ratio
        bayer_dark = bayer_clean / ratio

        # Add physical sensor noise:
        # 1. Shot noise (Poisson, signal-dependent)
        # 2. Read noise (Gaussian, dark current / ADC noise)
        shot_noise = self.rng.normal(0, np.sqrt(np.maximum(bayer_dark, 0.0) / 100.0 + 1e-6)) * 0.01
        read_noise = self.rng.normal(0, 0.005, size=bayer_dark.shape)
        noisy_dark = np.clip(bayer_dark + shot_noise + read_noise, 0.0, 1.0)

        # Scale by amplification ratio (matches SID preprocessing pipeline)
        amplified_input = np.clip(noisy_dark * ratio, 0.0, 10.0).astype(np.float32)

        input_tensor = torch.from_numpy(amplified_input).float()
        target_tensor = torch.from_numpy(clean_rgb).float()
        alpha_tensor = torch.tensor([ratio], dtype=torch.float32)

        return input_tensor, target_tensor, alpha_tensor
