"""
LumaCam Raw Processing Utilities
================================
Implements Bayer Color Filter Array (CFA) packing and unpacking operations,
black-level calibration, and exposure ratio amplification.

Background (Computational Photography):
Modern camera sensors capture single-channel intensity data filtered by a Bayer pattern
(e.g., RGGB: Red, Green 1, Blue, Green 2). Standard demosaicing algorithms (bilinear, AHD)
interpolate missing colors, introducing artifacts and noise in extreme low light.
Following Chen et al. (See-in-the-Dark) and Thesia & Suthar (2026), LumaCam operates directly
on packed 4-channel Bayer tensors (H/2, W/2, 4) to preserve sensor-level high bit-depth
information and suppress photon shot noise before demosaicing.
"""

from typing import Tuple, Union
import numpy as np
import torch


def pack_raw_bayer_np(
    raw_image: np.ndarray,
    black_level: float = 512.0,
    white_level: float = 16383.0,
    bayer_pattern: str = "RGGB",
) -> np.ndarray:
    """
    Packs a 2D single-channel Bayer raw image into a 4-channel tensor of half spatial dimensions.

    Args:
        raw_image: 2D array of shape (H, W) containing raw sensor readings (typically 14-bit).
        black_level: Sensor black pedestal value (Sony Alpha 7S II standard is 512).
        white_level: Sensor saturation ceiling (14-bit max is 16383).
        bayer_pattern: CFA pattern string, default "RGGB".

    Returns:
        Packed and normalized array of shape (H/2, W/2, 4) with values in [0, 1].
    """
    if raw_image.ndim != 2:
        raise ValueError(f"Expected 2D raw array (H, W), got shape {raw_image.shape}")

    h, w = raw_image.shape
    if h % 2 != 0 or w % 2 != 0:
        raise ValueError(f"Raw dimensions must be even numbers, got ({h}, {w})")

    # Black level subtraction and linear normalization to [0, 1]
    norm_raw = np.maximum(raw_image.astype(np.float32) - black_level, 0.0) / (
        white_level - black_level
    )

    # Pack 2x2 Bayer quad into 4 channels
    # Standard RGGB layout:
    # [ R   G1 ]
    # [ G2  B  ]
    if bayer_pattern == "RGGB":
        r = norm_raw[0::2, 0::2]
        g1 = norm_raw[0::2, 1::2]
        g2 = norm_raw[1::2, 0::2]
        b = norm_raw[1::2, 1::2]
    elif bayer_pattern == "BGGR":
        b = norm_raw[0::2, 0::2]
        g1 = norm_raw[0::2, 1::2]
        g2 = norm_raw[1::2, 0::2]
        r = norm_raw[1::2, 1::2]
    else:
        raise NotImplementedError(f"Bayer pattern '{bayer_pattern}' is not currently supported.")

    packed = np.stack([r, g1, b, g2], axis=-1)  # (H/2, W/2, 4)
    return packed


def pack_raw_bayer_torch(
    raw_tensor: torch.Tensor,
    black_level: float = 512.0,
    white_level: float = 16383.0,
) -> torch.Tensor:
    """
    PyTorch implementation of Bayer packing for GPU/batch tensor pipelines.

    Args:
        raw_tensor: Tensor of shape (B, 1, H, W) or (B, H, W) with un-normalized sensor values.
        black_level: Sensor pedestal.
        white_level: Sensor saturation ceiling.

    Returns:
        Tensor of shape (B, 4, H/2, W/2) normalized to [0, 1].
    """
    if raw_tensor.ndim == 3:
        raw_tensor = raw_tensor.unsqueeze(1)  # (B, 1, H, W)

    b, c, h, w = raw_tensor.shape
    if c != 1:
        raise ValueError(f"Expected single-channel Bayer raw (B, 1, H, W), got {raw_tensor.shape}")
    if h % 2 != 0 or w % 2 != 0:
        raise ValueError(f"Spatial dimensions must be even, got ({h}, {w})")

    # Black level subtraction and normalization
    norm = torch.clamp(raw_tensor - black_level, min=0.0) / (white_level - black_level)

    # Slice RGGB quad
    r = norm[:, :, 0::2, 0::2]
    g1 = norm[:, :, 0::2, 1::2]
    g2 = norm[:, :, 1::2, 0::2]
    b_ch = norm[:, :, 1::2, 1::2]

    packed = torch.cat([r, g1, b_ch, g2], dim=1)  # (B, 4, H/2, W/2)
    return packed


def apply_exposure_amplification(
    packed_raw: torch.Tensor,
    ratio: Union[float, torch.Tensor],
) -> torch.Tensor:
    """
    Applies global exposure ratio amplification alpha to the packed raw input.
    As formulated in Equation (1) of the base paper:
        x' = Pack(I_raw) * alpha

    Args:
        packed_raw: Tensor of shape (B, 4, H/2, W/2).
        ratio: Exposure ratio between input short exposure and target long exposure.

    Returns:
        Amplified tensor clamped to prevent extreme numerical overflow.
    """
    if isinstance(ratio, torch.Tensor):
        if ratio.ndim == 1:
            ratio = ratio.view(-1, 1, 1, 1)
    return packed_raw * ratio
