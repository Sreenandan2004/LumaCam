"""
LumaCam Post-Inference Color Science
====================================
Lightweight color and tone corrections applied to the generator output before display / storage.

Background (Computational Photography):
1. What the generator outputs:
   Training targets are produced by `rawpy.postprocess(use_camera_wb=True, no_auto_bright=True)`,
   which applies the camera white balance AND a display gamma curve (BT.709-style, ~1/2.2).
   The model therefore already outputs gamma-encoded, white-balanced sRGB. Applying another
   `I^(1/2.2)` gamma on top would double-encode it and produce a flat, washed-out image, so this
   module only encodes to sRGB when the input is declared linear (`input_is_linear=True`).
2. White balance belongs in linear light:
   Light mixes linearly, so channel gains are only physically meaningful on linear intensities.
   Gray-World AWB here decodes sRGB -> linear, applies per-channel gains, and re-encodes.
   Because the model already learned the camera white balance, the correction is blended with
   `strength` (default 0.5) and each gain is capped, so AWB nudges residual casts instead of
   fighting the model.
3. Hue-preserving saturation without an HSV round trip:
   For a pixel with V = max(R, G, B), scaling HSV saturation by k at fixed hue and value is exactly
       c' = V - k * (V - c)          for every channel c.
   S can never exceed 1, so k is capped per pixel at V / (V - min(R, G, B)); this keeps colors in
   gamut without the hue shifts that per-channel clipping would cause.

All functions operate on PyTorch tensors of shape (..., 3, H, W) with values in [0, 1], so the same
code runs on CPU, GPU, or batched model outputs. `apply_color_pipeline_np` wraps the pipeline for
NumPy (H, W, 3) images.
"""

import numpy as np
import torch

_EPS = 1e-6

# IEC 61966-2-1 (sRGB) transfer-function constants
_SRGB_LINEAR_THRESHOLD = 0.0031308
_SRGB_ENCODED_THRESHOLD = 0.04045
_SRGB_LINEAR_SLOPE = 12.92
_SRGB_GAMMA = 2.4
_SRGB_OFFSET = 0.055


def _check_rgb(img: torch.Tensor) -> None:
    """Raises ValueError unless img has shape (..., 3, H, W)."""
    if img.ndim < 3 or img.shape[-3] != 3:
        raise ValueError(f"Expected an RGB tensor of shape (..., 3, H, W), got {tuple(img.shape)}")


def srgb_to_linear(img: torch.Tensor) -> torch.Tensor:
    """
    Decodes gamma-encoded sRGB to linear light (IEC 61966-2-1).

    Args:
        img: (..., 3, H, W) sRGB values in [0, 1].
    Returns:
        (..., 3, H, W) linear values in [0, 1].
    """
    img = img.clamp(0.0, 1.0)
    low = img / _SRGB_LINEAR_SLOPE
    high = ((img + _SRGB_OFFSET) / (1.0 + _SRGB_OFFSET)) ** _SRGB_GAMMA
    return torch.where(img <= _SRGB_ENCODED_THRESHOLD, low, high)


def linear_to_srgb(img: torch.Tensor) -> torch.Tensor:
    """
    Encodes linear light to gamma-encoded sRGB (IEC 61966-2-1). This is the "gamma correction"
    step for linear inputs; it must NOT be applied to the generator output, which is already sRGB.

    Args:
        img: (..., 3, H, W) linear values in [0, 1].
    Returns:
        (..., 3, H, W) sRGB values in [0, 1].
    """
    img = img.clamp(0.0, 1.0)
    low = img * _SRGB_LINEAR_SLOPE
    high = (1.0 + _SRGB_OFFSET) * img.clamp(min=_SRGB_LINEAR_THRESHOLD) ** (1.0 / _SRGB_GAMMA) - _SRGB_OFFSET
    return torch.where(img <= _SRGB_LINEAR_THRESHOLD, low, high)


def gray_world_awb(
    img_linear: torch.Tensor,
    strength: float = 1.0,
    max_gain: float = 2.0,
    clip_threshold: float = 0.98,
) -> torch.Tensor:
    """
    Gray-World auto white balance on LINEAR RGB.

    Assumes the average scene reflectance is achromatic: per-channel means (mu_R, mu_G, mu_B) are
    equalized to mu_gray = (mu_R + mu_G + mu_B) / 3 with gains k_c = mu_gray / mu_c.

    Args:
        img_linear: (..., 3, H, W) linear RGB in [0, 1].
        strength: Blend between no correction (0.0) and full Gray-World (1.0).
        max_gain: Each gain is clamped to [1 / max_gain, max_gain] to avoid wild casts on
            scenes dominated by one color (where the Gray-World assumption fails).
        clip_threshold: Pixels with any channel above this are excluded from the statistics,
            because clipped highlights carry no reliable color information.

    Returns:
        (..., 3, H, W) white-balanced linear RGB in [0, 1].
    """
    _check_rgb(img_linear)
    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"strength must be in [0, 1], got {strength}")
    if max_gain < 1.0:
        raise ValueError(f"max_gain must be >= 1, got {max_gain}")

    # Statistics mask: (..., 1, H, W), True where no channel is clipped
    valid = (img_linear.amax(dim=-3, keepdim=True) < clip_threshold).to(img_linear.dtype)
    valid_count = valid.sum(dim=(-2, -1), keepdim=True)                                  # (..., 1, 1, 1)
    channel_means = (img_linear * valid).sum(dim=(-2, -1), keepdim=True) / (valid_count + _EPS)  # (..., 3, 1, 1)

    gray = channel_means.mean(dim=-3, keepdim=True)                                     # (..., 1, 1, 1)
    gains = (gray / (channel_means + _EPS)).clamp(1.0 / max_gain, max_gain)
    # Images with no usable pixels (fully clipped or black) are left untouched
    has_stats = (valid_count > 0) & (gray > _EPS)
    gains = torch.where(has_stats, gains, torch.ones_like(gains))
    gains = 1.0 + strength * (gains - 1.0)

    return (img_linear * gains).clamp(0.0, 1.0)


def adjust_saturation(img: torch.Tensor, factor: float = 1.15) -> torch.Tensor:
    """
    Scales HSV saturation by `factor` while preserving hue and value (see module docstring).

    Args:
        img: (..., 3, H, W) RGB in [0, 1] (sRGB or linear; display-space sRGB is typical).
        factor: Saturation multiplier (1.0 = identity, < 1 desaturates, > 1 boosts).

    Returns:
        (..., 3, H, W) RGB in [0, 1].
    """
    _check_rgb(img)
    if factor < 0.0:
        raise ValueError(f"factor must be >= 0, got {factor}")

    value = img.amax(dim=-3, keepdim=True)        # V = max(R, G, B), (..., 1, H, W)
    minimum = img.amin(dim=-3, keepdim=True)
    chroma = value - minimum                      # V * S
    # Largest k that keeps S' = k * S <= 1 (i.e. the new minimum >= 0)
    k_max = value / chroma.clamp(min=_EPS)
    k = torch.clamp(k_max, max=factor)
    return (value - k * (value - img)).clamp(0.0, 1.0)


def apply_color_pipeline(
    img: torch.Tensor,
    awb_strength: float = 0.5,
    saturation: float = 1.15,
    input_is_linear: bool = False,
) -> torch.Tensor:
    """
    Full post-inference color pipeline: [decode] -> AWB (linear) -> encode -> saturation.

    Args:
        img: (..., 3, H, W) RGB in [0, 1]. By default this is the generator output (sRGB).
        awb_strength: Gray-World AWB blend in [0, 1]; 0 disables AWB.
        saturation: HSV saturation multiplier; 1.0 disables the boost.
        input_is_linear: Set True for linear inputs (e.g. a linear demosaic). They are then
            encoded to sRGB (the gamma-correction step); sRGB inputs are never re-encoded.

    Returns:
        (..., 3, H, W) display-ready sRGB in [0, 1].
    """
    _check_rgb(img)
    if torch.isnan(img).any():
        raise ValueError("Input image contains NaN values")

    linear = img.clamp(0.0, 1.0) if input_is_linear else srgb_to_linear(img)
    if awb_strength > 0.0:
        linear = gray_world_awb(linear, strength=awb_strength)

    if input_is_linear or awb_strength > 0.0:
        srgb = linear_to_srgb(linear)
    else:
        srgb = img.clamp(0.0, 1.0)  # Untouched sRGB input: skip a lossy decode/encode round trip

    if saturation != 1.0:
        srgb = adjust_saturation(srgb, factor=saturation)
    return srgb


def apply_color_pipeline_np(
    img_hwc: np.ndarray,
    awb_strength: float = 0.5,
    saturation: float = 1.15,
    input_is_linear: bool = False,
) -> np.ndarray:
    """
    NumPy convenience wrapper around `apply_color_pipeline`.

    Args:
        img_hwc: (H, W, 3) float image in [0, 1] (e.g. a saved model output).
        awb_strength, saturation, input_is_linear: See `apply_color_pipeline`.

    Returns:
        (H, W, 3) float32 sRGB image in [0, 1].
    """
    if img_hwc.ndim != 3 or img_hwc.shape[-1] != 3:
        raise ValueError(f"Expected an (H, W, 3) image, got shape {img_hwc.shape}")
    if not np.issubdtype(img_hwc.dtype, np.floating):
        raise TypeError(f"Expected a float image in [0, 1], got dtype {img_hwc.dtype}")

    tensor = torch.from_numpy(np.ascontiguousarray(img_hwc, dtype=np.float32)).permute(2, 0, 1)  # (3, H, W)
    with torch.no_grad():
        out = apply_color_pipeline(
            tensor, awb_strength=awb_strength, saturation=saturation, input_is_linear=input_is_linear
        )
    return out.permute(1, 2, 0).numpy()  # (H, W, 3)
