"""
LumaCam Composite Loss Functions Module
=======================================
Implements the multi-objective training loss for the Exposure-Conditioned Attention U-Net GAN:
    L_total = λ_adv * L_adv + λ_l1 * L_l1 + λ_ms * L_ms_ssim

Background (Computational Photography):
In low-light image enhancement, optimizing strictly for pixel-level L1/L2 distance produces blurry,
over-smoothed results ("waxy skin", lack of high-frequency texture). Conversely, pure adversarial
loss can hallucinate fake artifacts that do not correspond to sensor reality.

By combining three complementary objectives:
1. Pixel L1 Loss: Enforces global color constancy and low-frequency tonal accuracy.
2. MS-SSIM Loss: Enforces structural preservation and contrast across 5 spatial scales (Wang et al., 2003).
3. Adversarial PatchGAN Loss: Forces the generator to synthesize crisp, realistic micro-textures.
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_window_1d(win_size: int, sigma: float) -> torch.Tensor:
    """Creates a normalized 1D Gaussian kernel."""
    coords = torch.arange(win_size, dtype=torch.float32) - (win_size - 1) / 2.0
    gauss = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    return gauss / gauss.sum()


def _gaussian_window_2d(win_size: int, sigma: float, channels: int) -> torch.Tensor:
    """Creates a 2D Gaussian convolution kernel for depthwise spatial filtering."""
    w1d = _gaussian_window_1d(win_size, sigma)
    w2d = torch.outer(w1d, w1d)
    # Shape: (channels, 1, win_size, win_size) for grouped conv2d
    return w2d.expand(channels, 1, win_size, win_size).contiguous()


def _ssim_components(
    x: torch.Tensor,
    y: torch.Tensor,
    window: torch.Tensor,
    c1: float,
    c2: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes luminance (l) and contrast-structure (cs) components of SSIM.

    Args:
        x: Tensor of shape (B, C, H, W) in [0, 1].
        y: Tensor of shape (B, C, H, W) in [0, 1].
        window: 2D Gaussian filter tensor of shape (C, 1, win_size, win_size).
        c1: Stability constant for luminance (K1 * L)^2.
        c2: Stability constant for contrast (K2 * L)^2.

    Returns:
        l: Luminance similarity map (B, C, H', W').
        cs: Contrast-structure similarity map (B, C, H', W').
    """
    channels = x.size(1)
    padding = window.size(-1) // 2

    mu_x = F.conv2d(x, window, padding=padding, groups=channels)
    mu_y = F.conv2d(y, window, padding=padding, groups=channels)

    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.relu(F.conv2d(x * x, window, padding=padding, groups=channels) - mu_x_sq)
    sigma_y_sq = F.relu(F.conv2d(y * y, window, padding=padding, groups=channels) - mu_y_sq)
    sigma_xy = F.conv2d(x * y, window, padding=padding, groups=channels) - mu_xy

    # Luminance measure: l(x, y) = (2 * mu_x * mu_y + C1) / (mu_x^2 + mu_y^2 + C1)
    l_map = (2.0 * mu_xy + c1) / torch.clamp(mu_x_sq + mu_y_sq + c1, min=1e-6)

    # Contrast-structure measure: cs(x, y) = (2 * sigma_xy + C2) / (sigma_x^2 + sigma_y^2 + C2)
    cs_map = (2.0 * sigma_xy + c2) / torch.clamp(sigma_x_sq + sigma_y_sq + c2, min=1e-6)

    # Numerically clamp to [0.0, 1.0] to prevent negative log inputs or NaN gradients
    l_map = torch.clamp(l_map, min=0.0, max=1.0)
    cs_map = torch.clamp(cs_map, min=0.0, max=1.0)

    return l_map, cs_map


class MSSSIMLoss(nn.Module):
    """
    Differentiable Multi-Scale Structural Similarity (MS-SSIM) Loss.

    References:
        Z. Wang, E. P. Simoncelli, and A. C. Bovik, "Multiscale structural similarity
        for image quality assessment," IEEE Asilomar Conference on Signals, Systems and
        Computers, 2003.

    Mathematical Formulation:
        MS-SSIM(x, y) = [l_M(x, y)]^(alpha_M) * prod_{j=1}^M [cs_j(x, y)]^(beta_j)
        Loss = 1 - MS-SSIM(x, y)
    """

    # Standard scale weights from Wang et al. (2003) for 5 scales
    DEFAULT_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)

    def __init__(
        self,
        data_range: float = 1.0,
        n_scales: int = 5,
        win_size: int = 11,
        win_sigma: float = 1.5,
        k1: float = 0.01,
        k2: float = 0.03,
        weights: Optional[Tuple[float, ...]] = None,
    ):
        """
        Args:
            data_range: Dynamic range of input images (1.0 for normalized [0, 1] RGB).
            n_scales: Number of resolution downsampling scales (default: 5).
            win_size: Spatial size of Gaussian sliding window (default: 11).
            win_sigma: Standard deviation of Gaussian sliding window (default: 1.5).
            k1: Stability coefficient for mean luminance (default: 0.01).
            k2: Stability coefficient for contrast/structure (default: 0.03).
            weights: Tuple of per-scale power exponents. Defaults to Wang et al. weights.
        """
        super().__init__()
        self.data_range = float(data_range)
        self.n_scales = n_scales
        self.win_size = win_size
        self.win_sigma = win_sigma
        self.c1 = (k1 * self.data_range) ** 2
        self.c2 = (k2 * self.data_range) ** 2

        if weights is None:
            weights = self.DEFAULT_WEIGHTS[:n_scales]
        # Normalize weights so they sum to 1.0
        weight_tensor = torch.tensor(weights, dtype=torch.float32)
        weight_tensor = weight_tensor / weight_tensor.sum()
        self.register_buffer("weights", weight_tensor)

        # Precompute Gaussian window for 3 RGB channels (default)
        window = _gaussian_window_2d(win_size, win_sigma, channels=3)
        self.register_buffer("window", window)

    def _get_window(self, channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Returns or lazily creates window matching channel count and device."""
        if self.window.size(0) == channels and self.window.device == device and self.window.dtype == dtype:
            return self.window
        return _gaussian_window_2d(self.win_size, self.win_sigma, channels).to(device=device, dtype=dtype)

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes 1 - MS-SSIM(predicted, target).

        Args:
            predicted: (B, C, H, W) predicted image in [0, 1].
            target: (B, C, H, W) ground truth image in [0, 1].

        Returns:
            Scalar loss tensor.
        """
        if predicted.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: predicted {predicted.shape} vs target {target.shape}"
            )

        # Force float32 precision and clamp to [0, 1] to prevent float16 / out-of-bounds instability
        predicted = torch.clamp(predicted.float(), 0.0, 1.0)
        target = torch.clamp(target.float(), 0.0, 1.0)

        b, c, h, w = predicted.shape
        window = self._get_window(c, predicted.device, predicted.dtype)

        # Check maximum allowable scales for image resolution
        # At each scale downsample by 2. Minimum spatial dim must exceed win_size.
        min_dim = min(h, w)
        available_scales = 1
        while available_scales < self.n_scales and (min_dim // (2 ** available_scales)) >= self.win_size:
            available_scales += 1

        active_weights = self.weights[:available_scales]
        active_weights = active_weights / active_weights.sum()

        mcs_list = []
        cur_pred = predicted
        cur_target = target

        for scale_idx in range(available_scales):
            l_map, cs_map = _ssim_components(
                cur_pred, cur_target, window, self.c1, self.c2
            )

            # Spatial mean over H' and W' (and mean across channels)
            cs_val = cs_map.mean(dim=(-2, -1))  # (B, C)

            if scale_idx == available_scales - 1:
                # At coarsest scale M, also retain luminance
                l_val = l_map.mean(dim=(-2, -1))   # (B, C)
                # Clamp before power to prevent NaN or complex numbers from negative noise
                l_val = torch.clamp(l_val, min=1e-4, max=1.0)
                mcs_list.append((cs_val, l_val))
            else:
                mcs_list.append((cs_val, None))
                # Iterative downsampling by factor of 2 (Average pooling)
                cur_pred = F.avg_pool2d(cur_pred, kernel_size=2, stride=2, padding=0)
                cur_target = F.avg_pool2d(cur_target, kernel_size=2, stride=2, padding=0)

        # Compute product across scales in stable log space:
        # MS-SSIM = (l_M^w_M) * prod(cs_j^w_j)
        # log(MS-SSIM) = w_M * log(l_M) + sum_j(w_j * log(cs_j))
        log_ms_ssim = torch.zeros((b, c), device=predicted.device, dtype=predicted.dtype)

        for scale_idx, (cs_val, l_val) in enumerate(mcs_list):
            w_s = active_weights[scale_idx]
            cs_clamped = torch.clamp(cs_val, min=1e-4, max=1.0)
            log_ms_ssim = log_ms_ssim + w_s * torch.log(cs_clamped)

            if l_val is not None:
                l_clamped = torch.clamp(l_val, min=1e-4, max=1.0)
                log_ms_ssim = log_ms_ssim + w_s * torch.log(l_clamped)

        ms_ssim = torch.clamp(torch.exp(log_ms_ssim).mean(), min=0.0, max=1.0)
        # Loss = 1 - MS-SSIM
        return 1.0 - ms_ssim


class AdversarialLoss(nn.Module):
    """
    Standard Conditional GAN Adversarial Loss using Binary Cross-Entropy with Logits.

    The PatchGAN discriminator outputs a spatial map of patch validity logits:
    - Real targets are labeled as 1.0
    - Generated fake targets are labeled as 0.0
    """

    def __init__(self):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def generator_loss(self, fake_pred: torch.Tensor) -> torch.Tensor:
        """
        The Generator aims to fool the Discriminator into classifying fake patches as real (1.0).

        Args:
            fake_pred: Logit predictions from discriminator D(condition, fake_image), shape (B, 1, N, N).

        Returns:
            Scalar generator adversarial loss.
        """
        real_labels = torch.ones_like(fake_pred)
        return self.criterion(fake_pred, real_labels)

    def discriminator_loss(
        self, real_pred: torch.Tensor, fake_pred: torch.Tensor
    ) -> torch.Tensor:
        """
        The Discriminator aims to classify real pairs as 1.0 and synthesized fake pairs as 0.0.

        Args:
            real_pred: Logit predictions on ground truth D(condition, real_image), shape (B, 1, N, N).
            fake_pred: Logit predictions on generated images D(condition, fake_image.detach()), shape (B, 1, N, N).

        Returns:
            Scalar discriminator adversarial loss: (loss_real + loss_fake) / 2.
        """
        real_labels = torch.ones_like(real_pred)
        fake_labels = torch.zeros_like(fake_pred)

        loss_real = self.criterion(real_pred, real_labels)
        loss_fake = self.criterion(fake_pred, fake_labels)

        return 0.5 * (loss_real + loss_fake)


class CompositeLoss(nn.Module):
    """
    Composite Multi-Objective Training Loss combining:
    - Pixel-wise L1 Reconstruction Loss (color/brightness baseline)
    - Multi-Scale Structural Similarity (MS-SSIM) Loss (spatial & contrast fidelity)
    - Adversarial PatchGAN Loss (fine micro-texture hallucination)

    L_total = λ_l1 * L_l1 + λ_ms * L_ms_ssim + λ_adv * L_adv
    """

    def __init__(
        self,
        lambda_l1: float = 100.0,
        lambda_ms_ssim: float = 10.0,
        lambda_adv: float = 1.0,
        data_range: float = 1.0,
    ):
        """
        Args:
            lambda_l1: Weight for L1 pixel reconstruction loss (default: 100.0).
            lambda_ms_ssim: Weight for MS-SSIM structural loss (default: 10.0).
            lambda_adv: Weight for PatchGAN adversarial loss (default: 1.0).
            data_range: Dynamic range of input images (default: 1.0).
        """
        super().__init__()
        self.lambda_l1 = float(lambda_l1)
        self.lambda_ms_ssim = float(lambda_ms_ssim)
        self.lambda_adv = float(lambda_adv)

        self.l1_loss = nn.L1Loss()
        self.ms_ssim_loss = MSSSIMLoss(data_range=data_range)
        self.adv_loss = AdversarialLoss()

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        disc_fake_pred: Optional[torch.Tensor] = None,
        use_adv: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes all component losses and total generator objective.

        Args:
            predicted: (B, 3, H, W) Generator output in [0, 1].
            target: (B, 3, H, W) Ground truth sRGB target in [0, 1].
            disc_fake_pred: Optional (B, 1, N, N) Discriminator output on predicted image.
            use_adv: Whether to include adversarial loss (set False during warmup).

        Returns:
            Dictionary with keys:
                - 'total': Weighted sum scalar for generator backprop.
                - 'l1': Unweighted L1 scalar loss.
                - 'ms_ssim': Unweighted MS-SSIM scalar loss.
                - 'adversarial': Unweighted generator adversarial scalar loss.
        """
        l1 = self.l1_loss(predicted, target)
        ms_ssim = self.ms_ssim_loss(predicted, target)

        if disc_fake_pred is not None and use_adv and self.lambda_adv > 0.0:
            adv = self.adv_loss.generator_loss(disc_fake_pred)
        else:
            adv = torch.tensor(0.0, device=predicted.device, dtype=predicted.dtype)

        total = (
            self.lambda_l1 * l1
            + self.lambda_ms_ssim * ms_ssim
            + self.lambda_adv * adv
        )

        return {
            "total": total,
            "l1": l1,
            "ms_ssim": ms_ssim,
            "adversarial": adv,
        }

