"""
LumaCam PatchGAN Discriminator Module
====================================
Implements a 70x70 receptive field Markovian PatchGAN discriminator (Isola et al., 2017)
as formulated in Thesia & Suthar (2026).

Background (Computational Photography):
Standard Mean Squared Error (L2) or Absolute Error (L1) regression losses average all
plausible high-frequency solutions, causing the infamous "waxy" or blurred texture problem.
The PatchGAN penalizes structure at the scale of local 70x70 image patches, classifying
whether each patch is real ground-truth or synthesized by the generator.
This forces the generator to hallucinate realistic high-frequency textures (foliage, grain, edges)
without drifting into unrealistic semantic hallucinations.
"""

import torch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 6, base_channels: int = 64, use_norm: bool = True):
        """
        Args:
            in_channels: Concatenated channel count of condition + target.
                         E.g., 3 (RGB condition) + 3 (RGB target) = 6 channels,
                         or 4 (Bayer raw upsampled) + 3 (sRGB) = 7 channels.
            base_channels: Number of feature maps in the first convolution layer.
            use_norm: Whether to apply InstanceNorm2d between convolution layers.
        """
        super().__init__()
        norm_layer = nn.InstanceNorm2d if use_norm else nn.Identity

        # 70x70 PatchGAN Receptive Field Structure
        # Stage 1: 4x4 conv, stride 2 (No norm on first layer as per pix2pix / PatchGAN design)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Stage 2: 4x4 conv, stride 2
        self.block2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            norm_layer(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Stage 3: 4x4 conv, stride 2
        self.block3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1, bias=False),
            norm_layer(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Stage 4: 4x4 conv, stride 1 (Preserves spatial resolution before final patch classification)
        self.block4 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=1, padding=1, bias=False),
            norm_layer(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Stage 5: 4x4 conv, stride 1 -> 1-channel patch probability map
        self.head = nn.Conv2d(base_channels * 8, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, condition: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            condition: Input image tensor, shape (B, C_in, H, W).
            target: Real or generated RGB image tensor, shape (B, 3, H, W).

        Returns:
            Patch validity matrix of shape (B, 1, N, N).
        """
        # Ensure condition matches spatial size of target if condition was packed Bayer (H/2, W/2)
        if condition.shape[2:] != target.shape[2:]:
            condition = nn.functional.interpolate(
                condition, size=target.shape[2:], mode="bilinear", align_corners=False
            )

        x = torch.cat([condition, target], dim=1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.head(x)
