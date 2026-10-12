"""
LumaCam Spatial Attention Gate Module
=====================================
Implements Oktay et al. (2018) Attention Gates as applied in Thesia & Suthar (2026).

Background (Computational Photography):
In extreme low-light scenarios, sensor noise dominates uniform background regions.
Standard U-Net skip connections blindly forward high-amplitude noise from the encoder
directly to the decoder, causing mottled chroma noise and artifacts.
The Attention Gate filters the encoder feature map (x) using gating signals (g)
from the coarser decoder representation, computing a spatial soft attention mask:
    x_hat = x * Sigmoid(psi(ReLU(W_x * x + W_g * g + b)))
This suppresses irrelevant noise while retaining sharp structural details.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate(nn.Module):
    def __init__(self, in_channels_x: int, in_channels_g: int, inter_channels: int = None):
        """
        Args:
            in_channels_x: Number of channels in the skip-connection encoder feature map x.
            in_channels_g: Number of channels in the decoder gating signal g.
            inter_channels: Intermediate bottleneck channels. Defaults to in_channels_x // 2.
        """
        super().__init__()
        if inter_channels is None:
            inter_channels = max(in_channels_x // 2, 1)

        # Linear transformations for skip features x and gating signal g
        self.w_x = nn.Conv2d(in_channels_x, inter_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.w_g = nn.Conv2d(in_channels_g, inter_channels, kernel_size=1, stride=1, padding=0, bias=True)

        # Normalization to prevent activation explosion or negative drift in deep layers
        num_groups = min(8, inter_channels)
        while inter_channels % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        self.norm = nn.GroupNorm(num_groups, inter_channels)
        self.relu = nn.ReLU(inplace=True)

        # Attention weight mapping psi: inter_channels -> 1 channel
        psi_conv = nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True)
        # Initialize bias to +1.0 so initial attention is ~0.73-0.85 (starts open, preventing dead gates)
        nn.init.constant_(psi_conv.bias, 1.0)
        nn.init.kaiming_normal_(psi_conv.weight, mode="fan_in")

        self.psi = nn.Sequential(
            psi_conv,
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Skip connection features from encoder, shape (B, C_x, H_x, W_x).
            g: Gating signal from decoder, shape (B, C_g, H_g, W_g).

        Returns:
            gated_x: Element-wise filtered features, shape (B, C_x, H_x, W_x).
            alpha: Soft spatial attention map, shape (B, 1, H_x, W_x).
        """
        # If g is lower resolution than x, interpolate g to match x
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=False)

        # Compute joint alignment score with GroupNorm
        theta_x = self.w_x(x)
        phi_g = self.w_g(g)
        f = self.relu(self.norm(theta_x + phi_g))

        # Compute soft attention coefficient map alpha in [0, 1]
        alpha = self.psi(f)

        # Modulate encoder features
        gated_x = x * alpha
        return gated_x, alpha

