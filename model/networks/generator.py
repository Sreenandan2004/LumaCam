"""
LumaCam Exposure-Conditioned Attention U-Net Generator
======================================================
Implements the core generator architecture bridging Chen et al. (SID) and Thesia & Suthar (2026),
augmented with dynamic FiLM exposure conditioning at the bottleneck.

Architecture Flow:
1. Input: Packed 4-channel Bayer tensor (B, 4, H/2, W/2) + User Exposure scalar alpha (B, 1).
2. Encoder: 4 hierarchical downsampling stages extracting multi-scale contextual features.
3. Bottleneck: FiLM layer applies user exposure modulation in latent space.
4. Skip Connections: Spatial Attention Gates suppress shot noise in underexposed regions.
5. Decoder: 4 hierarchical upsampling stages reconstructing spatial and textural details.
6. Output Head: Sub-pixel PixelShuffle(2) reconstructs full-resolution sRGB (B, 3, H, W).
"""

from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention_gate import AttentionGate
from .conditioning import FiLMLayer


class ConvBlock(nn.Module):
    """Dual 3x3 Convolution block with LeakyReLU activation."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ExposureConditionedAttentionUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        base_channels: int = 32,
        cond_dim: int = 1,
        use_pixel_shuffle: bool = True,
    ):
        """
        Args:
            in_channels: Input channels (4 for packed RGGB Bayer raw).
            out_channels: Output sRGB channels (3).
            base_channels: Feature width of the first encoder stage (default 32 for edge efficiency).
            cond_dim: Dimension of exposure conditioning vector (default 1 for alpha scalar).
            use_pixel_shuffle: If True, uses Sub-Pixel Conv (PixelShuffle) to reconstruct full (H, W).
        """
        super().__init__()
        self.use_pixel_shuffle = use_pixel_shuffle
        c = base_channels  # e.g., 32

        # ------------------ ENCODER ------------------
        self.enc1 = ConvBlock(in_channels, c)         # (B, c, H/2, W/2)
        self.pool1 = nn.MaxPool2d(2)                  # (B, c, H/4, W/4)

        self.enc2 = ConvBlock(c, c * 2)               # (B, 2c, H/4, W/4)
        self.pool2 = nn.MaxPool2d(2)                  # (B, 2c, H/8, W/8)

        self.enc3 = ConvBlock(c * 2, c * 4)           # (B, 4c, H/8, W/8)
        self.pool3 = nn.MaxPool2d(2)                  # (B, 4c, H/16, W/16)

        self.enc4 = ConvBlock(c * 4, c * 8)           # (B, 8c, H/16, W/16)
        self.pool4 = nn.MaxPool2d(2)                  # (B, 8c, H/32, W/32)

        # ----------------- BOTTLENECK ----------------
        self.bottleneck = ConvBlock(c * 8, c * 16)    # (B, 16c, H/32, W/32)
        self.film_cond = FiLMLayer(cond_dim=cond_dim, feature_channels=c * 16)

        # ------------- ATTENTION GATES --------------
        self.ag4 = AttentionGate(in_channels_x=c * 8, in_channels_g=c * 16)
        self.ag3 = AttentionGate(in_channels_x=c * 4, in_channels_g=c * 8)
        self.ag2 = AttentionGate(in_channels_x=c * 2, in_channels_g=c * 4)
        self.ag1 = AttentionGate(in_channels_x=c, in_channels_g=c * 2)

        # ------------------ DECODER ------------------
        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(c * 16, c * 8)

        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(c * 8, c * 4)

        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(c * 4, c * 2)

        self.up1 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(c * 2, c)

        # ---------------- OUTPUT HEAD ----------------
        # Maps (H/2, W/2) back to full resolution (H, W, 3)
        if self.use_pixel_shuffle:
            # 12 channels at (H/2, W/2) -> PixelShuffle(2) yields 3 channels at (H, W)
            self.head = nn.Sequential(
                nn.Conv2d(c, out_channels * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.Sigmoid(),
            )
        else:
            self.head = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(c, out_channels, kernel_size=3, padding=1),
                nn.Sigmoid(),
            )

    def forward(
        self,
        x: torch.Tensor,
        alpha_cond: Optional[torch.Tensor] = None,
        return_attention_maps: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Forward pass of the Exposure-Conditioned Attention U-Net.

        Args:
            x: Packed Bayer tensor, shape (B, 4, H/2, W/2).
            alpha_cond: Exposure conditioning vector, shape (B, 1) or scalar.
                        If None, defaults to neutral conditioning (1.0).
            return_attention_maps: If True, returns soft attention weight tensors for visualization.

        Returns:
            out: Enhanced full-resolution sRGB image, shape (B, 3, H, W) in [0, 1].
            attn_maps: Optional list of attention maps [ag1, ag2, ag3, ag4].
        """
        if alpha_cond is None:
            alpha_cond = torch.ones((x.size(0), 1), device=x.device, dtype=x.dtype)
        elif alpha_cond.ndim == 1:
            alpha_cond = alpha_cond.unsqueeze(-1)

        # Encoder path
        e1 = self.enc1(x)             # (B, c, H/2, W/2)
        p1 = self.pool1(e1)           # (B, c, H/4, W/4)

        e2 = self.enc2(p1)           # (B, 2c, H/4, W/4)
        p2 = self.pool2(e2)           # (B, 2c, H/8, W/8)

        e3 = self.enc3(p2)           # (B, 4c, H/8, W/8)
        p3 = self.pool3(e3)           # (B, 4c, H/16, W/16)

        e4 = self.enc4(p3)           # (B, 8c, H/16, W/16)
        p4 = self.pool4(e4)           # (B, 8c, H/32, W/32)

        # Bottleneck + Latent FiLM Modulation
        bot = self.bottleneck(p4)     # (B, 16c, H/32, W/32)
        bot_modulated = self.film_cond(bot, alpha_cond)

        # Decoder path with Attention Gated skips
        u4 = self.up4(bot_modulated)  # (B, 8c, H/16, W/16)
        gated_e4, a4 = self.ag4(x=e4, g=bot_modulated)
        d4 = self.dec4(torch.cat([u4, gated_e4], dim=1))

        u3 = self.up3(d4)             # (B, 4c, H/8, W/8)
        gated_e3, a3 = self.ag3(x=e3, g=d4)
        d3 = self.dec3(torch.cat([u3, gated_e3], dim=1))

        u2 = self.up2(d3)             # (B, 2c, H/4, W/4)
        gated_e2, a2 = self.ag2(x=e2, g=d3)
        d2 = self.dec2(torch.cat([u2, gated_e2], dim=1))

        u1 = self.up1(d2)             # (B, c, H/2, W/2)
        gated_e1, a1 = self.ag1(x=e1, g=d2)
        d1 = self.dec1(torch.cat([u1, gated_e1], dim=1))

        # Reconstruct full-resolution sRGB (B, 3, H, W)
        out = self.head(d1)

        if return_attention_maps:
            return out, [a1, a2, a3, a4]
        return out, None
