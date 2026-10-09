"""
LumaCam Exposure Latent Conditioning Module (FiLM)
==================================================
Implements Feature-wise Linear Modulation (FiLM) for dynamic user exposure control.

Background (Computational Photography):
Traditional LLIE networks (SID, EnlightenGAN) are rigid black-box models: given a dark image,
they produce a single fixed brightness output. In real mobile camera apps (such as Pixel GCam),
users expect tactile control over exposure compensation via an on-screen slider.

Rather than retraining separate networks or running brute-force post-hoc brightness scaling,
LumaCam introduces an interactive Exposure Conditioning Vector into the U-Net bottleneck.
A dedicated MLP predicts affine transformation parameters (gamma: scale, beta: shift):
    F_modulated = (1 + gamma(alpha)) * F + beta(alpha)
By initializing weights such that gamma=0 and beta=0, the network starts as an identity
pass-through and learns a smooth, continuous latent manifold for user exposure control.
"""

import torch
import torch.nn as nn


class FiLMLayer(nn.Module):
    def __init__(self, cond_dim: int, feature_channels: int, hidden_dim: int = None):
        """
        Args:
            cond_dim: Dimension of conditioning vector (typically 1 for exposure slider alpha).
            feature_channels: Channel depth of feature map at the U-Net bottleneck.
            hidden_dim: Hidden layer size for the modulation MLP.
        """
        super().__init__()
        if hidden_dim is None:
            hidden_dim = feature_channels // 2

        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, feature_channels * 2),
        )

        # Zero-initialize the final linear layer so modulation starts as identity
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, feature_map: torch.Tensor, cond_vector: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feature_map: Bottleneck tensor of shape (B, C, H, W).
            cond_vector: Exposure conditioning vector of shape (B, cond_dim) or (B,).

        Returns:
            Modulated feature map of shape (B, C, H, W).
        """
        if cond_vector.ndim == 1:
            cond_vector = cond_vector.unsqueeze(-1)  # (B, 1)

        # Project condition vector to affine parameters (gamma, beta)
        params = self.mlp(cond_vector)  # (B, 2 * C)
        gamma, beta = torch.chunk(params, 2, dim=-1)

        # Reshape for broadcasting across spatial dimensions
        gamma = gamma.view(-1, gamma.shape[-1], 1, 1)
        beta = beta.view(-1, beta.shape[-1], 1, 1)

        # Affine modulation: (1 + gamma) preserves identity when gamma=0
        return (1.0 + gamma) * feature_map + beta
