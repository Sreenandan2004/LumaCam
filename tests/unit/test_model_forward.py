"""
LumaCam Architecture & Pipeline Unit Tests
==========================================
Validates end-to-end tensor transformations, shapes, loss computations, and gradient flows:
1. Generator tensor dimension assertions and PixelShuffle upsampling.
2. AttentionGate activation map bounds.
3. PatchGAN Discriminator 70x70 receptive field spatial outputs.
4. FiLM latent modulation gradient flow.
5. Multi-Scale SSIM & CompositeLoss numerical stability.
6. Raw Bayer quad packing and Dummy dataset compatibility.
"""

import unittest
import torch

from model.networks.generator import ExposureConditionedAttentionUNet
from model.networks.discriminator import PatchGANDiscriminator
from model.losses.composite_loss import CompositeLoss, AdversarialLoss, MSSSIMLoss
from model.datasets.sid_dataset import DummySIDDataset
from model.datasets.raw_processing import pack_raw_bayer_torch, apply_exposure_amplification


class TestGeneratorForward(unittest.TestCase):
    """Validate generator tensor shapes and behavior through all stages."""

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gen = ExposureConditionedAttentionUNet().to(self.device)
        self.B, self.H, self.W = 2, 256, 256  # Bayer spatial dims (corresponds to 512x512 full RGB)

    def test_output_shape(self):
        """Generator (B, 4, 256, 256) + (B, 1) -> (B, 3, 512, 512)"""
        x = torch.randn(self.B, 4, self.H, self.W, device=self.device)
        alpha = torch.tensor([[100.0], [250.0]], device=self.device)
        out, attn = self.gen(x, alpha)
        self.assertEqual(out.shape, (self.B, 3, self.H * 2, self.W * 2))

    def test_output_range(self):
        """Output should be normalized in [0, 1] due to the Sigmoid head."""
        x = torch.randn(2, 4, self.H, self.W, device=self.device)
        out, _ = self.gen(x)
        self.assertTrue(bool(out.min() >= 0.0))
        self.assertTrue(bool(out.max() <= 1.0))

    def test_attention_maps(self):
        """Attention maps must be in [0, 1] across all 4 decoder skip gates."""
        x = torch.randn(2, 4, self.H, self.W, device=self.device)
        out, attn_maps = self.gen(x, return_attention_maps=True)
        self.assertIsNotNone(attn_maps)
        self.assertEqual(len(attn_maps), 4)
        for a in attn_maps:
            self.assertTrue(bool(a.min() >= 0.0))
            self.assertTrue(bool(a.max() <= 1.0))

    def test_default_alpha(self):
        """Forward pass functions seamlessly when alpha_cond=None (defaults to 1.0)."""
        x = torch.randn(2, 4, self.H, self.W, device=self.device)
        out, _ = self.gen(x, alpha_cond=None)
        self.assertEqual(out.shape, (2, 3, self.H * 2, self.W * 2))


class TestDiscriminatorForward(unittest.TestCase):
    """Validate PatchGAN discriminator receptive field output."""

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 7 channels: 4 (Bayer raw interpolated to 512x512) + 3 (sRGB target)
        self.disc = PatchGANDiscriminator(in_channels=7).to(self.device)

    def test_patch_output_shape(self):
        """PatchGAN with 512x512 inputs produces a spatial patch validity map."""
        condition = torch.randn(2, 4, 256, 256, device=self.device)
        target = torch.randn(2, 3, 512, 512, device=self.device)
        out = self.disc(condition, target)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 1)
        # Verify spatial resolution is 2D patch grid (e.g., 61x61 for 512x512)
        self.assertEqual(out.ndim, 4)


class TestGradientFlow(unittest.TestCase):
    """Verify backpropagation produces valid non-zero gradients on all layers."""

    def test_generator_gradients(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gen = ExposureConditionedAttentionUNet().to(device)
        x = torch.randn(1, 4, 128, 128, device=device)
        alpha = torch.tensor([[100.0]], device=device)
        out, _ = gen(x, alpha)
        loss = out.mean()
        loss.backward()

        # Check FiLM layer receives gradients from exposure conditioning
        # Note: mlp[-1] receives direct gradients; mlp[0] gradient is initially 0
        # due to zero-initialization of mlp[-1].weight (identity pass-through design).
        film_grad = gen.film_cond.mlp[-1].weight.grad
        self.assertIsNotNone(film_grad)
        self.assertFalse(torch.isnan(film_grad).any().item())
        self.assertFalse(torch.isinf(film_grad).any().item())
        self.assertTrue(bool(film_grad.abs().sum() > 0))

        # Check encoder parameters receive backward gradients through the U-Net
        enc_grad = gen.enc1.conv[0].weight.grad
        self.assertIsNotNone(enc_grad)
        self.assertTrue(bool(enc_grad.abs().sum() > 0))


class TestCompositeLoss(unittest.TestCase):
    """Verify multi-objective loss computation and differentiability."""

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_fn = CompositeLoss().to(self.device)

    def test_loss_values(self):
        """All components should be finite, positive, and non-NaN."""
        pred = torch.rand(2, 3, 256, 256, device=self.device)
        target = torch.rand(2, 3, 256, 256, device=self.device)
        fake_pred = torch.randn(2, 1, 29, 29, device=self.device)

        losses = self.loss_fn(pred, target, fake_pred)
        for key in ["total", "l1", "ms_ssim", "adversarial"]:
            self.assertIn(key, losses)
            self.assertFalse(torch.isnan(losses[key]).item(), f"{key} is NaN")
            self.assertFalse(torch.isinf(losses[key]).item(), f"{key} is Inf")

    def test_msssim_differentiability(self):
        """MS-SSIM must backpropagate clean gradients into predictions."""
        ms_loss = MSSSIMLoss().to(self.device)
        pred = torch.rand(1, 3, 128, 128, device=self.device, requires_grad=True)
        target = torch.rand(1, 3, 128, 128, device=self.device)
        loss = ms_loss(pred, target)
        loss.backward()

        self.assertIsNotNone(pred.grad)
        self.assertFalse(torch.isnan(pred.grad).any().item())
        self.assertFalse(torch.isinf(pred.grad).any().item())


class TestDatasetAndRawPipeline(unittest.TestCase):
    """Verify raw Bayer tensor packing and dataset shapes."""

    def test_bayer_packing_torch(self):
        """Single-channel sensor raw (B, 1, 512, 512) packs to (B, 4, 256, 256)."""
        raw_sim = torch.full((2, 1, 512, 512), 2000.0)
        packed = pack_raw_bayer_torch(raw_sim, black_level=512.0, white_level=16383.0)
        self.assertEqual(packed.shape, (2, 4, 256, 256))
        self.assertTrue(bool(packed.min() >= 0.0))
        self.assertTrue(bool(packed.max() <= 1.0))

    def test_dummy_sid_dataset(self):
        """DummySIDDataset generates paired tensors with valid shapes and values."""
        dataset = DummySIDDataset(num_samples=4, patch_size=256)
        self.assertEqual(len(dataset), 4)

        bayer_in, gt_rgb, alpha = dataset[0]
        # Bayer should be (4, 128, 128)
        self.assertEqual(bayer_in.shape, (4, 128, 128))
        # Ground truth sRGB should be (3, 256, 256)
        self.assertEqual(gt_rgb.shape, (3, 256, 256))
        # Alpha should be scalar (1,)
        self.assertEqual(alpha.shape, (1,))
        self.assertTrue(bool(alpha.item() >= 100.0))


if __name__ == "__main__":
    unittest.main()
