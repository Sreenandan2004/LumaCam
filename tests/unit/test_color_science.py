"""
Color Science Tests
===================
Verifies model/postprocess/color_science.py: sRGB transfer functions, Gray-World AWB,
hue-preserving saturation, and the composite pipeline's no-double-gamma guarantee.
"""

import unittest

import numpy as np
import torch

from model.postprocess.color_science import (
    adjust_saturation,
    apply_color_pipeline,
    apply_color_pipeline_np,
    gray_world_awb,
    linear_to_srgb,
    srgb_to_linear,
)


def _random_image(seed: int = 0, shape=(2, 3, 32, 32)) -> torch.Tensor:
    return torch.rand(*shape, generator=torch.Generator().manual_seed(seed))


class TestTransferFunctions(unittest.TestCase):
    def test_round_trip(self):
        img = _random_image()
        self.assertTrue(torch.allclose(linear_to_srgb(srgb_to_linear(img)), img, atol=1e-5))

    def test_known_values(self):
        """Mid-gray sRGB 0.5 is ~0.214 linear; endpoints are fixed."""
        values = torch.tensor([0.0, 0.5, 1.0]).view(1, 3, 1, 1)
        linear = srgb_to_linear(values).flatten()
        self.assertAlmostEqual(float(linear[0]), 0.0, places=6)
        self.assertAlmostEqual(float(linear[1]), 0.2140, places=3)
        self.assertAlmostEqual(float(linear[2]), 1.0, places=5)


class TestGrayWorldAWB(unittest.TestCase):
    def test_neutral_image_unchanged(self):
        gray = torch.full((1, 3, 8, 8), 0.3)
        self.assertTrue(torch.allclose(gray_world_awb(gray), gray, atol=1e-5))

    def test_removes_color_cast(self):
        """A blue cast is equalized: channel means converge at full strength."""
        img = _random_image(shape=(1, 3, 16, 16)) * 0.5
        img[:, 2] *= 1.5
        means = gray_world_awb(img, strength=1.0).mean(dim=(-2, -1)).flatten()
        self.assertLess(float(means.max() - means.min()), 1e-3)

    def test_strength_zero_is_identity(self):
        img = _random_image() * 0.9
        self.assertTrue(torch.allclose(gray_world_awb(img, strength=0.0), img))

    def test_gains_are_capped(self):
        """A nearly single-color scene must not be amplified beyond max_gain."""
        img = torch.zeros(1, 3, 4, 4)
        img[:, 0] = 0.5
        img[:, 1] = 0.01
        out = gray_world_awb(img, strength=1.0, max_gain=2.0)
        self.assertLessEqual(float(out[:, 1].max()), 0.02 + 1e-6)

    def test_black_image_is_finite(self):
        out = gray_world_awb(torch.zeros(1, 3, 4, 4))
        self.assertTrue(torch.isfinite(out).all())
        self.assertEqual(float(out.abs().sum()), 0.0)


class TestSaturation(unittest.TestCase):
    def test_factor_one_is_identity(self):
        img = _random_image()
        self.assertTrue(torch.allclose(adjust_saturation(img, 1.0), img, atol=1e-6))

    def test_gray_pixels_unaffected(self):
        gray = torch.full((1, 3, 4, 4), 0.42)
        self.assertTrue(torch.allclose(adjust_saturation(gray, 2.0), gray))

    def test_boost_preserves_value_and_hue_order(self):
        img = _random_image()
        out = adjust_saturation(img, 1.5)
        self.assertTrue(torch.allclose(out.amax(dim=-3), img.amax(dim=-3), atol=1e-6))  # V preserved
        self.assertTrue(torch.equal(out.argmax(dim=-3), img.argmax(dim=-3)))            # dominant hue kept
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0)

    def test_boost_increases_saturation(self):
        img = torch.tensor([0.6, 0.4, 0.3]).view(1, 3, 1, 1)
        out = adjust_saturation(img, 1.2)
        sat = lambda t: float((t.amax(dim=-3) - t.amin(dim=-3)) / t.amax(dim=-3))  # noqa: E731
        self.assertAlmostEqual(sat(out), sat(img) * 1.2, places=5)


class TestColorPipeline(unittest.TestCase):
    def test_neutral_settings_are_identity_for_srgb(self):
        """No double gamma: model output (sRGB) passes through unchanged when corrections are off."""
        img = _random_image()
        out = apply_color_pipeline(img, awb_strength=0.0, saturation=1.0)
        self.assertTrue(torch.equal(out, img))

    def test_linear_input_is_encoded_once(self):
        img = _random_image()
        out = apply_color_pipeline(img, awb_strength=0.0, saturation=1.0, input_is_linear=True)
        self.assertTrue(torch.allclose(out, linear_to_srgb(img)))

    def test_defaults_stay_in_range(self):
        out = apply_color_pipeline(_random_image())
        self.assertTrue(torch.isfinite(out).all())
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0)

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            apply_color_pipeline(torch.rand(1, 4, 8, 8))
        with self.assertRaises(ValueError):
            apply_color_pipeline(torch.full((1, 3, 2, 2), float("nan")))

    def test_numpy_wrapper(self):
        img = np.random.default_rng(0).random((16, 24, 3), dtype=np.float32)
        out = apply_color_pipeline_np(img)
        self.assertEqual(out.shape, (16, 24, 3))
        self.assertEqual(out.dtype, np.float32)
        with self.assertRaises(TypeError):
            apply_color_pipeline_np((img * 255).astype(np.uint8))


if __name__ == "__main__":
    unittest.main()
