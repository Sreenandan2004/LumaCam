"""
ONNX Exporter Tests
===================
Verifies model/export_onnx.py: export-wrapper contract, input-size validation, checkpoint loading,
ONNX graph validity, and PyTorch <-> ONNX Runtime numerical parity.

Small spatial sizes (64x64 packed Bayer) keep the suite fast on CPU-only laptops.
"""

import tempfile
import unittest
from pathlib import Path

import torch

from model.export_onnx import (
    DEFAULT_ATOL,
    GeneratorExportWrapper,
    export_to_onnx,
    load_generator,
    validate_input_size,
    verify_numerical_parity,
)
from model.networks.generator import ExposureConditionedAttentionUNet

try:
    import onnx  # noqa: F401
    import onnxruntime  # noqa: F401

    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

SIZE = (64, 64)  # Packed Bayer (H/2, W/2) -> 128x128 sRGB


class TestExportWrapper(unittest.TestCase):
    def test_wrapper_returns_single_tensor(self):
        """The ONNX graph must have exactly one output (the generator returns a tuple)."""
        wrapper = GeneratorExportWrapper(ExposureConditionedAttentionUNet().eval())
        with torch.no_grad():
            out = wrapper(torch.rand(1, 4, *SIZE), torch.tensor([[100.0]]))
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, (1, 3, SIZE[0] * 2, SIZE[1] * 2))

    def test_validate_input_size(self):
        validate_input_size(256, 128)
        for bad in ((100, 64), (64, 72), (0, 64)):
            with self.assertRaises(ValueError):
                validate_input_size(*bad)


class TestLoadGenerator(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_loads_train_py_checkpoint_with_config(self):
        """train.py-style checkpoints rebuild the generator from the stored model config."""
        source = ExposureConditionedAttentionUNet(base_channels=16)
        path = self.tmp / "ckpt.pt"
        torch.save(
            {"generator_state_dict": source.state_dict(), "config": {"model": {"base_channels": 16}}}, path
        )
        loaded = load_generator(str(path))
        self.assertFalse(loaded.training)
        for key, value in source.state_dict().items():
            self.assertTrue(torch.equal(value, loaded.state_dict()[key]), key)

    def test_loads_bare_state_dict(self):
        path = self.tmp / "bare.pth"
        torch.save(ExposureConditionedAttentionUNet().state_dict(), path)
        self.assertIsInstance(load_generator(str(path)), ExposureConditionedAttentionUNet)

    def test_missing_checkpoint_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_generator(str(self.tmp / "missing.pt"))


@unittest.skipUnless(HAS_ONNX, "onnx and onnxruntime are required")
class TestOnnxExport(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        torch.manual_seed(0)
        self.model = load_generator(None)

    def tearDown(self):
        self._tmp.cleanup()

    def test_fixed_shape_export_and_parity(self):
        path = export_to_onnx(self.model, str(self.tmp / "fixed.onnx"), input_size=SIZE)
        graph = onnx.load(str(path)).graph
        self.assertEqual([i.name for i in graph.input], ["input_bayer", "alpha_exposure"])
        self.assertEqual([o.name for o in graph.output], ["enhanced_rgb"])
        self.assertLess(verify_numerical_parity(self.model, str(path), input_size=SIZE), DEFAULT_ATOL)

    def test_dynamic_export_runs_at_other_sizes(self):
        path = export_to_onnx(self.model, str(self.tmp / "dynamic.onnx"), input_size=SIZE, dynamic=True)
        diff = verify_numerical_parity(self.model, str(path), input_size=(32, 96), batch_size=2)
        self.assertLess(diff, DEFAULT_ATOL)

    def test_export_rejects_invalid_size(self):
        with self.assertRaises(ValueError):
            export_to_onnx(self.model, str(self.tmp / "bad.onnx"), input_size=(60, 64))


if __name__ == "__main__":
    unittest.main()
