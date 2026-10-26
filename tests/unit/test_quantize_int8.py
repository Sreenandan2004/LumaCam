"""
Unit Tests for LumaCam INT8 Post-Training Quantization
======================================================
Verifies:
1. CalibrationDataReader yields correctly shaped dictionaries with expected keys.
2. CalibrationDataReader rewinds correctly.
3. Quantization pipeline generates smaller INT8 model and passes sanity check.
"""

from pathlib import Path
import pytest
import numpy as np

from edge_opt.quantize_int8 import SIDCalibrationDataReader, quantize_lumacam_onnx

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FP32_MODEL = PROJECT_ROOT / "edge_opt" / "models" / "lumacam_generator.onnx"


class TestCalibrationDataReader:
    def test_reader_yields_valid_batches(self):
        reader = SIDCalibrationDataReader(num_samples=3, height=64, width=64)
        batch = reader.get_next()
        assert batch is not None
        assert "input_bayer" in batch
        assert "alpha_exposure" in batch
        assert batch["input_bayer"].shape == (1, 4, 64, 64)
        assert batch["alpha_exposure"].shape == (1, 1)

    def test_reader_rewind(self):
        reader = SIDCalibrationDataReader(num_samples=2, height=32, width=32)
        _ = reader.get_next()
        _ = reader.get_next()
        assert reader.get_next() is None
        reader.rewind()
        assert reader.get_next() is not None


class TestQuantizationExecution:
    @pytest.mark.skipif(not FP32_MODEL.exists(), reason="FP32 ONNX model must exist")
    def test_quantize_creates_smaller_model(self, tmp_path):
        out_int8 = tmp_path / "lumacam_test_int8.onnx"
        res = quantize_lumacam_onnx(
            FP32_MODEL,
            out_int8,
            use_static=True,
            num_calibration_samples=2,
        )
        assert out_int8.exists()
        assert res["int8_size_mb"] < res["fp32_size_mb"]
        assert res["compression_percent"] > 50.0
        assert res["output_shape"] == [1, 3, 512, 512]
