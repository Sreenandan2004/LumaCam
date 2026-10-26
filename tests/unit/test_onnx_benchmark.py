"""
Unit Tests for LumaCam ONNX Benchmark
=====================================
Verifies:
1. Benchmark runner executes without error on exported ONNX model.
2. Latency dictionary returns non-zero positive numbers and correct keys.
"""

from pathlib import Path
import pytest
from edge_opt.benchmark_onnx import benchmark_onnx_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "edge_opt" / "models" / "lumacam_generator.onnx"


class TestONNXBenchmark:
    @pytest.mark.skipif(not MODEL_PATH.exists(), reason="ONNX model must be exported first")
    def test_benchmark_onnx_returns_valid_metrics(self):
        res = benchmark_onnx_model(MODEL_PATH, num_runs=2, warmup_runs=1)
        assert res["size_mb"] > 25.0
        assert res["mean_latency_ms"] > 0.0
        assert res["fps"] > 0.0
        assert "provider" in res
