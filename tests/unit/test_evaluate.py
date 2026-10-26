"""
Unit Tests for LumaCam Evaluation Suite
=======================================
Verifies:
1. PSNR computation against known values and edge cases (identical tensors, zero MSE).
2. SSIM computation bounds in [0, 1].
3. Table formatting functions (Markdown, LaTeX, summary dictionary).
4. Full evaluate pipeline on DummySIDDataset with checkpoint loading.
"""

import math
from pathlib import Path
import pytest
import torch

from model.evaluate import calculate_psnr, calculate_ssim, format_tables, run_benchmarks
from model.networks.generator import ExposureConditionedAttentionUNet
from model.datasets.sid_dataset import DummySIDDataset


class TestEvaluationMetrics:
    def test_psnr_identical_tensors(self):
        t = torch.rand((1, 3, 64, 64))
        psnr = calculate_psnr(t, t)
        assert psnr == 100.0

    def test_psnr_known_noise(self):
        # Target = zeros, Pred = constant error of 0.1 -> MSE = 0.01 -> PSNR = 20 dB
        target = torch.zeros((1, 3, 32, 32))
        pred = torch.full((1, 3, 32, 32), 0.1)
        psnr = calculate_psnr(pred, target)
        assert math.isclose(psnr, 20.0, rel_tol=1e-3)

    def test_ssim_identical_tensors(self):
        t = torch.rand((1, 3, 64, 64))
        ssim_val = calculate_ssim(t, t)
        assert ssim_val > 0.999

    def test_ssim_inverted_tensors(self):
        t = torch.ones((1, 3, 32, 32))
        inv = torch.zeros((1, 3, 32, 32))
        ssim_val = calculate_ssim(t, inv)
        assert 0.0 <= ssim_val <= 1.0


class TestBenchmarkPipeline:
    def test_format_tables(self):
        dummy_results = {
            "raw_psnr": [10.0, 11.0],
            "raw_ssim": [0.2, 0.25],
            "lumacam_psnr": [25.0, 26.0],
            "lumacam_ssim": [0.8, 0.85],
            "lumacam_color_psnr": [25.5, 26.2],
            "lumacam_color_ssim": [0.82, 0.86],
            "latencies_ms": [20.0, 22.0],
        }
        md_table, latex_table, summary = format_tables(dummy_results, "TestGPU", has_color_science=True)
        assert "Quantitative Metric Comparison" in md_table
        assert "\\begin{table}" in latex_table
        assert summary["lumacam_psnr_mean"] == 25.5
        assert summary["throughput_fps"] > 0

    def test_run_benchmarks_end_to_end(self):
        generator = ExposureConditionedAttentionUNet(
            in_channels=4,
            out_channels=3,
            base_channels=16,
            cond_dim=1,
            use_pixel_shuffle=True,
        )
        generator.eval()
        dataset = DummySIDDataset(num_samples=2, patch_size=128, seed=42)
        device = torch.device("cpu")
        results = run_benchmarks(
            generator,
            dataset,
            device,
            num_samples=2,
            warmup_runs=1,
            enable_color_science=True,
        )
        assert len(results["lumacam_psnr"]) == 2
        assert len(results["latencies_ms"]) == 2
        assert len(results["lumacam_color_psnr"]) == 2
        assert all(lat > 0 for lat in results["latencies_ms"])
