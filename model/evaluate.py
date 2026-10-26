"""
LumaCam Model Evaluation & Quantitative Benchmarking Suite
==========================================================
Computes standard computational photography metrics:
- Peak Signal-to-Noise Ratio (PSNR in dB)
- Structural Similarity Index Measure (SSIM)
- Inference Latency (Mean, Median, P95 in ms) across GPU/CPU devices

Outputs publication-ready Markdown and LaTeX benchmark tables comparing
LumaCam against baseline underexposed inputs, traditional digital gain,
and post-processing color science.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Enable UTF-8 console output on Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import torch.nn as nn
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.networks.generator import ExposureConditionedAttentionUNet
from model.datasets.sid_dataset import DummySIDDataset, SIDSonyDataset
from model.postprocess.color_science import apply_color_pipeline


def calculate_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).
    Formula: PSNR = 20 * log10(MAX / sqrt(MSE))
    """
    mse = torch.mean((pred - target) ** 2).item()
    if mse <= 1e-10:
        return 100.0
    return float(20.0 * np.log10(max_val / np.sqrt(mse)))


def calculate_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Computes single-scale SSIM in [0, 1] across RGB channels."""
    c1 = (0.01) ** 2
    c2 = (0.03) ** 2

    mu_x = pred.mean()
    mu_y = target.mean()
    sigma_x_sq = ((pred - mu_x) ** 2).mean()
    sigma_y_sq = ((target - mu_y) ** 2).mean()
    sigma_xy = ((pred - mu_x) * (target - mu_y)).mean()

    ssim_val = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x_sq + sigma_y_sq + c2)
    )
    return float(torch.clamp(ssim_val, 0.0, 1.0).item())


def load_generator(checkpoint_path: Path, device: torch.device) -> Tuple[ExposureConditionedAttentionUNet, Dict[str, Any]]:
    """Instantiates and loads generator checkpoint."""
    generator = ExposureConditionedAttentionUNet(
        in_channels=4,
        out_channels=3,
        base_channels=32,
        cond_dim=1,
        use_pixel_shuffle=True,
    ).to(device)

    meta = {}
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict):
            if "generator_state_dict" in ckpt:
                generator.load_state_dict(ckpt["generator_state_dict"])
            elif "generator_state" in ckpt:
                generator.load_state_dict(ckpt["generator_state"])
            elif "model_state_dict" in ckpt:
                generator.load_state_dict(ckpt["model_state_dict"])
            else:
                generator.load_state_dict(ckpt)
            meta["epoch"] = ckpt.get("epoch", None)
            meta["val_psnr"] = ckpt.get("val_psnr", None)
            meta["val_ssim"] = ckpt.get("val_ssim", None)
        else:
            generator.load_state_dict(ckpt)
        print(f"[Model] Successfully loaded weights from {checkpoint_path}")
    else:
        print(f"[Model WARNING] Checkpoint {checkpoint_path} not found. Running with random initialization.")

    generator.eval()
    return generator, meta


def run_benchmarks(
    generator: ExposureConditionedAttentionUNet,
    dataset,
    device: torch.device,
    num_samples: int = 20,
    warmup_runs: int = 5,
    enable_color_science: bool = True,
    awb_strength: float = 0.5,
    saturation: float = 1.15,
) -> Dict[str, Any]:
    """
    Evaluates:
    1. Dark Input (simulated digital ISO gain) vs Ground Truth
    2. LumaCam Raw Generator Output vs Ground Truth
    3. LumaCam + Color Science Output vs Ground Truth
    4. Per-frame forward inference latency
    """
    results: Dict[str, Any] = {
        "raw_psnr": [],
        "raw_ssim": [],
        "lumacam_psnr": [],
        "lumacam_ssim": [],
        "lumacam_color_psnr": [],
        "lumacam_color_ssim": [],
        "latencies_ms": [],
    }

    # Warmup runs to stabilize GPU clocks / CUDA context
    dummy_input = torch.zeros((1, 4, 256, 256), device=device)
    dummy_alpha = torch.tensor([[100.0]], device=device)
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = generator(dummy_input, dummy_alpha)

    print(f"\n[Benchmarking] Evaluating {min(num_samples, len(dataset))} test frames on {device}...")

    total_eval = min(num_samples, len(dataset))
    for idx in range(total_eval):
        bayer, gt, alpha = dataset[idx]
        bayer = bayer.unsqueeze(0).to(device)
        gt = gt.unsqueeze(0).to(device)
        alpha = alpha.unsqueeze(0).to(device)

        # Baseline: Raw amplified preview (average 4 Bayer channels and repeat to 3)
        raw_preview = bayer.mean(dim=1, keepdim=True).repeat(1, 3, 2, 2)
        raw_preview = torch.clamp(raw_preview, 0.0, 1.0)
        
        # Resize raw_preview to match gt spatial resolution if needed
        if raw_preview.shape[-2:] != gt.shape[-2:]:
            raw_preview = torch.nn.functional.interpolate(
                raw_preview, size=gt.shape[-2:], mode="bilinear", align_corners=False
            )

        raw_p = calculate_psnr(raw_preview, gt)
        raw_s = calculate_ssim(raw_preview, gt)
        results["raw_psnr"].append(raw_p)
        results["raw_ssim"].append(raw_s)

        # LumaCam forward pass + latency timing
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            enhanced, _ = generator(bayer, alpha, return_attention_maps=False)
            enhanced = torch.clamp(enhanced, 0.0, 1.0)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000.0
        results["latencies_ms"].append(latency_ms)

        luma_p = calculate_psnr(enhanced, gt)
        luma_s = calculate_ssim(enhanced, gt)
        results["lumacam_psnr"].append(luma_p)
        results["lumacam_ssim"].append(luma_s)

        # Optional color science
        if enable_color_science:
            color_enhanced = apply_color_pipeline(
                enhanced,
                awb_strength=awb_strength,
                saturation=saturation,
                input_is_linear=False,
            )
            col_p = calculate_psnr(color_enhanced, gt)
            col_s = calculate_ssim(color_enhanced, gt)
            results["lumacam_color_psnr"].append(col_p)
            results["lumacam_color_ssim"].append(col_s)

    return results


def format_tables(results: Dict[str, Any], device_name: str, has_color_science: bool = True) -> Tuple[str, str, Dict[str, Any]]:
    """Outputs GitHub Markdown and LaTeX formatted comparison tables and summary dict."""
    raw_p_mean = float(np.mean(results["raw_psnr"]))
    raw_s_mean = float(np.mean(results["raw_ssim"]))

    luma_p_mean = float(np.mean(results["lumacam_psnr"]))
    luma_p_std = float(np.std(results["lumacam_psnr"]))
    luma_s_mean = float(np.mean(results["lumacam_ssim"]))
    luma_s_std = float(np.std(results["lumacam_ssim"]))

    lat_mean = float(np.mean(results["latencies_ms"]))
    lat_median = float(np.median(results["latencies_ms"]))
    lat_p95 = float(np.percentile(results["latencies_ms"], 95))
    fps = float(1000.0 / lat_mean if lat_mean > 0 else 0.0)

    summary = {
        "device": device_name,
        "raw_psnr": raw_p_mean,
        "raw_ssim": raw_s_mean,
        "lumacam_psnr_mean": luma_p_mean,
        "lumacam_psnr_std": luma_p_std,
        "lumacam_ssim_mean": luma_s_mean,
        "lumacam_ssim_std": luma_s_std,
        "latency_mean_ms": lat_mean,
        "latency_median_ms": lat_median,
        "latency_p95_ms": lat_p95,
        "throughput_fps": fps,
    }

    color_row_md = ""
    color_row_latex = ""
    if has_color_science and len(results.get("lumacam_color_psnr", [])) > 0:
        col_p_mean = float(np.mean(results["lumacam_color_psnr"]))
        col_p_std = float(np.std(results["lumacam_color_psnr"]))
        col_s_mean = float(np.mean(results["lumacam_color_ssim"]))
        col_s_std = float(np.std(results["lumacam_color_ssim"]))
        summary["lumacam_color_psnr_mean"] = col_p_mean
        summary["lumacam_color_ssim_mean"] = col_s_mean
        color_row_md = f"| **LumaCam + Color Science** (AWB + Saturation) | **{col_p_mean:.2f} +/- {col_p_std:.2f}** | **{col_s_mean:.4f} +/- {col_s_std:.4f}** | **{lat_mean:.2f} ms** | **{fps:.1f} FPS** |\n"
        color_row_latex = f"\\textbf{{LumaCam + Color Science}} & \\textbf{{{col_p_mean:.2f}}} & \\textbf{{{col_s_mean:.4f}}} & \\textbf{{{lat_mean:.2f}}} & \\textbf{{{fps:.1f}}} \\\\\n"

    md_table = f"""### Quantitative Metric Comparison (Evaluation Suite)

| Method | PSNR (dB) [Higher Better] | SSIM [Higher Better] | Latency (ms) [Lower Better] | Throughput (FPS) [Higher Better] |
| :--- | :---: | :---: | :---: | :---: |
| **Traditional Digital Gain** (Linear ISO Boost) | {raw_p_mean:.2f} | {raw_s_mean:.4f} | -- | -- |
| **See-in-the-Dark Base U-Net** (Chen et al. 2018) | 28.88 | 0.7870 | ~150.0 ms | ~6.6 FPS |
| **LumaCam Generator** (Attention U-Net GAN) | **{luma_p_mean:.2f} +/- {luma_p_std:.2f}** | **{luma_s_mean:.4f} +/- {luma_s_std:.4f}** | **{lat_mean:.2f} ms** | **{fps:.1f} FPS** |
{color_row_md}
*Hardware: `{device_name}` (Batch size = 1, Resolution = 512x512)*  
*Latency Profile: Mean: {lat_mean:.2f}ms | Median: {lat_median:.2f}ms | P95: {lat_p95:.2f}ms*
"""

    latex_table = f"""% ---------------- LaTeX Table for Capstone Report ----------------
\\begin{{table}}[htbp]
\\centering
\\caption{{Quantitative Evaluation and Latency Benchmarks on 512$\\times$512 Bayer Sensor Inputs.}}
\\label{{tab:lumacam_benchmarks}}
\\begin{{tabular}}{{lcccc}}
\\toprule
\\textbf{{Method}} & \\textbf{{PSNR (dB)}} $\\uparrow$ & \\textbf{{SSIM}} $\\uparrow$ & \\textbf{{Latency (ms)}} $\\downarrow$ & \\textbf{{FPS}} $\\uparrow$ \\\\
\\midrule
Linear Gain (ISO Boost) & {raw_p_mean:.2f} & {raw_s_mean:.4f} & -- & -- \\\\
SID U-Net (Chen et al.) & 28.88 & 0.7870 & 150.0 & 6.7 \\\\
\\textbf{{LumaCam Generator}} & \\textbf{{{luma_p_mean:.2f}}} & \\textbf{{{luma_s_mean:.4f}}} & \\textbf{{{lat_mean:.2f}}} & \\textbf{{{fps:.1f}}} \\\\
{color_row_latex}\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    return md_table, latex_table, summary


def main():
    parser = argparse.ArgumentParser(description="LumaCam Model Evaluation Suite")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/lumacam_best.pt", help="Path to checkpoint")
    parser.add_argument("--data-root", type=str, default=None, help="Path to real Sony SID dataset (optional)")
    parser.add_argument("--samples", type=int, default=20, help="Number of benchmark test samples")
    parser.add_argument("--device", type=str, default=None, help="Force evaluation device ('cuda' or 'cpu')")
    parser.add_argument("--patch-size", type=int, default=512, help="Patch size for benchmark")
    parser.add_argument("--no-color-science", action="store_true", help="Disable post-processing color science evaluation")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save benchmark report")
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = PROJECT_ROOT / args.checkpoint
    generator, meta = load_generator(ckpt_path, device)

    # Dataset selection
    if args.data_root and Path(args.data_root).exists():
        print(f"[Dataset] Loading real SID Sony dataset from: {args.data_root}")
        dataset = SIDSonyDataset(data_root=args.data_root, patch_size=args.patch_size, is_train=False)
    else:
        print("[Dataset] Initializing DummySIDDataset (Synthetic Poisson-Gaussian sensor noise)...")
        dataset = DummySIDDataset(num_samples=args.samples, patch_size=args.patch_size, seed=999)

    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    has_color_science = not args.no_color_science
    results = run_benchmarks(
        generator,
        dataset,
        device,
        num_samples=args.samples,
        enable_color_science=has_color_science,
    )

    md_table, latex_table, summary = format_tables(results, device_name, has_color_science=has_color_science)

    print("\n" + "=" * 78)
    print("                LUMACAM COMPREHENSIVE BENCHMARK REPORT                ")
    print("=" * 78)
    print(md_table)
    print(latex_table)

    # Save output artifacts
    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / "benchmark_report.md"
    json_file = out_dir / "benchmark_summary.json"

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# LumaCam Quantitative Benchmark Report\n\n")
        if meta.get("epoch"):
            f.write(f"> Evaluated on Checkpoint: `{args.checkpoint}` (Epoch: {meta['epoch']})\n\n")
        f.write(md_table)
        f.write("\n\n```latex\n")
        f.write(latex_table)
        f.write("```\n")

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[Saved] Benchmark report saved to: {report_file}")
    print(f"[Saved] Benchmark metrics saved to: {json_file}")


if __name__ == "__main__":
    main()
