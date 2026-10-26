"""
LumaCam ONNX Runtime Edge Benchmark
===================================
Benchmarks the exported ONNX model (`lumacam_generator.onnx`) using ONNX Runtime.
Measures latency statistics (mean, median, P95) and throughput (FPS) across
available execution providers (CPU, CUDA, DirectML).
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any

import numpy as np
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = PROJECT_ROOT / "edge_opt" / "models" / "lumacam_generator.onnx"


def benchmark_onnx_model(
    model_path: Path,
    num_runs: int = 50,
    warmup_runs: int = 10,
    height: int = 256,
    width: int = 256,
) -> Dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    # Initialize ONNX Runtime session
    providers = ort.get_available_providers()
    session = ort.InferenceSession(str(model_path), providers=providers)
    active_providers = session.get_providers()

    print(f"[ONNX Benchmark] Model: {model_path.name} ({model_path.stat().st_size / (1024*1024):.2f} MB)")
    print(f"[ONNX Benchmark] Active Providers: {active_providers}")

    # Prepare dummy inputs matching contract: input_bayer: (1, 4, H, W), alpha_exposure: (1, 1)
    input_bayer = np.random.uniform(0.0, 1.0, size=(1, 4, height, width)).astype(np.float32)
    alpha_exposure = np.array([[100.0]], dtype=np.float32)
    inputs = {"input_bayer": input_bayer, "alpha_exposure": alpha_exposure}

    # Warmup
    print(f"[ONNX Benchmark] Running {warmup_runs} warmup iterations...")
    for _ in range(warmup_runs):
        _ = session.run(None, inputs)

    # Benchmark loop
    print(f"[ONNX Benchmark] Timing {num_runs} timed iterations...")
    latencies_ms = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = session.run(None, inputs)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    mean_lat = float(np.mean(latencies_ms))
    median_lat = float(np.median(latencies_ms))
    p95_lat = float(np.percentile(latencies_ms, 95))
    fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0

    print("\n" + "=" * 60)
    print("             ONNX RUNTIME BENCHMARK RESULTS                 ")
    print("=" * 60)
    print(f"Model File:          {model_path.name}")
    print(f"Model Size:          {model_path.stat().st_size / (1024*1024):.2f} MB")
    print(f"Input Resolution:    {width*2}x{height*2} (Packed: {width}x{height}x4)")
    print(f"Execution Provider:  {active_providers[0]}")
    print(f"Mean Latency:        {mean_lat:.2f} ms")
    print(f"Median Latency:      {median_lat:.2f} ms")
    print(f"P95 Latency:         {p95_lat:.2f} ms")
    print(f"Throughput:          {fps:.1f} FPS")
    print("=" * 60)

    return {
        "model_name": model_path.name,
        "size_mb": model_path.stat().st_size / (1024 * 1024),
        "mean_latency_ms": mean_lat,
        "median_latency_ms": median_lat,
        "p95_latency_ms": p95_lat,
        "fps": fps,
        "provider": active_providers[0],
    }


def main():
    parser = argparse.ArgumentParser(description="LumaCam ONNX Benchmark")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL), help="Path to ONNX model")
    parser.add_argument("--runs", type=int, default=50, help="Number of timed runs")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup runs")
    args = parser.parse_args()

    benchmark_onnx_model(Path(args.model), num_runs=args.runs, warmup_runs=args.warmup)


if __name__ == "__main__":
    main()
