"""
LumaCam Post-Training INT8 Quantization (PTQ) Pipeline
======================================================
Quantizes the FP32 ONNX model to INT8 precision for mobile NPU/GPU deployment
(Google Pixel 7 Tensor G2, Qualcomm Snapdragon, MediaTek Dimensity).

Background (Edge AI & Computational Photography):
- FP32 models consume 4 bytes per weight parameter. In a 31+ MB model, loading and
  memory bandwidth saturate mobile bus limits, draining battery and generating heat.
- INT8 Post-Training Quantization compresses 32-bit floating point weights into 8-bit
  integers, shrinking the memory footprint by ~75% (to ~7.8 MB) and unlocking hardware
  SIMD / Tensor NPU instructions with minimal numerical degradation (< 0.5 dB PSNR drop).
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_dynamic,
    quantize_static,
    QuantType,
    CalibrationDataReader,
    CalibrationMethod,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_MODEL = PROJECT_ROOT / "edge_opt" / "models" / "lumacam_generator.onnx"
DEFAULT_OUTPUT_MODEL = PROJECT_ROOT / "edge_opt" / "models" / "lumacam_generator_int8.onnx"


class SIDCalibrationDataReader(CalibrationDataReader):
    """Feeds representative low-light Bayer raw patches for static INT8 calibration."""

    def __init__(self, num_samples: int = 30, height: int = 256, width: int = 256):
        self.num_samples = num_samples
        self.height = height
        self.width = width
        self.current_idx = 0

        # Generate representative dark Bayer distributions
        np.random.seed(42)
        self.data: List[Dict[str, np.ndarray]] = []
        for _ in range(num_samples):
            # Simulated Poisson-Gaussian sensor noise in dark range [0.001, 0.05]
            clean = np.random.uniform(0.001, 0.03, size=(1, 4, height, width)).astype(np.float32)
            noise = np.random.normal(0, 0.005, size=clean.shape).astype(np.float32)
            alpha_val = float(np.random.choice([50.0, 100.0, 200.0, 300.0]))
            
            # Amplified input as per equation (1): x' = Pack(I_raw) * alpha
            amplified = np.clip((clean + noise) * alpha_val, 0.0, 1.0).astype(np.float32)
            alpha_tensor = np.array([[alpha_val]], dtype=np.float32)

            self.data.append({
                "input_bayer": amplified,
                "alpha_exposure": alpha_tensor,
            })

    def get_next(self) -> Dict[str, np.ndarray]:
        if self.current_idx < len(self.data):
            batch = self.data[self.current_idx]
            self.current_idx += 1
            return batch
        return None

    def rewind(self):
        self.current_idx = 0


def quantize_lumacam_onnx(
    input_model_path: Path,
    output_model_path: Path,
    use_static: bool = True,
    num_calibration_samples: int = 20,
) -> Dict[str, Any]:
    if not input_model_path.exists():
        raise FileNotFoundError(f"Source FP32 model not found: {input_model_path}")

    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    fp32_size_mb = input_model_path.stat().st_size / (1024 * 1024)

    print(f"\n[Quantization] Starting INT8 Post-Training Quantization...")
    print(f"[Quantization] Input Model:  {input_model_path.name} ({fp32_size_mb:.2f} MB)")
    print(f"[Quantization] Output Model: {output_model_path.name}")

    t0 = time.perf_counter()

    if use_static:
        print(f"[Quantization] Mode: Static PTQ with {num_calibration_samples} representative calibration patches...")
        data_reader = SIDCalibrationDataReader(num_samples=num_calibration_samples)
        quantize_static(
            model_input=str(input_model_path),
            model_output=str(output_model_path),
            calibration_data_reader=data_reader,
            quant_format=QuantType.QInt8,
            calibrate_method=CalibrationMethod.MinMax,
        )
    else:
        print("[Quantization] Mode: Dynamic INT8 PTQ (Weights to QInt8)...")
        quantize_dynamic(
            model_input=str(input_model_path),
            model_output=str(output_model_path),
            weight_type=QuantType.QInt8,
        )

    t1 = time.perf_counter()
    int8_size_mb = output_model_path.stat().st_size / (1024 * 1024)
    compression_ratio = (1.0 - (int8_size_mb / fp32_size_mb)) * 100.0

    print(f"[Quantization] Completed in {t1 - t0:.2f} seconds.")
    print("\n" + "=" * 65)
    print("             INT8 QUANTIZATION COMPRESSION AUDIT             ")
    print("=" * 65)
    print(f"Original FP32 Model Size:      {fp32_size_mb:.2f} MB")
    print(f"Quantized INT8 Model Size:     {int8_size_mb:.2f} MB")
    print(f"Memory Footprint Reduction:    {compression_ratio:.1f}%")
    print(f"Mobile Hardware Target:        Google Pixel 7 (Tensor G2 NPU/GPU)")
    print("=" * 65)

    # Sanity check: Run a test inference on the INT8 model
    session = ort.InferenceSession(str(output_model_path), providers=["CPUExecutionProvider"])
    dummy_bayer = np.random.uniform(0.0, 1.0, size=(1, 4, 256, 256)).astype(np.float32)
    dummy_alpha = np.array([[100.0]], dtype=np.float32)
    out = session.run(None, {"input_bayer": dummy_bayer, "alpha_exposure": dummy_alpha})[0]

    assert out.shape == (1, 3, 512, 512), f"Expected (1, 3, 512, 512), got {out.shape}"
    print(f"[Quantization Verification] INT8 inference sanity check PASSED. Output shape: {out.shape}")

    return {
        "fp32_size_mb": fp32_size_mb,
        "int8_size_mb": int8_size_mb,
        "compression_percent": compression_ratio,
        "output_shape": list(out.shape),
    }


def main():
    parser = argparse.ArgumentParser(description="LumaCam INT8 Post-Training Quantizer")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT_MODEL), help="Path to FP32 ONNX model")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_MODEL), help="Path to save INT8 model")
    parser.add_argument("--dynamic", action="store_true", help="Use dynamic quantization instead of static QDQ")
    parser.add_argument("--samples", type=int, default=20, help="Number of calibration samples")
    args = parser.parse_args()

    quantize_lumacam_onnx(
        Path(args.input),
        Path(args.output),
        use_static=not args.dynamic,
        num_calibration_samples=args.samples,
    )


if __name__ == "__main__":
    main()
