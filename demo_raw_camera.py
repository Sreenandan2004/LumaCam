"""
LumaCam Interactive Raw Camera Demonstration & Simulator
========================================================
Demonstrates the full raw Bayer computational photography pipeline:
1. Simulates physical Bayer CFA sensor acquisition (RGGB mosaic + Poisson-Gaussian noise).
2. Performs sensor black-level calibration and 4-channel Bayer packing.
3. Modulates exposure dynamically via the FiLM bottleneck conditioning vector (alpha).
4. Restores full-resolution, high-fidelity sRGB through the Attention U-Net Generator.

Usage:
    # Run synthetic sensor simulation and save visual comparison
    python demo_raw_camera.py --synthetic --exposure 150.0 --output results/demo_synthetic.png

    # Process an existing test image as if captured by a dark raw sensor
    python demo_raw_camera.py --input-image path/to/dark.jpg --exposure 200.0 --output results/enhanced.png

    # Live webcam demo (if webcam is available)
    python demo_raw_camera.py --webcam --exposure 100.0
"""

import sys
import argparse
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.networks.generator import ExposureConditionedAttentionUNet
from model.datasets.raw_processing import pack_raw_bayer_np, pack_raw_bayer_torch


def simulate_bayer_sensor(
    rgb_image: np.ndarray,
    exposure_fraction: float = 0.01,
    read_noise_std: float = 0.005,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulates physical CMOS camera sensor capture:
    1. Linearizes sRGB via inverse gamma (~2.2).
    2. Underexposes scene according to exposure_fraction (simulating extreme darkness).
    3. Samples 2D Bayer CFA (RGGB mosaic).
    4. Injects physical sensor shot noise (Poisson) and read noise (Gaussian).

    Args:
        rgb_image: (H, W, 3) image in [0, 1].
        exposure_fraction: Relative exposure time (e.g. 0.01 for 1/100th normal light).
        read_noise_std: Standard deviation of sensor readout amplifier noise.

    Returns:
        bayer_raw: 2D array (H, W) raw sensor readings in 14-bit integer range [0, 16383].
        clean_linear: 3D array (H, W, 3) linear RGB ground truth.
    """
    h, w, c = rgb_image.shape
    # Crop to even dimensions if necessary
    h_even = (h // 2) * 2
    w_even = (w // 2) * 2
    rgb_image = rgb_image[:h_even, :w_even, :]

    # Inverse gamma transformation to linear sensor photon space
    linear_rgb = np.power(np.maximum(rgb_image, 1e-6), 2.2)

    # Underexpose
    dark_linear = linear_rgb * exposure_fraction

    # Apply 2x2 Bayer CFA filter array (RGGB pattern):
    # [ R   G1 ]
    # [ G2  B  ]
    bayer_pattern = np.zeros((h_even, w_even), dtype=np.float32)
    bayer_pattern[0::2, 0::2] = dark_linear[0::2, 0::2, 0]  # Red
    bayer_pattern[0::2, 1::2] = dark_linear[0::2, 1::2, 1]  # Green 1
    bayer_pattern[1::2, 0::2] = dark_linear[1::2, 0::2, 1]  # Green 2
    bayer_pattern[1::2, 1::2] = dark_linear[1::2, 1::2, 2]  # Blue

    # Add realistic photon shot noise (Poisson approximation) + thermal read noise
    shot_std = np.sqrt(np.maximum(bayer_pattern, 0.0) / 1000.0)
    noisy_bayer = bayer_pattern + np.random.normal(0, shot_std) + np.random.normal(0, read_noise_std, size=bayer_pattern.shape)
    noisy_bayer = np.clip(noisy_bayer, 0.0, 1.0)

    # Scale to 14-bit ADC readings with Sony 512 black pedestal: [512, 16383]
    sensor_raw = 512.0 + noisy_bayer * (16383.0 - 512.0)
    return sensor_raw.astype(np.float32), linear_rgb


def run_pipeline(
    raw_2d: np.ndarray,
    alpha: float,
    model: ExposureConditionedAttentionUNet,
    device: torch.device,
) -> np.ndarray:
    """Runs sensor raw 2D array through Bayer packing, amplification, and Attention U-Net."""
    # 1. Pack Bayer: (H, W) -> (H/2, W/2, 4)
    packed_np = pack_raw_bayer_np(raw_2d, black_level=512.0, white_level=16383.0, bayer_pattern="RGGB")
    
    # 2. Analytical amplification: x' = x * alpha
    amplified_np = packed_np * alpha

    # 3. Format tensor: (1, 4, H/2, W/2)
    bayer_tensor = torch.from_numpy(amplified_np.transpose((2, 0, 1))).unsqueeze(0).float().to(device)
    alpha_tensor = torch.tensor([[alpha]], dtype=torch.float32, device=device)

    # 4. Neural inference
    with torch.no_grad():
        enhanced_rgb, _ = model(bayer_tensor, alpha_tensor)
        enhanced_rgb = torch.clamp(enhanced_rgb, 0.0, 1.0)

    # Convert back to (H, W, 3) numpy array
    out_np = enhanced_rgb.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return out_np


def main():
    parser = argparse.ArgumentParser(description="LumaCam Bayer Raw Camera Demo & Simulator")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic test pattern")
    parser.add_argument("--input-image", type=str, default=None, help="Input test image file")
    parser.add_argument("--webcam", action="store_true", help="Capture single frame from connected webcam")
    parser.add_argument("--exposure", type=float, default=100.0, help="Exposure amplification alpha (default: 100.0)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--output", type=str, default="demo_result.png", help="Output comparison path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[LumaCam Demo] Device: {device} | Exposure Factor alpha: {args.exposure:.1f}x")

    # Initialize Model
    model = ExposureConditionedAttentionUNet(base_channels=32).to(device)
    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"[LumaCam Demo] Loading weights from {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        state_dict = ckpt["generator_state_dict"] if "generator_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
    model.eval()

    # Acquire image
    if args.input_image and Path(args.input_image).exists():
        try:
            from PIL import Image
            img = Image.open(args.input_image).convert("RGB")
            rgb_in = np.array(img, dtype=np.float32) / 255.0
        except ImportError:
            import cv2
            img = cv2.imread(args.input_image)
            rgb_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    elif args.webcam:
        try:
            import cv2
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                raise RuntimeError("Failed to capture frame from webcam.")
            rgb_in = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        except Exception as e:
            print(f"[LumaCam Demo] Webcam capture failed ({e}). Falling back to synthetic pattern.")
            args.synthetic = True

    if args.synthetic or (not args.input_image and not args.webcam):
        # Generate colorful synthetic scene (512x512)
        print("[LumaCam Demo] Generating synthetic test pattern (512x512)...")
        y, x = np.mgrid[0:512, 0:512]
        r = 0.5 + 0.4 * np.sin(x / 30.0) * np.cos(y / 40.0)
        g = 0.5 + 0.4 * np.sin(y / 25.0)
        b = 0.5 + 0.4 * np.cos(x / 35.0)
        rgb_in = np.stack([r, g, b], axis=-1).astype(np.float32)

    # Crop to maximum 512x512 for fast demonstration
    h, w, _ = rgb_in.shape
    crop_h = min(512, (h // 2) * 2)
    crop_w = min(512, (w // 2) * 2)
    rgb_in = rgb_in[:crop_h, :crop_w, :]

    # 1. Simulate underexposed Bayer raw sensor
    exposure_frac = 1.0 / max(args.exposure, 1.0)
    raw_2d, _ = simulate_bayer_sensor(rgb_in, exposure_fraction=exposure_frac)

    # 2. Run through LumaCam Neural Pipeline
    enhanced_rgb = run_pipeline(raw_2d, args.exposure, model, device)

    # 3. Create simple bilinear demosaic of noisy raw for comparison
    norm_raw = np.clip((raw_2d - 512.0) / (16383.0 - 512.0) * args.exposure, 0.0, 1.0)
    raw_view = np.repeat(norm_raw[:, :, np.newaxis], 3, axis=-1)

    # Combine side-by-side: [Dark / Noisy Raw Preview | LumaCam Enhanced]
    comparison = np.hstack([raw_view, enhanced_rgb])
    comparison_uint8 = (np.clip(comparison, 0.0, 1.0) * 255.0).astype(np.uint8)

    # Save output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
        Image.fromarray(comparison_uint8).save(str(out_path))
        print(f"[SUCCESS] Demo completed! Comparison saved to: {out_path.resolve()}")
    except ImportError:
        import cv2
        cv2.imwrite(str(out_path), cv2.cvtColor(comparison_uint8, cv2.COLOR_RGB2BGR))
        print(f"[SUCCESS] Demo completed! Comparison saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
