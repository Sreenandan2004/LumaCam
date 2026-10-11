"""
LumaCam ONNX Exporter
=====================
Exports the trained ExposureConditionedAttentionUNet to an ONNX graph for edge deployment
(ONNX -> LiteRT / TFLite -> INT8 quantization, see edge_opt/).

Background (Edge Deployment):
ONNX export records ONE forward pass of the network as a static graph of tensor operations.
Only tensors become graph outputs (the tracer silently drops the generator's `None` attention-map
slot), and every shape-dependent Python branch is frozen at the traced shape.
This script therefore:
1. Wraps the generator so the graph contract is explicit (exactly two inputs and one output)
   instead of depending on how a given PyTorch version flattens `(out, None)`:
       input_bayer    (B, 4, H/2, W/2) float32, amplified packed Bayer in [0, 1]
       alpha_exposure (B, 1)           float32, exposure ratio (training range ~100-300)
       enhanced_rgb   (B, 3, H, W)     float32, gamma-encoded sRGB in [0, 1]
2. Exports a FIXED input size by default. The Attention Gates resize decoder features with
   F.interpolate; at a fixed size those resizes become constants, which LiteRT conversion and
   INT8 quantization handle far more reliably. Use --dynamic for desktop evaluation graphs.
   The TracerWarning from attention_gate.py (`if g.shape[2:] != x.shape[2:]`) is benign: in this
   generator every gating signal comes from one level deeper (half resolution), so the branch is
   True for every valid input size and freezing it is correct.
3. Rejects spatial sizes the U-Net cannot process: 4 MaxPool(2) stages require packed H and W
   to be divisible by 2**4 = 16.
4. Verifies the ONNX graph (onnx.checker) and its numerical parity with PyTorch (ONNX Runtime).

The full input/output contract is documented in docs/api_specifications.md.

Usage:
    # Export a trained checkpoint (from model/train.py or the Kaggle notebook)
    python model/export_onnx.py --checkpoint checkpoints/lumacam_best.pt \
                                --output edge_opt/models/lumacam_fp32.onnx \
                                --height 256 --width 256 --test

    # Pipeline smoke test before a trained checkpoint exists
    python model/export_onnx.py --random-init --test
"""

import argparse
import inspect
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# Allow `python model/export_onnx.py` from the repository root (same convention as model/train.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.networks.generator import ExposureConditionedAttentionUNet

INPUT_NAMES = ("input_bayer", "alpha_exposure")
OUTPUT_NAME = "enhanced_rgb"
DEFAULT_OPSET = 17          # Opset 17+ exports PixelShuffle (DepthToSpace) and LeakyReLU cleanly
DEFAULT_ALPHA = 100.0       # Inside the SID training range of alpha (~100-300)
NUM_POOL_STAGES = 4         # Encoder MaxPool(2) stages in ExposureConditionedAttentionUNet
SPATIAL_DIVISOR = 2 ** NUM_POOL_STAGES  # Packed H and W must be multiples of 16
DEFAULT_ATOL = 1e-4

# Generator hyperparameters used when a checkpoint carries no config (matches train_config.yaml)
DEFAULT_MODEL_CONFIG: Dict[str, Any] = {
    "in_channels": 4,
    "out_channels": 3,
    "base_channels": 32,
    "cond_dim": 1,
    "use_pixel_shuffle": True,
}


class GeneratorExportWrapper(nn.Module):
    """
    Export-only view of the generator: two tensor inputs, one tensor output.

    The training-time forward returns `(out, attention_maps_or_None)`; the deployed graph only
    needs `out`, so the optional attention maps are dropped here instead of changing the generator.
    """

    def __init__(self, generator: ExposureConditionedAttentionUNet):
        super().__init__()
        self.generator = generator

    def forward(self, input_bayer: torch.Tensor, alpha_exposure: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_bayer: (B, 4, H/2, W/2) amplified packed Bayer.
            alpha_exposure: (B, 1) exposure conditioning scalar.
        Returns:
            (B, 3, H, W) enhanced sRGB in [0, 1].
        """
        enhanced, _ = self.generator(input_bayer, alpha_exposure)
        return enhanced


def validate_input_size(height: int, width: int) -> None:
    """
    Raises ValueError unless the packed Bayer size can pass through all U-Net pooling stages.

    Args:
        height: Packed Bayer height (H/2 of the full-resolution image).
        width: Packed Bayer width (W/2 of the full-resolution image).
    """
    for name, value in (("height", height), ("width", width)):
        if value <= 0 or value % SPATIAL_DIVISOR != 0:
            raise ValueError(
                f"Packed Bayer {name}={value} must be a positive multiple of {SPATIAL_DIVISOR} "
                f"({NUM_POOL_STAGES} MaxPool(2) stages). Pad or crop the input on-device."
            )


def load_generator(
    checkpoint_path: Optional[str],
    device: torch.device = torch.device("cpu"),
) -> ExposureConditionedAttentionUNet:
    """
    Builds the generator and loads trained weights.

    Supported checkpoint formats:
      - model/train.py:   {"generator_state_dict": ..., "config": {"model": {...}}, ...}
      - Kaggle notebook:  {"generator_state_dict": ..., "config": {...} (newer) or no config}
      - Bare state_dict:  {"enc1.conv.0.weight": ..., ...}

    Args:
        checkpoint_path: Path to a .pt/.pth checkpoint, or None for randomly initialized weights
            (only useful for pipeline smoke tests; the caller must opt in explicitly).
        device: Device to load the model onto.

    Returns:
        Generator in eval mode.

    Raises:
        FileNotFoundError: If checkpoint_path does not exist.
        KeyError: If the checkpoint has no recognizable generator weights.
    """
    model_config = dict(DEFAULT_MODEL_CONFIG)
    state_dict: Optional[Dict[str, torch.Tensor]] = None

    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: '{path}'")
        # weights_only=False: train.py checkpoints also store the YAML config dict
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if "generator_state_dict" in checkpoint:
            state_dict = checkpoint["generator_state_dict"]
            model_config.update(checkpoint.get("config", {}).get("model", {}))
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            state_dict = checkpoint
        else:
            raise KeyError(
                f"'{path}' has no 'generator_state_dict'/'state_dict' and is not a bare state_dict. "
                f"Top-level keys: {sorted(checkpoint.keys())}"
            )

    generator = ExposureConditionedAttentionUNet(
        in_channels=model_config["in_channels"],
        out_channels=model_config["out_channels"],
        base_channels=model_config["base_channels"],
        cond_dim=model_config["cond_dim"],
        use_pixel_shuffle=model_config["use_pixel_shuffle"],
    )
    if state_dict is not None:
        generator.load_state_dict(state_dict)  # strict=True: a mismatched architecture must fail
    return generator.to(device).eval()


def _make_dummy_inputs(
    input_size: Tuple[int, int], batch_size: int = 1, alpha: float = DEFAULT_ALPHA, seed: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Realistic inputs: amplified Bayer in [0, 1] (as after clipping) and an in-range alpha."""
    generator = torch.Generator().manual_seed(seed)
    bayer = torch.rand(batch_size, 4, input_size[0], input_size[1], generator=generator)
    alpha_tensor = torch.full((batch_size, 1), float(alpha))
    return bayer, alpha_tensor


def export_to_onnx(
    model: ExposureConditionedAttentionUNet,
    output_path: str,
    input_size: Tuple[int, int] = (256, 256),
    dynamic: bool = False,
    opset_version: int = DEFAULT_OPSET,
) -> Path:
    """
    Exports the generator to ONNX and validates the resulting graph.

    Args:
        model: Generator (any mode; it is switched to eval for export).
        output_path: Destination .onnx path (parent directories are created).
        input_size: Packed Bayer (H/2, W/2); e.g. (256, 256) -> 512x512 sRGB output.
        dynamic: If True, batch and spatial axes are symbolic (desktop use). If False, all shapes
            are fixed, which is what the LiteRT / INT8 pipeline expects.
        opset_version: ONNX opset (>= 17).

    Returns:
        Path of the written .onnx file.
    """
    import onnx  # Imported lazily so the module stays importable without the ONNX toolchain

    validate_input_size(*input_size)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wrapper = GeneratorExportWrapper(model).eval()
    dummy_bayer, dummy_alpha = _make_dummy_inputs(input_size)

    dynamic_axes = None
    if dynamic:
        dynamic_axes = {
            INPUT_NAMES[0]: {0: "batch_size", 2: "height", 3: "width"},
            INPUT_NAMES[1]: {0: "batch_size"},
            OUTPUT_NAME: {0: "batch_size", 2: "out_height", 3: "out_width"},
        }

    export_kwargs: Dict[str, Any] = {}
    # Newer PyTorch releases default to the dynamo-based exporter; pin the TorchScript exporter so
    # the graph (and the dynamic_axes semantics above) is identical across PyTorch versions.
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        torch.onnx.export(
            wrapper,
            (dummy_bayer, dummy_alpha),
            str(out_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=list(INPUT_NAMES),
            output_names=[OUTPUT_NAME],
            dynamic_axes=dynamic_axes,
            **export_kwargs,
        )

    onnx.checker.check_model(onnx.load(str(out_path)))
    return out_path


def verify_numerical_parity(
    model: ExposureConditionedAttentionUNet,
    onnx_path: str,
    input_size: Tuple[int, int] = (256, 256),
    alpha: float = DEFAULT_ALPHA,
    batch_size: int = 1,
) -> float:
    """
    Runs identical inputs through PyTorch and ONNX Runtime and returns the max absolute difference.

    Args:
        model: The generator that was exported.
        onnx_path: Path of the exported .onnx graph.
        input_size: Packed Bayer (H/2, W/2) to test (must match the export for fixed-shape graphs).
        alpha: Exposure conditioning value to test.
        batch_size: Batch size to test (must be 1 for fixed-shape graphs).

    Returns:
        max |torch_output - onnx_output| over all elements.
    """
    import onnxruntime as ort

    validate_input_size(*input_size)
    bayer, alpha_tensor = _make_dummy_inputs(input_size, batch_size=batch_size, alpha=alpha, seed=1)

    wrapper = GeneratorExportWrapper(model).eval()
    with torch.no_grad():
        torch_out = wrapper(bayer, alpha_tensor).cpu().numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = session.run(
        [OUTPUT_NAME], {INPUT_NAMES[0]: bayer.numpy(), INPUT_NAMES[1]: alpha_tensor.numpy()}
    )[0]

    if ort_out.shape != torch_out.shape:
        raise RuntimeError(f"Shape mismatch: PyTorch {torch_out.shape} vs ONNX {ort_out.shape}")
    return float(np.max(np.abs(torch_out - ort_out)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the LumaCam generator to ONNX")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=str, help="Trained checkpoint (.pt/.pth)")
    source.add_argument(
        "--random-init", action="store_true", help="Export untrained weights (pipeline smoke test only)"
    )
    parser.add_argument("--output", type=str, default="edge_opt/models/lumacam_fp32.onnx")
    parser.add_argument("--height", type=int, default=256, help="Packed Bayer height (H/2)")
    parser.add_argument("--width", type=int, default=256, help="Packed Bayer width (W/2)")
    parser.add_argument("--dynamic", action="store_true", help="Symbolic batch/spatial axes")
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    parser.add_argument("--test", action="store_true", help="Verify parity with ONNX Runtime")
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL, help="Parity tolerance")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_size = (args.height, args.width)
    try:
        validate_input_size(*input_size)
        model = load_generator(None if args.random_init else args.checkpoint)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"[LumaCam Export] ERROR: {exc}")
        return 1

    if args.random_init:
        print("[LumaCam Export] WARNING: exporting RANDOM weights (smoke test only, not for deployment).")

    onnx_path = export_to_onnx(
        model, args.output, input_size=input_size, dynamic=args.dynamic, opset_version=args.opset
    )
    size_mb = onnx_path.stat().st_size / (1024 ** 2)
    print(f"[LumaCam Export] Wrote {onnx_path} ({size_mb:.2f} MB, opset {args.opset}, "
          f"{'dynamic' if args.dynamic else 'fixed'} shape)")
    print(f"[LumaCam Export] Inputs: {INPUT_NAMES[0]} (1, 4, {args.height}, {args.width}), "
          f"{INPUT_NAMES[1]} (1, 1) | Output: {OUTPUT_NAME} (1, 3, {args.height * 2}, {args.width * 2})")
    print("[LumaCam Export] onnx.checker passed.")

    if args.test:
        max_diff = verify_numerical_parity(model, str(onnx_path), input_size=input_size)
        status = "PASS" if max_diff < args.atol else "FAIL"
        print(f"[LumaCam Export] Parity vs PyTorch: max |diff| = {max_diff:.3e} (atol {args.atol:.0e}) -> {status}")
        if max_diff >= args.atol:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
