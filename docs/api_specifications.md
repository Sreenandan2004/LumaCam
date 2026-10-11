# LumaCam API & Data Contract Specification

> **Owner:** Sreelakshmi A (docs). **Last updated:** 2026-10-12.
> This is the single source of truth for the tensors exchanged between the dataset, the model, the
> ONNX/LiteRT artifacts, and the Android app. Any change here is a **cross-team contract change**:
> record it in [`CHANGELOG.md`](../CHANGELOG.md) and get sign-off from the affected owners first
> ([AGENTS.md § 0](../AGENTS.md)).

---

## 1. End-to-End Data Flow

```
Bayer RAW (H, W) uint16
  └─ pack_raw_bayer_*          → (H/2, W/2, 4) float32 in [0, 1]   (black/white level normalized)
       └─ × alpha, min(·, 1.0) → input_bayer (B, 4, H/2, W/2)     (AMPLIFIED_INPUT_CEILING)
            └─ Generator(input_bayer, alpha_exposure)
                 └─ enhanced_rgb (B, 3, H, W) gamma-encoded sRGB in [0, 1]
                      └─ apply_color_pipeline (optional) → display / JPEG
```

---

## 2. Model Inputs

### 2.1 `input_bayer` — amplified packed Bayer

| Property | Value |
|---|---|
| Shape | `(B, 4, H/2, W/2)`, NCHW, float32 |
| Channel order | `[R, G1, B, G2]`. Note: channel 2 is **Blue** and channel 3 is the second green (`model/datasets/raw_processing.py`) |
| Normalization | `max(raw - black_level, 0) / (white_level - black_level)` |
| Amplification | `min(packed × alpha, 1.0)`, where the 1.0 ceiling is `AMPLIFIED_INPUT_CEILING` |
| Value range | `[0, 1]` |
| Spatial constraint | `H/2` and `W/2` must be multiples of **16** (4 MaxPool(2) stages). Pad or crop on device. |
| Bayer phase | Crops must start at **even** `(row, col)` in the RAW mosaic, or R/G/B channels get swapped |

**Sensor constants:**
- Training uses Sony α7S II values: `black_level = 512`, `white_level = 16383` (14-bit), RGGB.
- On Android, read the deployment sensor's values from Camera2 `CameraCharacteristics`:
  - `SENSOR_BLACK_LEVEL_PATTERN` / `CaptureResult.SENSOR_DYNAMIC_BLACK_LEVEL`
  - `SENSOR_INFO_WHITE_LEVEL`
  - `SENSOR_INFO_COLOR_FILTER_ARRANGEMENT`

  Do not hard-code the Sony values on the phone. `pack_raw_bayer_np` currently supports only RGGB and BGGR.

### 2.2 `alpha_exposure` — exposure conditioning

| Property | Value |
|---|---|
| Shape | `(B, 1)` float32 |
| Current meaning | Exposure ratio `t_long / t_short` (SID Sony ratios are 100, 250, 300) |
| Trained range | ~100–300. Values outside this range are **out of distribution** for the FiLM MLP. |
| Used twice | It multiplies the input (§ 2.1) **and** conditions the bottleneck through FiLM. |

> ⚠️ See **OD-1** below. The app spec (`veni.md`) currently maps the EV slider to α ∈ [0.5, 3.0],
> which does not match this contract.

---

## 3. Model Output — `enhanced_rgb`

| Property | Value |
|---|---|
| Shape | `(B, 3, H, W)`, NCHW, float32 (PixelShuffle(2) doubles `H/2, W/2`) |
| Value range | `[0, 1]` (Sigmoid head) |
| Color encoding | **Gamma-encoded sRGB with camera white balance already applied.** Training targets come from `rawpy.postprocess(use_camera_wb=True, no_auto_bright=True)`, which applies a BT.709-style ~1/2.2 gamma. |
| Do **not** | Apply another `I^(1/2.2)` gamma (that would double-encode and wash out the image). |

---

## 4. Checkpoint Format (`*.pt`)

Produced by `model/train.py` and `notebooks/train_lumacam_kaggle.ipynb`:

```python
{
    "generator_state_dict": OrderedDict,      # required
    "config": {"model": {"in_channels": 4, "out_channels": 3, "base_channels": 32,
                         "cond_dim": 1, "use_pixel_shuffle": True}},   # used to rebuild the generator
    "epoch": int, "val_psnr": float, "val_ssim": float,             # metadata
    # train.py also stores discriminator / optimizer states for --resume
}
```

`model/export_onnx.py::load_generator` also accepts `{"state_dict": ...}` and bare state dicts.
If no config is stored, it uses the defaults above. Checkpoints are loaded with
`weights_only=False` (they contain the config dict), so **only load checkpoints from trusted sources**.

---

## 5. ONNX Artifact (`model/export_onnx.py`)

| Property | Value |
|---|---|
| Inputs | `input_bayer` (§ 2.1), `alpha_exposure` (§ 2.2) |
| Output | `enhanced_rgb` (§ 3) |
| Opset | 17 (TorchScript exporter, pinned with `dynamo=False`) |
| Default shape | **Fixed** `(1, 4, 256, 256)` → `(1, 3, 512, 512)`. Fixed shapes turn the Attention Gate resizes into constants, which is what LiteRT conversion and INT8 PTQ need. |
| Dynamic option | `--dynamic` → symbolic batch/height/width (desktop evaluation only) |
| Default path | `edge_opt/models/lumacam_fp32.onnx` |
| Verification | `onnx.checker` + PyTorch↔ONNX Runtime parity, max \|diff\| < 1e-4 (`--test`) |

```bash
python model/export_onnx.py --checkpoint checkpoints/lumacam_best.pt --test
python model/export_onnx.py --random-init --test        # smoke test without trained weights
```

---

## 6. Post-Processing (`model/postprocess/color_science.py`)

`apply_color_pipeline(img, awb_strength=0.5, saturation=1.15, input_is_linear=False)`

| Step | Space | Default |
|---|---|---|
| sRGB → linear decode | IEC 61966-2-1 | Only when AWB runs. Skipped otherwise. |
| Gray-World AWB | **linear** | Strength 0.5, gains capped to [0.5, 2], clipped pixels (≥ 0.98) ignored |
| linear → sRGB encode | IEC 61966-2-1 | Once. Linear inputs (`input_is_linear=True`) are always encoded. |
| Saturation | display sRGB | ×1.15, hue- and value-preserving, gamut-safe |

With `awb_strength=0` and `saturation=1`, sRGB input passes through unchanged (tested).
NumPy `(H, W, 3)` callers use `apply_color_pipeline_np`.

---

## 7. Open Decisions (need owner sign-off — not implemented)

| ID | Decision | Owners | Notes / proposal |
|---|---|---|---|
| **OD-1** | **Units and range of `alpha_exposure` from the app.** The model is trained on α ≈ 100–300, but `veni.md` maps the EV slider to α ∈ [0.5, 3.0]. | Nandan, Veni | **Short term (no retraining):** compute α in the ratio domain on device, e.g. `alpha = alpha_base × 2^EV`, where `alpha_base` comes from metering (target exposure / actual exposure) and is clamped to [100, 300]. **At the next retrain:** feed FiLM a normalized value such as `log2(alpha) / log2(300)`, so the conditioning MLP sees inputs of order 1 instead of raw 100–300. That change alters the checkpoint semantics and needs a coordinated retrain plus a re-export. |
| **OD-2** | **α is applied twice** (input amplification + FiLM). | Nandan | This is intentional per `sid_dataset.py`, but it means the slider changes the input brightness and the latent modulation together. Decide whether FiLM should instead receive a *residual* exposure offset (EV) independent of the base amplification. |
| **OD-3** | **Owner of INT8 quantization (Task 7) and edge benchmarks (Task 8).** `TEAM_PLAN.md` / `me.md` say Nandan (Task 7) and Veni (Task 8 = `evaluate.py`). `Progress.md` § 4 lists both under Lakshmi. | Nandan | Confirm and update the handoff files. |
| **OD-4** | **Model binaries vs `.gitignore`.** AGENTS.md § 6 says to track `.onnx`/`.tflite` with Git LFS, but `.gitignore` ignores `*.onnx` and `*.tflite`. | Nandan | Either add `git lfs track "*.onnx" "*.tflite"` with a `.gitignore` exception for `edge_opt/models/`, or publish artifacts as GitHub release assets. |
| **OD-5** | **Ground-truth bit depth.** The loader uses `output_bps=8` (the spec said 16). 8-bit targets quantize smooth gradients in dark regions. | Nandan | Switch to `output_bps=16` / 65535 at the next retrain if PSNR on gradients matters. |
