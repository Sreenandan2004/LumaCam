# LumaCam: Edge-Native Adaptive Low-Light Computational Photography
> Final Year Capstone Project — Team 21 (SCTCE, Guided by Asst. Prof. Arya Sreedhar)
> Team Members: Sreelakshmi A (262), Sreenandan S (263), Sreeveni R (264)

---

## 0. Master Rules (Mandatory for Every Change)

These rules override every other section. They apply to all team members and all AI coding assistants.

1. **Log every change in [`CHANGELOG.md`](CHANGELOG.md)**, in the same branch and commit as the change.
   Add an entry under `[Unreleased]`. Each entry states:
   - **What** changed and **Why**: the problem, with the evidence.
   - **Files** touched.
   - **Verification**: the exact command and its result.
   - **Impact / Follow-ups**: behavior changes, retraining needs, and owners to notify.
   A change without a log entry is incomplete and must not be merged.
2. **Update the other affected documents in the same branch:**
   - [`Progress.md`](Progress.md): milestone or task status changes.
   - [`docs/api_specifications.md`](docs/api_specifications.md): any change to a cross-team contract (tensor names,
     shapes, value ranges, α semantics, checkpoint format, ONNX/LiteRT I/O).
   - Member handoff files (`me.md`, `veni.md`, `lakshmi.md`): when task scope or status changes.
   - This file's Section 5 (Repository File Structure): when files or folders are added.
3. **One logical change per branch.** Use branch sequences for dependent changes (Section 6.2).
4. **Follow software engineering principles:**
   - **Tests with every behavior change.** Add or extend tests in `tests/unit/` that fail before the fix and pass after it.
     Run `python -m pytest tests/unit -v` before committing.
   - **Fail fast and loudly.** Raise clear exceptions instead of silent defaults or fallbacks that hide bad data.
   - **Single source of truth.** Shared constants (e.g. `AMPLIFIED_INPUT_CEILING`, `SID_SPLIT_PREFIXES`) are defined once and imported everywhere else.
   - **Minimal, focused diffs.** No drive-by refactors. Preserve file formatting (notebooks: edit sources, keep JSON layout).
   - **Backward compatibility.** Any break in behavior or format is called out in the change log under *Impact*.
   - **Readable code.** Type hints (PEP 484), docstrings with tensor shapes, and comments that explain *why*.
5. **Never change a cross-team contract on your own.** Record the proposal under
   `Open Decisions` in `CHANGELOG.md` and in `docs/api_specifications.md`, and get the owner's sign-off first.

---

## 1. Project Overview & Mission

**LumaCam** is a real-time, edge-native low-light image enhancement mobile camera application inspired by the Google Pixel GCam interface. It bridges the critical efficiency-quality gap in low-light computational photography by combining:
1. **Exposure-Conditioned Attention U-Net GAN**: A single-pass generator with Attention Gates and dynamic user exposure latent injection at the bottleneck.
2. **Edge Quantization & Acceleration**: Post-Training INT8 Quantization (PTQ) executed via LiteRT (TensorFlow Lite) with GPU/NPU delegates.
3. **High-Performance Camera Pipeline**: Native Android Camera2 integration bridged with Flutter for zero-copy/low-latency viewfinder streaming (~30 FPS target) and full-fidelity image capture.
4. **Post-Inference Color Science**: Automated white balance, tonal mapping, and saturation correction to produce vibrant, professional-grade sRGB imagery.

---

## 2. System Architecture

```
                  ┌──────────────────────────────────────────────┐
                  │            Android Camera2 Sensor            │
                  │   (Preview Stream: YUV420 / RAW Sensor)      │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │       Native C++ / Kotlin Preprocessing      │
                  │ (Bayer Packing / Normalization / Amplification)│
                  └──────────────────────┬───────────────────────┘
                                         │
       User Exposure Slider ─────────────┼──────────────┐
       (Conditioning Vector α)           │              │
                                         ▼              ▼
                  ┌──────────────────────────────────────────────┐
                  │            LiteRT Inference Engine           │
                  │  (Quantized INT8 Exposure-Conditioned U-Net) │
                  │     Hardware Delegates: NPU (NNAPI) / GPU    │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │            Color & Tonal Correction          │
                  │ (Auto-White Balance, Saturation, Tone Curve) │
                  └──────────────┬────────────────┬──────────────┘
                                 │                │
            Viewfinder Stream    ▼                ▼    Full Capture
         ┌──────────────────────────────┐  ┌─────────────────────────────┐
         │     Flutter GCam-Style UI    │  │       Storage Service       │
         │ (Texture Widget / 30 FPS Live│  │ (DNG / JPEG Gallery Export) │
         └──────────────────────────────┘  └─────────────────────────────┘
```

---

## 3. Tech Stack Matrix

| Layer | Technologies | Key Responsibilities |
|---|---|---|
| **Deep Learning** | PyTorch, PyTorch Lightning, TorchVision | Model definition (Attention U-Net + PatchGAN), Conditioning Bottleneck, Loss functions (L1, MS-SSIM, cGAN) |
| **Model Optimization** | ONNX, LiteRT (TFLite Converter), Olive | Graph optimization, INT8 Post-Training Quantization (PTQ), representative dataset calibration |
| **Mobile Frontend** | Flutter (Dart), Flutter BLoC / Riverpod | GCam UI (Exposure sliders, viewfinder, histogram, mode selector, gallery integration) |
| **Mobile Native Core** | Android Camera2 API, Kotlin / C++ (NDK) | Zero-latency camera frame polling, Native SurfaceTexture rendering, hardware buffer handling |
| **Edge Inference** | LiteRT Android C++ / Java API | Delegate acceleration (NNAPI, Qualcomm QNN / MediaTek NeuroPilot / GPU delegate) |
| **Version Control & CI** | Git, GitHub, Git LFS | Modular code branches, checkpoint versioning, collaborative reviews |

---

## 4. Team Work Breakdown Structure (3-Member Split)

To ensure clear ownership, avoid merge conflicts, and allow parallel progress across the three team members:

### Workstream 1: Model Engineering & Training (Member 1)
* **Lead Focus**: PyTorch model development, dataset pipeline, training, and objective functions.
* **Key Tasks**:
  - Implement Bayer raw data packer: $(H, W, 1) \rightarrow (H/2, W/2, 4)$ with amplification factor $\alpha$.
  - Build Attention U-Net Generator with Attention Gates on skip connections.
  - Implement Exposure Latent Conditioning vector injection (FiLM layer or feature concatenation at bottleneck).
  - Implement PatchGAN Discriminator ($70 \times 70$ receptive field).
  - Train on See-in-the-Dark (SID) Sony dataset with composite loss ($L_1 + \text{MS-SSIM} + L_{cGAN}$).
  - Benchmark PSNR, SSIM, and LPIPS against base paper metrics.

### Workstream 2: Quantization, Edge Optimization & LiteRT Runtime (Member 2)
* **Lead Focus**: Model conversion, INT8 compression, mobile profiling, and LiteRT C++/Kotlin wrapper.
* **Key Tasks**:
  - Export trained PyTorch model to ONNX with dynamic/fixed batch shapes.
  - Convert ONNX to LiteRT (`.tflite`) with INT8 quantization using representative low-light calibration data.
  - Evaluate quantization loss: compare FP32 vs INT8 PSNR degradation (target: $< 0.5$ dB drop).
  - Build lightweight Android native inference harness with hardware delegates (NNAPI, GPU, Hexagon/NPU).
  - Measure per-frame latency, memory footprint, thermal throttling, and throughput (target: $\le 33\text{ms}$ / frame).

### Workstream 3: Mobile Application & Camera2 Integration (Member 3)
* **Lead Focus**: Flutter GCam-style interface, native Android Camera2 bridge, and end-to-end integration.
* **Key Tasks**:
  - Develop Pixel GCam UI: minimalist viewfinder, bottom mode carousel, double-tap zoom, interactive exposure slider.
  - Build Android native plugin using Camera2 API with `ImageReader` / `SurfaceTexture`.
  - Connect user exposure slider to inference conditioning input in real time.
  - Implement post-processing pipeline (auto white balance, contrast stretching, gamma correction). The generator
    already outputs gamma-encoded sRGB, so apply gamma encoding only to linear data (`model/postprocess/color_science.py`).
  - Handle full-resolution photo capture, storage permissions, and photo gallery preview.

---

## 5. Repository File Structure

```text
LumaCam/
├── AGENTS.md                               # Project specification & AI Agent guidance (this file)
├── CHANGELOG.md                            # Mandatory change log (see Section 0, Master Rules)
├── Progress.md                             # Milestone status & next steps
├── README.md                               # Public project documentation & demo media
├── .gitignore                              # Comprehensive git ignore (Python, Flutter, Android, Models)
├── docs/                                   # Architectural diagrams, papers, and phase reports
│   ├── base_paper_notes.md
│   ├── phase1_deliverables.md
│   └── api_specifications.md
├── model/                                  # Workstream 1: PyTorch Model Training & Evaluation
│   ├── configs/                            # YAML training hyperparameters & paths
│   ├── datasets/                           # SID dataset loaders, Bayer raw packing utils
│   ├── networks/                           # PyTorch modules
│   │   ├── generator.py                    # Exposure-Conditioned Attention U-Net
│   │   ├── attention_gate.py               # Spatial Attention Gate module
│   │   ├── conditioning.py                 # Latent vector injection layer (FiLM / Bottleneck)
│   │   └── discriminator.py                # 70x70 PatchGAN Discriminator
│   ├── losses/                             # L1, MS-SSIM, Perceptual, Adversarial loss functions
│   ├── postprocess/                        # Post-inference color science (AWB, sRGB transfer, saturation)
│   ├── train.py                            # Main training loop
│   ├── evaluate.py                         # PSNR, SSIM, LPIPS benchmarking script
│   └── export_onnx.py                      # ONNX exporter
├── edge_opt/                               # Workstream 2: Quantization & LiteRT Deployment
│   ├── quantize_int8.py                    # INT8 PTQ converter with representative calibration set
│   ├── benchmark_tflite.py                 # Desktop & ADB mobile latency/memory profiler
│   ├── calibration_samples/                # Extracted representative low-light patches
│   └── models/                             # Exported .tflite model binaries (Git LFS)
│       ├── lumacam_fp32.tflite
│       └── lumacam_int8.tflite
├── mobile/                                 # Workstream 3: Flutter & Android Native App
│   ├── android/                            # Native Android Host
│   │   └── app/src/main/
│   │       ├── kotlin/.../camera/          # Camera2 API Controller & Frame Pipe
│   │       ├── kotlin/.../inference/       # LiteRT Native Bridge & Delegate Setup
│   │       └── cpp/                        # Optional C++ NDK fast color/raw transformations
│   ├── lib/                                # Flutter UI & State Management
│   │   ├── core/                           # Constants, themes, utilities
│   │   ├── presentation/
│   │   │   ├── screens/camera_screen.dart  # GCam Viewfinder & Overlay UI
│   │   │   ├── widgets/exposure_slider.dart# Real-time exposure adjustment slider
│   │   │   └── widgets/shutter_button.dart # Animated capture button
│   │   ├── bloc/                           # Camera & Settings BLoCs
│   │   └── main.dart                       # App entry point
│   └── pubspec.yaml                        # Flutter dependencies
└── tests/                                  # Unit & Integration Tests
    ├── unit/                               # Tensor shape verification, loss assertions
    └── mobile_tests/                       # Flutter widget & platform channel tests
```

---

## 6. Git Branching & Version Control Protocol

1. **Branch Naming Standard**:
   - `main`: Production-ready, stable releases (Phase 1, Mid-term, Final demo).
   - `dev`: Active integration branch.
   - `feature/model-<task>`: Model engineering (Member 1).
   - `feature/opt-<task>`: Quantization & edge profiling (Member 2).
   - `feature/app-<task>`: Flutter UI and Camera2 implementation (Member 3).
   - `feature/docs-<task>`: Documentation, specifications, and process changes (any member).

2. **Branch Sequences (stacked branches)**:
   - Each branch carries exactly one logical change. When changes depend on each other, create them
     as a sequence: the first branch starts from `dev`, each later branch starts from the previous one.
   - Open one PR per branch, with base `dev`, and merge them in the documented order. After a merge, rebase
     the next branch onto `dev` before merging it.
   - Record the sequence and its merge order in the `CHANGELOG.md` entry of the first branch.

3. **Git LFS Policy**:
   - Never commit raw datasets (`.ARW`, `.dng`, `.png`) or large model checkpoints (`.pth > 50MB`) to standard git.
   - Use `.gitignore` for data folders and track `.tflite` / `.onnx` models via Git LFS.

---

## 7. Instructions for AI Coding Assistants (Pair Programming)

When assisting this project:
1. **Explain the "Why"**: Always provide clear theoretical and practical intuition before introducing code snippets, as the team is learning computational photography.
2. **Prioritize Edge Efficiency**: Always check for memory leaks, buffer copies, and tensor allocation bottlenecks when writing native Android or LiteRT code.
3. **Verify Tensor Shapes**: In all Attention U-Net and Bayer packing code, explicitly document and verify tensor dimensions at every downsampling/upsampling stage.
4. **Adhere to Code Standards**: Use type annotations in Python (PEP 484), clean Dart formatting (`dart format`), and robust error handling in Android Kotlin.
5. **Enforce the Master Rules (Section 0)**: Every change you make must be logged in `CHANGELOG.md`, must update the affected docs, and must ship with tests. Do this without being asked.
