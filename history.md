# LumaCam: Day-by-Day Capstone Execution Log
> **Project Title**: Edge-Native Adaptive Low-Light Computational Photography  
> **Institution**: SCT College of Engineering (SCTCE), Guided by Asst. Prof. Arya Sreedhar  
> **Team 21 (Batch R7A)**: Sreelakshmi A (262), Sreenandan S (263), Sreeveni R (264)  
> **Sprint Horizon**: October 5, 2026 – October 19, 2026  
> **Target Review Milestone**: Wednesday, October 14, 2026  

---

## 📅 Sprint Calendar Overview

```mermaid
timeline
    title 2-Week Capstone Engineering Roadmap
    Section Week 1: Architecture, Core Pipeline & Training
        Oct 5 (Day 1) : Literature Survey : Base Paper Review : Project Charter
        Oct 6 (Day 2) : Repository Setup : Branching Protocols : AGENTS.md
        Oct 7 (Day 3) : Raw Bayer CFA Packer : Black Pedestal Calibration
        Oct 8 (Day 4) : Attention U-Net Generator : FiLM Bottleneck : PatchGAN
        Oct 9 (Day 5) : Multi-Scale SSIM : Adversarial BCE : Composite Loss Engine
        Oct 10 (Day 6) : SID Sony Dataset Loader : Poisson-Gaussian Synthetic Pipeline
        Oct 11 (Day 7) : Unit Test Suite (10/10) : Training Engine : GTX 1650 Dry Run
    Section Week 2: Cloud Scaling, Edge Optimization & Mobile
        Oct 12 (Day 8 - Today) : Kaggle P100 Training Notebook : Technical Report & History Docs
        Oct 13 (Day 9) : Full SID Training Convergence : PSNR/SSIM Analysis : ONNX Exporter
        Oct 14 (Day 10 - Milestone) : Wednesday Faculty Review : Demo Slides : Baseline Presentation
        Oct 15 (Day 11) : LiteRT INT8 Post-Training Quantization : Calibration Dataset
        Oct 16 (Day 12) : Flutter GCam UI Viewfinder : Interactive EV Slider
        Oct 17 (Day 13) : Android Camera2 SurfaceTexture Native Bridge
        Oct 18 (Day 14) : End-to-End Edge Integration : Benchmark Profiling (FPS & Latency)
        Oct 19 (Day 15) : Final Capstone Defense : Final Demonstration
```

---

## 📝 Daily Detailed Execution Logs

### Day 1 — Monday, October 5, 2026
* **Focus**: Project Charter & Literature Review
* **Planned**:
  - Analyze computational limits of diffusion-based LLIE models (e.g., DiffLight, RetinexMamba).
  - Study base paper: Thesia & Suthar (2026) *Real-Time Low-Light Enhancement via Attention U-Net GAN*.
  - Formulate mobile edge constraint: Target $\le 33\text{ms}$ / frame (30 FPS) on Google Pixel 7.
* **Executed**:
  - Established project objective: replace multi-second diffusion steps with single-pass Exposure-Conditioned Attention U-Net GAN.
  - Finalized work breakdown structure across 3 team members (Sreenandan, Sreelakshmi, Sreeveni).
  - Drafted project abstract under Asst. Prof. Arya Sreedhar.

---

### Day 2 — Tuesday, October 6, 2026
* **Focus**: Repository Architecture & Team Governance
* **Planned**:
  - Set up Git repository and modular directory structure.
  - Establish branch protection and Git LFS policies.
  - Define coding standards and guidelines for AI pair-programming harnesses.
* **Executed**:
  - Initialized repository `D:\Development\LumaCam` with active branches (`main`, `dev`, `feature/*`).
  - Authored comprehensive `AGENTS.md` specifying architecture, computational photography guidelines, and coding standards.
  - Configured `.gitignore` for PyTorch checkpoints, `.ARW` raw datasets, Flutter, and Android build artifacts.
  - Created initial `README.md` and `Progress.md`.
* **Git Commits**:
  - `3a42c9e` — *feat: initialize LumaCam architecture, AGENTS.md, Progress.md, and core Attention U-Net GAN modules*
  - `bd42526` — *docs: add comprehensive README with architecture, team context, and setup instructions*

---

### Day 3 — Wednesday, October 7, 2026
* **Focus**: Bayer CFA Processing & Sensor Physics
* **Planned**:
  - Implement Bayer Color Filter Array (CFA) quad packing $(H, W) \rightarrow (H/2, W/2, 4)$.
  - Implement Sony $\alpha$7S II 14-bit black-level subtraction ($512$ pedestal).
  - Implement exposure amplification ratio calculation $\alpha = t_{long} / t_{short}$.
* **Executed**:
  - Created `model/datasets/raw_processing.py` providing NumPy (`pack_raw_bayer_np`) and PyTorch batch (`pack_raw_bayer_torch`) implementations.
  - Validated RGGB spatial slicing: $[R, G_1, B, G_2]$ quad extraction without phase shifting.
  - Added mathematical documentation explaining why demosaicing before denoising scrambles photon shot noise into chromatic blotches.
* **Git Commits**:
  - `4bbeb2c` — *feat: add raw Bayer processing module and fix .gitignore*

---

### Day 4 — Thursday, October 8, 2026
* **Focus**: Attention U-Net Generator & PatchGAN Discriminator
* **Planned**:
  - Implement Spatial Attention Gate module with soft gating map $\alpha \in [0, 1]$.
  - Implement Feature-wise Linear Modulation (FiLM) latent injection bottleneck.
  - Implement 4-stage Attention U-Net Generator with Sub-Pixel `PixelShuffle(2)` head.
  - Implement $70 \times 70$ receptive field Markovian PatchGAN Discriminator.
* **Executed**:
  - Created `model/networks/attention_gate.py`: computes soft spatial attention coefficients suppressing underexposed sensor noise.
  - Created `model/networks/conditioning.py`: zero-initialized affine projection $\text{FiLM}(F) = (1 + \gamma(\alpha)) \odot F + \beta(\alpha)$ enabling seamless identity pass-through.
  - Created `model/networks/generator.py`: 4 downsampling stages (32 $\rightarrow$ 64 $\rightarrow$ 128 $\rightarrow$ 256 channels), FiLM bottleneck (512 channels), 4 Attention Gates, and `PixelShuffle(2)` output head.
  - Created `model/networks/discriminator.py`: 5-layer convolutional PatchGAN evaluating local texture realism.
* **Git Commits**:
  - `d80f008` — *docs: add 2-week team sprint plan and AI handoff specs for Nandan, Veni, and Lakshmi*

---

### Day 5 — Friday, October 9, 2026
* **Focus**: Composite Loss Engine & Multi-Scale SSIM
* **Planned**:
  - Implement differentiable Multi-Scale Structural Similarity (`MSSSIMLoss`) module.
  - Implement conditional GAN loss (`AdversarialLoss`) using `BCEWithLogitsLoss`.
  - Implement composite generator objective: $\mathcal{L}_{total} = \lambda_1 \mathcal{L}_1 + \lambda_{ms} \mathcal{L}_{ms} + \lambda_{adv} \mathcal{L}_{adv}$.
* **Executed**:
  - Created `model/losses/composite_loss.py`.
  - Implemented 5-scale Gaussian spatial pyramid convolution using Wang et al. (2003) weights $[0.0448, 0.2856, 0.3001, 0.2363, 0.1333]$.
  - Stabilized variance computations with $\text{ReLU}$ to prevent negative floating-point variance: $\sigma^2 = \text{ReLU}((x^2 * w) - \mu^2)$.
  - Enforced float32 execution inside loss functions to eliminate half-precision underflow.
* **Git Commits**:
  - `0af94ea` — *feat(losses): implement MS-SSIM, Adversarial, and CompositeLoss modules*

---

### Day 6 — Saturday, October 10, 2026
* **Focus**: Dataset Pipelines (Real SID Sony & Synthetic Noise)
* **Planned**:
  - Build `SIDSonyDataset` for reading Sony $\alpha$7S II `.ARW` raw pairs using `rawpy`.
  - Enforce Bayer phase-aligned random cropping ($top \% 2 == 0, left \% 2 == 0$).
  - Build `DummySIDDataset` for local offline verification without requiring 25 GB downloads.
* **Executed**:
  - Created `model/datasets/sid_dataset.py`.
  - Built `SIDSonyDataset`: reads short raw frames, demosaics long-exposure ground truth using camera white balance (`rawpy.postprocess`), and computes exposure amplification factor $\alpha$.
  - Built `DummySIDDataset`: generates synthetic raw pairs injected with realistic signal-dependent Poisson shot noise and Gaussian readout amplifier noise.
* **Git Commits**:
  - `67b77f7` — *feat(datasets): add SIDSonyDataset loader and DummySIDDataset synthetic generator*

---

### Day 7 — Sunday, October 11, 2026
* **Focus**: Unit Testing, Training Engine & Local GTX 1650 Dry Run
* **Planned**:
  - Write comprehensive unit tests validating shapes, ranges, gradients, and loss functions.
  - Build main training loop `model/train.py` with validation PSNR/SSIM and checkpointing.
  - Build interactive raw camera demonstration script `demo_raw_camera.py`.
  - Validate local execution on NVIDIA GeForce GTX 1650 (4 GB VRAM).
* **Executed**:
  - Created `tests/unit/test_model_forward.py`: verified all 10 unit test cases passing in 3.75s.
  - Created `model/configs/train_config.yaml` and `model/train.py`:
    - Discovered Turing TU117 (GTX 1650) lacks Tensor Cores, causing float16 cuDNN half-precision overflow.
    - Resolved by setting `use_amp: false` for local GTX 1650 (runs float32 at only ~199 MB VRAM footprint) while keeping AMP enabled for cloud Tensor Core GPUs (Tesla P100 / T4).
    - Achieved **4.45s / epoch** dry-run speed on GTX 1650.
    - Implemented automated best-checkpoint saving (`checkpoints/lumacam_best.pt`).
  - Created `demo_raw_camera.py`: tested synthetic Bayer capture and generated visual comparison `results/demo_synthetic.png` (861 KB).
* **Git Commits**:
  - `4ff86ef` — *test: add comprehensive forward pass, gradient flow, and loss unit tests*
  - `f1dd7a9` — *feat(train): implement GAN training loop with validation PSNR/SSIM and dry-run mode*
  - `6b27e11` — *feat(demo): add interactive Bayer raw camera demo script and sensor simulator*
  - `e2ed6bd` — *docs: update Progress.md and me.md with completed training pipeline milestones and metrics*
  - Pushed all 6 commits to GitHub `origin/feature/train-pipeline`.
  - Created and merged Pull Request #1 into `origin/dev`.

---

### Day 8 — Monday, October 12, 2026 (Today)
* **Focus**: Cloud Training Notebook & Comprehensive Documentation
* **Planned**:
  - Author full Kaggle Tesla P100 GPU training notebook `notebooks/train_lumacam_kaggle.ipynb`.
  - Guide user through Kaggle GPU setup, phone verification, and mounting public SID Sony dataset.
  - Launch cloud training run to generate real PSNR/SSIM metrics for Wednesday review.
  - Author complete academic technical report `report.md` with LaTeX mathematical formulations.
* **Executed**:
  - Created `notebooks/train_lumacam_kaggle.ipynb` containing 10 modular cells (GPU check, repo clone, dataset auto-scanner, GAN training loop, loss/PSNR curves, 3-panel visual enhancement plot, Attention Gate heatmaps, and checkpoint downloader).
  - Committed and pushed notebook to `dev` branch (`c5e2945`).
  - Authored comprehensive day-by-day execution log `history.md`.
  - Authored formal academic project report `report.md` detailing theoretical physics, mathematical derivations, architecture, and team splits.
* **Git Commits**:
  - `c5e2945` — *feat(notebooks): add end-to-end Kaggle P100 training notebook for SID Sony dataset*
  - `c49320f` — *docs: update Progress.md and me.md with completed Kaggle training notebook milestone*

---

### Day 9 — Tuesday, October 13, 2026 (Planned)
* **Focus**: Training Convergence Analysis & ONNX Exporter (Lakshmi)
* **Milestones**:
  - Download trained `lumacam_best.pt` checkpoint from Kaggle run.
  - Evaluate convergence curves: verify PSNR reaches $\ge 28.0\text{ dB}$ and SSIM reaches $\ge 0.78$.
  - Export model to ONNX via `model/export_onnx.py` with dual dynamic inputs (`input_bayer`, `alpha_cond`).
  - Verify exported ONNX graph using ONNX Runtime.

---

### Day 10 — Wednesday, October 14, 2026 (Mid-Term Review Milestone)
* **Focus**: Faculty Review with Asst. Prof. Arya Sreedhar
* **Deliverables**:
  - Present Phase 1 / Mid-term progress slides:
    1. Verified architecture diagrams (Bayer CFA $\rightarrow$ FiLM U-Net $\rightarrow$ PatchGAN).
    2. Quantitative benchmark report (PSNR: $28+\text{ dB}$, SSIM: $0.78+$, VRAM: $\approx 200\text{ MB}$, dry-run: $4.45\text{s/epoch}$).
    3. High-resolution visual results: 3-panel [Dark Input vs LumaCam vs Ground Truth].
    4. Attention Gate heatmaps proving noise suppression in underexposed regions.
  - Demonstrate working `demo_raw_camera.py` simulator.

---

### Day 11 — Thursday, October 15, 2026 (Planned)
* **Focus**: LiteRT INT8 Post-Training Quantization (Lakshmi)
* **Milestones**:
  - Implement `edge_opt/quantize_int8.py`.
  - Extract 100 representative calibration patches from dark scenes.
  - Convert ONNX $\rightarrow$ LiteRT (`.tflite`) with symmetric INT8 quantization.
  - Verify quantization degradation: ensure $\Delta\text{PSNR} < 0.5\text{ dB}$ drop compared to FP32.

---

### Day 12 — Friday, October 16, 2026 (Planned)
* **Focus**: Flutter GCam Viewfinder UI (Sreeveni)
* **Milestones**:
  - Scaffold Flutter project in `mobile/`.
  - Build Pixel GCam minimalist interface:
    - Viewfinder display (`Texture` widget).
    - Interactive EV exposure slider (mapping EV $-3$ to $+3$ to $\alpha \in [0.5, 3.0]$).
    - Shutter button with capture animations.

---

### Day 13 — Saturday, October 17, 2026 (Planned)
* **Focus**: Android Camera2 Native Bridge (Sreeveni & Sreenandan)
* **Milestones**:
  - Build Android native Kotlin/NDK camera service:
    - Configure Camera2 `CaptureRequest` for `RAW_SENSOR` / `YUV_420_888`.
    - Zero-copy surface rendering via Android `SurfaceTexture`.
    - Native frame polling into C++ preprocessor buffer.

---

### Day 14 — Sunday, October 18, 2026 (Planned)
* **Focus**: End-to-End Pipeline Integration & Mobile Profiling
* **Milestones**:
  - Integrate LiteRT Android C++ runtime with NNAPI / GPU delegates.
  - Implement post-processing pipeline (Auto White Balance + Gamma + Saturation).
  - Profile on Google Pixel 7: measure per-frame latency (target $\le 33\text{ms}$ / 30 FPS).
  - Measure thermal throttling and battery consumption during continuous viewfinder preview.

---

### Day 15 — Monday, October 19, 2026 (Planned)
* **Focus**: Final Capstone Defense & Demonstration
* **Deliverables**:
  - Full end-to-end live demo on device.
  - Complete project thesis submission.
  - Merge `dev` branch into production `main`.
