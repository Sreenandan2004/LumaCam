# LumaCam: Edge-Native Adaptive Low-Light Computational Photography

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Flutter](https://img.shields.io/badge/Flutter-3.44+-02569B.svg?logo=flutter&logoColor=white)](https://flutter.dev/)
[![LiteRT](https://img.shields.io/badge/LiteRT-TFLite-FF6F00.svg?logo=tensorflow&logoColor=white)](https://ai.google.dev/edge/litert)

> **Final Year Capstone Project — Team 21**  
> **Institution**: Sree Chitra Thirunal College of Engineering (SCTCE), Dept. of Computer Science & Engineering  
> **Faculty Guide**: Asst. Prof. Arya Sreedhar  
> **Team Members**: Sreelakshmi A (262), Sreenandan S (263), Sreeveni R (264)  
> **Target Device**: Google Pixel 7 (Tensor G2, Camera2 Level 3 / RAW_SENSOR, NNAPI & GPU Delegates)  

---

## 📸 Overview

**LumaCam** is a real-time, edge-native Low-Light Image Enhancement (LLIE) mobile camera application inspired by the Google Pixel GCam interface.

While recent advancements in low-light computational photography rely heavily on multi-second diffusion models or heavy vision transformers (e.g., DiffLight, RetinexMamba), their computational latency (exceeding 2–4 seconds per frame) prohibits real-time mobile viewfinder deployment. Furthermore, existing feed-forward restoration networks operate as static "black-box" models without giving the photographer tactile control over exposure.

LumaCam bridges this efficiency-quality gap through four core innovations:
1. **Raw Bayer Pipeline**: Bypasses conventional demosaicing algorithms by operating directly on packed 4-channel raw sensor arrays $(H/2, W/2, 4)$, preserving 14-bit dynamic range and suppressing shot noise before color reconstruction.
2. **Exposure-Conditioned Attention U-Net GAN**: A single-pass generator combining **Spatial Attention Gates** on skip connections (to filter dark noise) and a **FiLM Latent Conditioning Layer** at the bottleneck, allowing continuous user exposure adjustment ($\alpha$).
3. **PatchGAN Discriminator**: A $70 \times 70$ Markovian patch discriminator penalizing synthetic over-smoothing and recovering realistic micro-textures.
4. **Edge Quantization & Zero-Copy Camera2 Streaming**: Post-Training INT8 Quantization (PTQ) executed via LiteRT with NNAPI/GPU delegates, piped directly into a Flutter GCam viewfinder via Android `SurfaceTexture`.

---

## 🏗️ System Architecture

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

## 📂 Repository Structure

```text
LumaCam/
├── AGENTS.md                               # AI agent specifications and coding standards
├── Progress.md                             # Milestone tracking, architecture state & handoff log
├── README.md                               # Public project documentation (this file)
├── requirements.txt                        # Python dependencies
├── .gitignore                              # Comprehensive git ignore rules
├── model/                                  # Workstream 1: Model Engineering & Training
│   ├── datasets/
│   │   ├── raw_processing.py               # Bayer quad packing, black-level calibration
│   │   └── sid_dataset.py                  # (Planned) See-in-the-Dark Sony dataset loader
│   ├── networks/
│   │   ├── attention_gate.py               # Spatial Attention Gate module
│   │   ├── conditioning.py                 # Latent FiLM exposure conditioning layer
│   │   ├── generator.py                    # Exposure-Conditioned Attention U-Net
│   │   └── discriminator.py                # 70x70 Markovian PatchGAN discriminator
│   ├── losses/                             # (Planned) MS-SSIM + L1 + cGAN composite loss
│   ├── train.py                            # (Planned) Training & validation loop
│   └── export_onnx.py                      # (Planned) ONNX model exporter
├── edge_opt/                               # Workstream 2: Quantization & LiteRT Deployment
│   ├── quantize_int8.py                    # (Planned) INT8 PTQ with representative calibration
│   └── benchmark_edge.py                   # (Planned) Latency & PSNR profiler
├── mobile/                                 # Workstream 3: Flutter & Android Native App
│   ├── android/                            # Native Android Camera2 & LiteRT host
│   └── lib/                                # Flutter GCam UI & BLoC state management
└── tests/                                  # Unit and integration test suites
    └── unit/                               # Tensor shape & forward pass tests
```

---

## ⚡ Current Implementation Status

See [`Progress.md`](Progress.md) for the detailed workstream breakdown and checklist.

| Module | Status | Description |
|---|---|---|
| Bayer Raw Processing | ✅ Implemented | NumPy & PyTorch quad packing $(H, W) \rightarrow (H/2, W/2, 4)$, black-pedestal calibration (512 for Sony). |
| Spatial Attention Gate | ✅ Implemented | Soft spatial attention gating $(x \odot \alpha)$ filtering dark shot noise on skip connections. |
| FiLM Latent Conditioning | ✅ Implemented | Zero-initialized affine modulation $(1 + \gamma(\alpha)) \odot F + \beta(\alpha)$ at bottleneck. |
| Core Generator | ✅ Implemented | 4-stage encoder-decoder U-Net with sub-pixel `PixelShuffle(2)` reconstruction head. |
| PatchGAN Discriminator | ✅ Implemented | $70 \times 70$ receptive field Markovian discriminator enforcing texture realism. |
| Composite Loss & Trainer | 🔄 In Progress | Multi-Scale SSIM, pixel $L_1$, and adversarial loss pipeline. |
| INT8 PTQ & LiteRT | ⏳ Planned | Quantization to INT8 with $< 0.5\text{ dB}$ PSNR drop target. |
| GCam UI & Camera2 | ⏳ Planned | Live preview on Pixel 7 with dynamic exposure slider. |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- PyTorch 2.2+ with CUDA support
- Flutter 3.44+
- Android Studio / Android SDK (API Level 31+)

### Installation
```bash
# Clone the repository
git clone https://github.com/Sreenandan2004/LumaCam.git
cd LumaCam

# Install Python dependencies
pip install -r requirements.txt
```

---

## 📖 Key References

1. **Base Paper**: Yash Thesia and Meera Suthar. *"Bridging robustness and efficiency: Real-time low-light enhancement via attention u-net gan."* arXiv preprint arXiv:2601.06518, 2026.
2. **SID**: Chen et al. *"Learning to See in the Dark."* CVPR, 2018.
3. **PatchGAN / pix2pix**: Isola et al. *"Image-to-Image Translation with Conditional Adversarial Networks."* CVPR, 2017.
4. **FiLM**: Perez et al. *"FiLM: Visual Reasoning with a General Conditioning Layer."* AAAI, 2018.
5. **Attention U-Net**: Oktay et al. *"Attention U-Net: Learning Where to Look for the Pancreas."* MIDL, 2018.
