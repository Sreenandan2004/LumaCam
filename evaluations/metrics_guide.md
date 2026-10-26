# LumaCam: Metrics Guide & Teacher Presentation Manual
> **Capstone Project Team 21 — SCTCE**  
> **Team Members**: Sreelakshmi A (262), Sreenandan S (263), Sreeveni R (264)  
> **Guided by**: Asst. Prof. Arya Sreedhar (Dept. of Computer Science & Engineering)  
> **Purpose**: A comprehensive guide explaining all visual figures, benchmark graphs, mathematical metrics, and defense talking points to confidently showcase LumaCam to evaluators and faculty.

---

## 1. Quick Elevator Pitch (What to Say First)

> *"Good morning/afternoon Professors. Our project is **LumaCam: Edge-Native Adaptive Low-Light Computational Photography**.*  
> *Existing low-light research is bifurcated: heavy diffusion models produce great photos but take 2 to 5 seconds per image (impossible for real-time camera viewfinders), while standard CNNs run fast but blur fine textures.  
> LumaCam bridges this gap: we built an **Exposure-Conditioned Attention U-Net GAN** operating directly on 4-channel raw Bayer sensor data. It gives us generative-grade texture fidelity with single-pass inference speeds (**14.2 ms / 70 FPS** on laptop GPU, targeting 30 FPS on a Google Pixel 7 NPU), paired with a live user exposure slider powered by FiLM conditioning and native edge color science."*

---

## 2. Visual Pipeline Breakdown

![Pipeline Comparison](figures/pipeline_comparison.png)

### What this figure shows (from left to right):
1. **Panel (A) — Physical Bayer Sensor Input**:
   - **What it is**: The raw mosaic reading straight from the camera sensor under extreme low-light (short exposure time, e.g. 1/30s).
   - **Why it looks almost pitch black**: The sensor captured very few photons. In standard digital photography, the signal is drowned in sensor read noise and Poisson photon shot noise.
2. **Panel (B) — Traditional Linear ISO Digital Gain**:
   - **What it is**: What happens when a standard camera multiplies pixel values by a constant gain factor ($\alpha = 100\times$).
   - **Why it fails**: Linear gain boosts the noise just as much as the signal. The image is covered in severe chroma and luminance noise, color casts, and grain (low PSNR: ~9.4 dB).
3. **Panel (C) — LumaCam Attention U-Net GAN Output**:
   - **What it is**: The output of our generator network.
   - **Why it succeeds**: The Attention Gates filter out the noise on skip connections, while the PatchGAN discriminator forces the generator to reconstruct realistic textures (edges, boundaries) rather than blurring them.
4. **Panel (D) — LumaCam + Color Science**:
   - **What it is**: The final production-ready image after our post-processing pipeline.
   - **Why it matters**: Raw neural network outputs often have subtle color casts or flat tones. Our color pipeline applies **Gray-World Auto White Balance in linear light** and **hue-preserving saturation scaling (1.15x)**, producing vivid, professional-grade sRGB photos.

---

## 3. Explainable AI (XAI): Spatial Attention Gates

![Attention Gate Heatmaps](figures/attention_gate_heatmaps.png)

### How to explain this to your teachers:
A common criticism in AI project vivas is: *"Your deep learning model is just a black box."*  
**This figure proves your model is NOT a black box!**

* **The Problem**: In standard U-Net skip connections, noisy low-level features from the encoder are concatenated directly to the decoder. In low light, this transfers severe background noise into the output.
* **The Solution**: Attention Gates compute a spatial attention mask $\alpha_{g} \in [0, 1]$:
  $$\hat{x}_l = x_l \cdot \sigma_g(W_x x_l + W_g g + b)$$
* **What the Heatmaps Prove**:
  - **Dark Purple / Black regions (value $\to 0$)**: The network has identified uniform dark background noise and **actively suppressed** it from entering the decoder.
  - **Bright Yellow / Orange regions (value $\to 1$)**: The network has detected structural edges, object contours, and textures, dynamically focusing computational capacity on high-frequency details.
  - Notice how **Attention Gate 4** captures coarse semantic regions, while **Attention Gate 1** captures crisp, high-resolution edge boundaries.

---

## 4. Latency & Real-Time Edge Feasibility

![Latency vs Throughput](figures/latency_throughput_benchmark.png)

### The Benchmark Comparison:

| Architecture / Model | Processing Paradigm | Latency (ms) | Throughput (FPS) | Feasible for Live Viewfinder? |
|---|---|:---:|:---:|:---:|
| **DiffLight** (CVPRW 2024) | Iterative Diffusion (DDPM) | ~2,500 ms | 0.4 FPS | ❌ Impossible (takes 2.5 seconds) |
| **Zero-Shot LDM** (AAAI 2024) | Pretrained Latent Diffusion | ~5,000 ms | 0.2 FPS | ❌ Impossible (takes 5 seconds) |
| **SID Base U-Net** (CVPR 2018) | Standard CNN (MSE Loss) | ~150 ms | 6.7 FPS | ❌ Too slow (stuttery preview) |
| **LumaCam** (Our Base Paper, RTX 3090) | Attention U-Net GAN | 62 ms | 16.1 FPS | ⚠️ Near Real-Time |
| **LumaCam** (Our Tested Laptop GTX 1650) | Attention U-Net GAN | **14.2 ms** | **70.4 FPS** | ✅ **Flawless (Sub-33ms, 2x margin)** |
| **LumaCam Mobile Target** (Pixel 7 LiteRT) | INT8 Quantized U-Net | ~25–30 ms | 30–40 FPS | ✅ **Target Edge Deployment** |

### Key Takeaway for Teachers:
* A camera viewfinder requires **at least 30 FPS** to look smooth to human eyes, which means every single frame must be processed in **under 33.3 milliseconds**.
* Diffusion models require 20 to 50 iterative denoising steps ($>2000\text{ ms}$), making them completely unusable for live camera previews.
* LumaCam processes the frame in a **single forward pass**. On our laptop's GTX 1650, it achieves **14.20 ms (70.4 FPS)**—more than double the required real-time speed!

---

## 5. Quality Metrics Explained (PSNR, SSIM, LPIPS)

![Quality Metrics Comparison](figures/quality_metrics_comparison.png)

### 1. PSNR (Peak Signal-to-Noise Ratio)
* **What it measures**: Signal reconstruction fidelity versus ground-truth noise.
* **Mathematical Formula**:
  $$\text{PSNR} = 10 \cdot \log_{10}\left(\frac{\text{MAX}_I^2}{\text{MSE}}\right) = 20 \cdot \log_{10}\left(\frac{\text{MAX}_I}{\sqrt{\text{MSE}}}\right)$$
* **Scale**: Measured in decibels (dB). **Higher is better.**
  - $< 15\text{ dB}$: Very noisy / unrecognizable colors.
  - $20 - 25\text{ dB}$: Acceptable quality.
  - $28 - 32+\text{ dB}$: State-of-the-art computational photography.
* **Why dB is logarithmic**: Every $+3\text{ dB}$ improvement represents a **halving of noise variance** (doubling the signal-to-noise ratio). Our baseline jump from 9.4 dB (linear gain) to ~28.96 dB (full trained LumaCam) represents a massive orders-of-magnitude reduction in sensor noise!

### 2. SSIM (Structural Similarity Index Measure)
* **What it measures**: Human visual perception of image structure, measuring three independent components:
  1. **Luminance comparison** ($l(x, y)$)
  2. **Contrast comparison** ($c(x, y)$)
  3. **Structural correlation** ($s(x, y)$)
* **Scale**: Ranges from $0.0$ (no correlation) to $1.0$ (perfect structural match). **Higher is better.**
  - Traditional Digital Gain: **0.009** (structural correlation destroyed by shot noise).
  - EnlightenGAN: **0.720**
  - Base SID U-Net: **0.732**
  - LumaCam GAN: **0.788** (superior preservation of object boundaries).

### 3. LPIPS (Learned Perceptual Image Patch Similarity)
* **What it measures**: Distance between deep feature representations extracted from a pretrained neural network (e.g. AlexNet/VGG).
* **Scale**: **Lower is better** (0.0 means identical perceptual appearance).
* **Why it is crucial for our project**:
  - Standard U-Nets trained with Mean Squared Error (MSE or $L_1$) regress to the mathematical average of possible pixel values, causing the infamous **"waxy / over-smoothed"** look where hair, foliage, and textures disappear.
  - LumaCam uses a **70×70 Markovian PatchGAN Discriminator**. The discriminator penalizes unrealistic patches, forcing the generator to synthesize sharp micro-textures.
  - This is why LumaCam achieves an **LPIPS of 0.112**, outperforming SID (0.125) and EnlightenGAN (0.165)!

---

## 6. Viva Defense: Anticipated Teacher Questions & Exact Answers

### Q1: *"Why operate on Raw Bayer data instead of standard RGB/JPEG images?"*
> **Answer**:  
> *"Standard JPEG/RGB images have already passed through the hardware Image Signal Processor (ISP), which applies irreversible tone mapping, 8-bit quantization, and lossy compression. In extreme darkness, 8-bit clipping permanently destroys shadow details.  
> By packing raw 14-bit Bayer sensor data directly into a 4-channel tensor $(H/2, W/2, 4)$, LumaCam preserves the complete linear dynamic range of the sensor and suppresses noise before demosaicing artifacts can form."*

### Q2: *"How does the user exposure slider work without retraining the model?"*
> **Answer**:  
> *"We integrated **Feature-wise Linear Modulation (FiLM)** into the U-Net bottleneck. When the user moves the GCam exposure slider, the scalar value $\alpha$ passes through an MLP that predicts scale ($\gamma$) and shift ($\beta$) vectors:  
> $$\text{FiLM}(F) = (1 + \gamma(\alpha)) \odot F + \beta(\alpha)$$  
> This adaptively modulates the latent feature maps in real time without retraining or running separate forward passes."*

### Q3: *"Why did you use a PatchGAN discriminator instead of a standard image classifier?"*
> **Answer**:  
> *"A standard global discriminator outputs a single scalar for the whole image, which often leads to mode collapse or global brightness shifts.  
> A 70×70 Markovian PatchGAN outputs an $N \times N$ matrix of validity scores, where each score evaluates a local 70×70 patch. This enforces texture realism at a micro-scale, ensuring fine structural details like leaves, fabric, and hair look crisp rather than waxy."*

### Q4: *"Why do you need post-processing color science if the GAN already outputs sRGB?"*
> **Answer**:  
> *"The generator learned the camera's average training distribution, but individual indoor/outdoor scenes have different illuminant color temperatures (e.g., tungsten warm vs fluorescent cool).  
> Our post-processing pipeline applies **Gray-World Auto White Balance in linear light** (decoding sRGB $\to$ linear, applying capped channel gains, then re-encoding) and **hue-preserving saturation scaling**, ensuring the final photo is vibrant and color-accurate without distorting hues."*

---

## 7. Artifact Summary File Map

All evaluation assets are organized in the repository:
* **Figures for Slides**: [`evaluations/figures/`](figures/)
  - `pipeline_comparison.png`: 4-panel visual pipeline demo.
  - `attention_gate_heatmaps.png`: 4-scale XAI attention masks.
  - `latency_throughput_benchmark.png`: Log-scale latency vs FPS comparison.
  - `quality_metrics_comparison.png`: PSNR and SSIM benchmarks.
* **Raw Numbers**: [`evaluations/benchmark_data.json`](benchmark_data.json)
* **Interactive Live Demo**: Run `.\.venv\Scripts\streamlit run app.py`
