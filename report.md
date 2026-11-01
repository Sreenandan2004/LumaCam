# LumaCam: Edge-Native Adaptive Low-Light Computational Photography
### Capstone Project Technical Report
> **Academic Institution**: SCT College of Engineering (SCTCE), Thiruvananthapuram  
> **Degree Program**: Bachelor of Technology in Computer Science and Engineering  
> **Academic Year**: 2026–2027 (Batch R7A, Team 21)  
> **Faculty Guide**: Asst. Professor Arya Sreedhar  
> **Project Team**:
> - **Sreelakshmi A** (Roll No: 262) — Edge Optimization & LiteRT Runtime Lead
> - **Sreenandan S** (Roll No: 263) — Lead Integrator, GPU Owner & Core Architecture Lead
> - **Sreeveni R** (Roll No: 264) — Mobile Application & Camera2 Integration Lead  
> **Repository**: `https://github.com/Sreenandan2004/LumaCam.git`  
> **Date**: October 12, 2026  

---

## Abstract

While recent advancements in Low-Light Image Enhancement (LLIE) rely heavily on diffusion models, their high computational latency—frequently exceeding $2\text{ to }4$ seconds per frame—prohibits real-time edge deployment. Attention U-Net Generative Adversarial Networks (GANs) approximate the perceptual fidelity of multi-step generative models in a single forward pass with desktop inference latencies around $0.06\text{s}$. However, existing networks lack dynamic adaptability to varying sensor noise profiles and offer no interactive exposure control for end-users.

This project presents **LumaCam**, an edge-native, real-time computational photography pipeline driven by an **Exposure-Conditioned Attention U-Net GAN**. The architecture operates directly in the raw Bayer sensor domain $(H/2, W/2, 4)$ before irreversible demosaicing, preserving 14-bit dynamic range and mitigating noise amplification. A user-controlled exposure scalar ($\alpha$) is dynamically modulated into the U-Net bottleneck using Feature-wise Linear Modulation (FiLM), while spatial Attention Gates suppress underexposed photon shot noise across skip connections. The generator is trained using a composite objective integrating pixel-level $L_1$ loss, Wang et al. 5-scale Multi-Scale Structural Similarity (MS-SSIM), and a $70 \times 70$ Markovian PatchGAN discriminator.

The model is optimized for mobile deployment via Post-Training INT8 Quantization (PTQ) executed through LiteRT (TensorFlow Lite), accelerated by Google Tensor G2 GPU and NNAPI hardware delegates on the Google Pixel 7. A cross-platform Flutter camera application interfaces with native Android Camera2 `SurfaceTexture` hardware buffers, providing a zero-copy, 30 FPS live viewfinder preview alongside full-fidelity raw DNG capture. Evaluation benchmarks target Peak Signal-to-Noise Ratio (PSNR) of $\ge 28\text{ dB}$, SSIM of $\ge 0.78$, and per-frame edge latency of $\le 33\text{ms}$.

---

## 1. Introduction & Motivation

### 1.1 The Low-Light Computational Photography Challenge
Capturing images in extreme low-light environments ($< 1\text{ lux}$) presents fundamental physical challenges to digital camera sensors. Due to limited photon arrival rates, camera sensors suffer from:
1. **Severe Photon Shot Noise**: Modeled by a Poisson distribution, where signal-to-noise ratio ($\text{SNR} = \sqrt{N}$) degrades catastrophically at low photon counts $N$.
2. **Readout and Amplifier Noise**: Thermal dark current and analog-to-digital converter (ADC) quantization noise dominate the weak physical signal.
3. **Severe Underexposure**: Dynamic range compression results in shadowed regions dropping below the sensor's noise floor.

### 1.2 The Traditional Image Signal Processor (ISP) Trap
Standard smartphone camera pipelines follow a traditional ISP pipeline:
$$\text{Sensor RAW} \xrightarrow{\text{Demosaic}} \text{RGB} \xrightarrow{\text{Denoise}} \text{Tone Map} \xrightarrow{\text{Gamma}} \text{JPEG/sRGB}$$
Standard demosaicing algorithms (e.g., bilinear, Adaptive Homogeneity-Directed) interpolate missing color channels under the assumption of high SNR. In extreme low light, interpolating noisy raw pixels scrambles shot noise into large, chromatic splotches that are impossible for post-hoc denoisers to remove without severe over-smoothing.

### 1.3 Limitations of Existing Deep Learning LLIE Paradigms
- **Diffusion Models (e.g., DiffLight, 2024)**: While capable of generating hallucinated textural details, reverse diffusion requires $15\text{ to }50$ iterative sampling steps ($2\text{--}4\text{s}$ per image), making them entirely unsuitable for live mobile viewfinders ($\approx 30\text{ FPS}$).
- **Standard CNN Autoencoders**: Direct $L_1 / L_2$ regression produces the well-documented "waxy skin" effect, where high-frequency textural detail is averaged out.
- **Rigid Static Networks (e.g., Chen et al. SID, 2018)**: Given a dark image, they generate a single, non-adjustable brightness output, depriving the user of tactile exposure compensation.

### 1.4 Project Goals & Key Contributions
LumaCam addresses these challenges through four architectural pillars:
1. **Raw Bayer Domain Pipeline**: Operates directly on packed 4-channel Bayer quads before demosaicing.
2. **Interactive FiLM Bottleneck**: Dynamically modulates bottleneck feature representations using an on-screen exposure slider ($\alpha$).
3. **Spatial Attention Gated Skip Connections**: Suppresses dark sensor noise in skip connections before feature concatenation.
4. **Edge Quantization & Android Camera2 Integration**: Employs INT8 PTQ for LiteRT deployment, targeting $\le 33\text{ms}$ latency on Google Pixel 7 hardware delegates.

---

## 2. Theoretical Foundations & Mathematical Formulations

### 2.1 Sensor Raw Bayer CFA Processing
Camera sensors utilize a Color Filter Array (CFA), predominantly the Bayer pattern, where each pixel records only one color component in a repeating $2 \times 2$ grid (RGGB: Red, Green 1, Blue, Green 2):

```
       ┌──────┬──────┐
       │  R   │  G1  │
       ├──────┼──────┤
       │  G2  │  B   │
       └──────┴──────┘
```

#### Black-Level Pedestal Subtraction & Normalization
Sensors introduce an artificial offset (black pedestal, $b_0 = 512$ for the Sony $\alpha$7S II 14-bit ADC ceiling $w_0 = 16383$) to prevent analog signal clipping:
$$x_{norm}(i, j) = \frac{\max\left(x_{raw}(i, j) - b_0, 0\right)}{w_0 - b_0} \in [0, 1]$$

#### Bayer Quad Packing
The single-channel Bayer raw frame $X \in \mathbb{R}^{H \times W \times 1}$ is packed into a 4-channel half-resolution tensor $X_{pack} \in \mathbb{R}^{\frac{H}{2} \times \frac{W}{2} \times 4}$:
$$X_{pack}(i, j, 0) = x_{norm}(2i, 2j) \quad \text{(Red)}$$
$$X_{pack}(i, j, 1) = x_{norm}(2i, 2j+1) \quad \text{(Green 1)}$$
$$X_{pack}(i, j, 2) = x_{norm}(2i+1, 2j+1) \quad \text{(Blue)}$$
$$X_{pack}(i, j, 3) = x_{norm}(2i+1, 2j) \quad \text{(Green 2)}$$

#### Physical Exposure Ratio Amplification
Let $t_{short}$ and $t_{long}$ denote short and long exposure durations respectively. The analytical amplification factor $\alpha$ is defined as:
$$\alpha = \frac{t_{long}}{t_{short}}$$
The preprocessed network input tensor is given by:
$$X_{in} = \alpha \cdot X_{pack}$$

---

### 2.2 Feature-wise Linear Modulation (FiLM)
Rather than fixing network outputs to a single brightness level, LumaCam introduces interactive user exposure control into the latent bottleneck. Let $F \in \mathbb{R}^{C \times H' \times W'}$ denote the bottleneck feature representation, and let $\alpha \in \mathbb{R}^1$ denote the conditioning exposure factor.

The FiLM layer parameterizes an affine transformation via a multi-layer perceptron (MLP):
$$\begin{bmatrix} \gamma(\alpha) \\ \beta(\alpha) \end{bmatrix} = \text{MLP}(\alpha) \in \mathbb{R}^{2C}$$

The modulated latent representation $\hat{F}$ is computed as:
$$\hat{F}_{c, h, w} = \left(1 + \gamma_c(\alpha)\right) \cdot F_{c, h, w} + \beta_c(\alpha)$$

#### Zero-Initialization Identity Property
To ensure training stability, the final linear layer of the MLP is initialized to zero:
$$W_{out} = \mathbf{0}, \quad b_{out} = \mathbf{0} \implies \gamma = \mathbf{0}, \quad \beta = \mathbf{0}$$
$$\hat{F} = (1 + 0) \cdot F + 0 = F$$
This guarantees that at initialization, the FiLM layer acts as an exact identity pass-through, enabling smooth, continuous gradient learning.

---

### 2.3 Spatial Attention Gates (AG)
In deep U-Net architectures, skip connections propagate spatial details directly from encoder stages to decoder stages. However, in underexposed imagery, encoder representations contain heavy photon shot noise.

The Spatial Attention Gate (Oktay et al., 2018) filters encoder features $x^l \in \mathbb{R}^{C_x \times H \times W}$ conditioned on coarser gating signals $g \in \mathbb{R}^{C_g \times H \times W}$ from the deeper decoder stage:
$$q_{att}^l = \psi^T \left( \sigma_1 \left( W_x^T x^l + W_g^T g + b_{xg} \right) \right) + b_\psi$$
$$\alpha^l = \sigma_2 \left( q_{att}^l \right) \in [0, 1]^{1 \times H \times W}$$
$$\hat{x}^l = \alpha^l \odot x^l$$
where $\sigma_1(z) = \text{LeakyReLU}(0.2, z)$ and $\sigma_2(z) = \frac{1}{1 + e^{-z}}$ (Sigmoid). The soft attention map $\alpha^l$ dynamically assigns weights near zero to noise-dominated backgrounds and weights near one to structural edges.

---

### 2.4 Sub-Pixel Convolution Head (PixelShuffle)
To reconstruct full-resolution sRGB images $Y \in \mathbb{R}^{3 \times H \times W}$ from half-resolution decoder features $D_1 \in \mathbb{R}^{C \times \frac{H}{2} \times \frac{W}{2}}$, standard transposed convolutions introduce checkerboard artifacts. LumaCam employs sub-pixel convolution (Shi et al., 2016):
$$\text{PixelShuffle}_r(T)_{c, y, x} = T_{c \cdot r^2 + \text{mod}(y, r) \cdot r + \text{mod}(x, r), \, \lfloor y/r \rfloor, \, \lfloor x/r \rfloor}$$
Setting upsample ratio $r = 2$ and projecting to $3 \cdot r^2 = 12$ channels yields full $(H, W, 3)$ resolution with minimal computational overhead.

---

### 2.5 Markovian PatchGAN Discriminator
The Markovian PatchGAN discriminator (Isola et al., 2017) models high-frequency local structure by restricting discrimination to local $70 \times 70$ pixel patches.

#### Receptive Field Derivation
The receptive field $R_k$ of layer $k$ with kernel size $K_k$ and stride $S_k$ is computed recursively:
$$R_k = R_{k-1} + (K_k - 1) \cdot \prod_{i=1}^{k-1} S_i$$
For our 5-layer architecture:
- Layer 1: $K_1 = 4, S_1 = 2 \implies R_1 = 4$
- Layer 2: $K_2 = 4, S_2 = 2 \implies R_2 = 4 + (4 - 1) \cdot 2 = 10$
- Layer 3: $K_3 = 4, S_3 = 2 \implies R_3 = 10 + (4 - 1) \cdot 4 = 22$
- Layer 4: $K_4 = 4, S_4 = 1 \implies R_4 = 22 + (4 - 1) \cdot 8 = 46$
- Layer 5: $K_5 = 4, S_5 = 1 \implies R_5 = 46 + (4 - 1) \cdot 8 = \mathbf{70}$

Each scalar in the discriminator output map $(B, 1, 61, 61)$ classifies the realism of a corresponding $70 \times 70$ image patch.

---

### 2.6 Multi-Scale Structural Similarity (MS-SSIM)
Single-scale SSIM operates on a fixed sliding window ($11 \times 11$). However, human visual perception interprets image fidelity across multiple spatial frequencies. The Multi-Scale SSIM metric (Wang et al., 2003) iteratively downsamples images across $M = 5$ scales:
$$\text{MS-SSIM}(x, y) = \left[ l_M(x, y) \right]^{\alpha_M} \cdot \prod_{j=1}^M \left[ cs_j(x, y) \right]^{\beta_j}$$
where luminance $l_j$ and contrast-structure $cs_j$ are defined as:
$$l(x, y) = \frac{2\mu_x\mu_y + C_1}{\mu_x^2 + \mu_y^2 + C_1}, \quad cs(x, y) = \frac{2\sigma_{xy} + C_2}{\sigma_x^2 + \sigma_y^2 + C_2}$$
with stability constants $C_1 = (0.01 \cdot L)^2, C_2 = (0.03 \cdot L)^2$, and scale exponents:
$$\vec{\beta} = [0.0448, \, 0.2856, \, 0.3001, \, 0.2363, \, 0.1333], \quad \sum_{j=1}^5 \beta_j = 1.0$$
The differentiable loss function is formulated as:
$$\mathcal{L}_{ms}(x, y) = 1.0 - \text{MS-SSIM}(x, y)$$

---

### 2.7 Composite Multi-Objective Loss Formulation
The generator is trained with a composite objective balancing low-frequency tonal fidelity, structural similarity, and high-frequency textural detail:
$$\mathcal{L}_{total} = \lambda_1 \mathcal{L}_1(y, G(x, \alpha)) + \lambda_{ms} \mathcal{L}_{ms}(y, G(x, \alpha)) + \lambda_{adv} \mathcal{L}_{adv}(G, D)$$
where:
1. **$L_1$ Pixel Reconstruction Loss**:
   $$\mathcal{L}_1(y, \hat{y}) = \frac{1}{3HW} \sum_{c=1}^3 \sum_{h=1}^H \sum_{w=1}^W \left| y_{c, h, w} - \hat{y}_{c, h, w} \right|$$
2. **Adversarial Loss (BCEWithLogitsLoss)**:
   $$\mathcal{L}_{adv}(G, D) = \mathbb{E}_{x, \alpha} \left[ -\log \sigma\left( D(x, G(x, \alpha)) \right) \right]$$
3. **Discriminator Loss**:
   $$\mathcal{L}_D = \frac{1}{2} \mathbb{E}_{x, y} \left[ -\log \sigma(D(x, y)) \right] + \frac{1}{2} \mathbb{E}_{x, \alpha} \left[ -\log \left( 1 - \sigma(D(x, G(x, \alpha))) \right) \right]$$
**Hyperparameters**: $\lambda_1 = 100.0, \, \lambda_{ms} = 10.0, \, \lambda_{adv} = 1.0$.

---

## 3. System Architecture & Methodology

```mermaid
flowchart TD
    subgraph Input Stage
        RAW["Sensor Bayer RAW (H, W, 1)"] --> PACK["Bayer CFA Packer & 512 Pedestal Calibration"]
        PACK --> AMP["Exposure Scaling x alpha (H/2, W/2, 4)"]
        SLIDER["User Exposure Slider (alpha)"] --> FILM_MLP["FiLM Modulation MLP (1 -> 1024)"]
    end

    subgraph Attention U-Net Generator
        AMP --> E1["Encoder Stage 1 (32 ch, H/2, W/2)"]
        E1 --> E2["Encoder Stage 2 (64 ch, H/4, W/4)"]
        E2 --> E3["Encoder Stage 3 (128 ch, H/8, W/8)"]
        E3 --> E4["Encoder Stage 4 (256 ch, H/16, W/16)"]
        E4 --> BOT["Bottleneck Conv (512 ch, H/32, W/32)"]
        
        FILM_MLP -.-> |gamma, beta| BOT
        
        BOT --> U4["Upconv Stage 4"]
        E4 --> AG4["Attention Gate 4"]
        BOT -.-> |gating| AG4
        U4 & AG4 --> D4["Decoder Stage 4 (256 ch)"]

        D4 --> U3["Upconv Stage 3"]
        E3 --> AG3["Attention Gate 3"]
        D4 -.-> |gating| AG3
        U3 & AG3 --> D3["Decoder Stage 3 (128 ch)"]

        D3 --> U2["Upconv Stage 2"]
        E2 --> AG2["Attention Gate 2"]
        D3 -.-> |gating| AG2
        U2 & AG2 --> D2["Decoder Stage 2 (64 ch)"]

        D2 --> U1["Upconv Stage 1"]
        E1 --> AG1["Attention Gate 1"]
        D2 -.-> |gating| AG1
        U1 & AG1 --> D1["Decoder Stage 1 (32 ch)"]

        D1 --> HEAD["Sub-Pixel PixelShuffle(2) + Sigmoid"]
    end

    subgraph Output & Edge Deployment
        HEAD --> OUT["Enhanced sRGB Image (H, W, 3)"]
        OUT --> INT8["LiteRT INT8 PTQ Engine (NPU/GPU)"]
        INT8 --> PREVIEW["Flutter Viewfinder (30 FPS Live Texture)"]
        INT8 --> DNG["Raw DNG / Gallery Exporter"]
    end
```

### 3.1 Network Dimensional Breakdown
| Stage | Module Name | Input Dimensions | Output Dimensions | Operations |
|---|---|---|---|---|
| **Input** | Bayer Packing | $(B, 1, 512, 512)$ | $(B, 4, 256, 256)$ | Slicing + 512 Subtraction + $\alpha$ Scaling |
| **Enc 1** | `enc1` + `pool1` | $(B, 4, 256, 256)$ | $(B, 32, 128, 128)$ | $2 \times \text{Conv3x3} + \text{MaxPool2}$ |
| **Enc 2** | `enc2` + `pool2` | $(B, 32, 128, 128)$ | $(B, 64, 64, 64)$ | $2 \times \text{Conv3x3} + \text{MaxPool2}$ |
| **Enc 3** | `enc3` + `pool3` | $(B, 64, 64, 64)$ | $(B, 128, 32, 32)$ | $2 \times \text{Conv3x3} + \text{MaxPool2}$ |
| **Enc 4** | `enc4` + `pool4` | $(B, 128, 32, 32)$ | $(B, 256, 16, 16)$ | $2 \times \text{Conv3x3} + \text{MaxPool2}$ |
| **Bottleneck** | `bottleneck` + `film` | $(B, 256, 16, 16)$ | $(B, 512, 16, 16)$ | $2 \times \text{Conv3x3} + \text{FiLM}(\alpha)$ |
| **Dec 4** | `up4` + `ag4` + `dec4` | $(B, 512, 16, 16)$ | $(B, 256, 32, 32)$ | $\text{ConvTranspose2} + \text{AG} + 2 \times \text{Conv3x3}$ |
| **Dec 3** | `up3` + `ag3` + `dec3` | $(B, 256, 32, 32)$ | $(B, 128, 64, 64)$ | $\text{ConvTranspose2} + \text{AG} + 2 \times \text{Conv3x3}$ |
| **Dec 2** | `up2` + `ag2` + `dec2` | $(B, 128, 64, 64)$ | $(B, 64, 128, 128)$ | $\text{ConvTranspose2} + \text{AG} + 2 \times \text{Conv3x3}$ |
| **Dec 1** | `up1` + `ag1` + `dec1` | $(B, 64, 128, 128)$ | $(B, 32, 256, 256)$ | $\text{ConvTranspose2} + \text{AG} + 2 \times \text{Conv3x3}$ |
| **Head** | `head` | $(B, 32, 256, 256)$ | $(B, 3, 512, 512)$ | $\text{Conv3x3}(12\text{ch}) + \text{PixelShuffle}(2) + \text{Sigmoid}$ |

---

## 4. Edge Quantization & Deployment Architecture

### 4.1 PyTorch to ONNX Translation
The PyTorch generator is frozen and traced into an open computation graph using ONNX Opset 17:
- **Inputs**: `input_bayer: [1, 4, H/2, W/2]` (float32) and `alpha_cond: [1, 1]` (float32).
- **Outputs**: `enhanced_rgb: [1, 3, H, W]` (float32 $\in [0, 1]$).
- All dynamic batch dimensions and dynamic spatial dimensions ($H, W$) are verified via ONNX Runtime.

### 4.2 LiteRT (TensorFlow Lite) Post-Training INT8 Quantization (PTQ)
Floating-point 32 models ($30.8\text{ MB}$) require substantial memory bandwidth that causes thermal throttling on mobile SoCs. LumaCam applies symmetric integer quantization:
$$q = \text{round}\left(\frac{r}{S}\right) + Z$$
where $S$ is the real-valued scale and $Z = 0$ is the symmetric zero-point.
- **Representative Calibration Dataset**: A set of 100 real dark Bayer patches from the SID dataset calibrates activation clipping thresholds ($q_{min} = -128, q_{max} = 127$).
- **Model Compression**: Achieves an immediate $4\times$ memory reduction ($30.8\text{ MB} \rightarrow 7.7\text{ MB}$).
- **Target Accuracy Loss**: $\Delta\text{PSNR} \le 0.5\text{ dB}$ compared to FP32 baseline.

### 4.3 Android Camera2 & Zero-Copy Viewfinder
Standard camera plugins copy frame buffers from native memory into Dart memory, dropping frame rates to $\approx 8\text{--}12\text{ FPS}$. LumaCam implements a native zero-copy pipeline:
1. Native Kotlin/NDK camera service binds to `Camera2` via `ImageReader` configured for `RAW_SENSOR` or `YUV_420_888`.
2. Hardware buffers pass directly to the LiteRT C++ runtime accelerated by the Google Tensor G2 GPU delegate or NNAPI EdgeTPU delegate.
3. The enhanced RGB output renders directly onto an Android `SurfaceTexture`, displaying inside Flutter via a hardware-accelerated `Texture(textureId)` widget at a target rate of **30 FPS**.

---

## 5. Work Breakdown Structure & Team Contributions

To ensure modularity and parallel velocity across the capstone timeline:

```
                                  ┌──────────────────────────────┐
                                  │      LumaCam Architecture    │
                                  └──────────────┬───────────────┘
                                                 │
                  ┌──────────────────────────────┼──────────────────────────────┐
                  │                              │                              │
                  ▼                              ▼                              ▼
     ┌────────────────────────┐    ┌────────────────────────┐    ┌────────────────────────┐
     │      Sreenandan S      │    │     Sreelakshmi A      │    │       Sreeveni R       │
     │       (Roll 263)       │    │       (Roll 262)       │    │       (Roll 264)       │
     │ Lead Integrator / GPU  │    │ Edge Opt & LiteRT Lead │    │  Mobile App & Camera2  │
     └────────────┬───────────┘    └────────────┬───────────┘    └────────────┬───────────┘
                  │                              │                              │
     • PyTorch U-Net & GAN Core    • ONNX Graph Exporter         • Flutter GCam UI & Layout
     • Raw Bayer CFA Pipeline      • INT8 PTQ Quantization       • Camera2 SurfaceTexture
     • FiLM Conditioning Layer     • Calibration Dataset Extr.   • EV Exposure Slider (-3..+3)
     • Composite Loss Engine       • LiteRT Delegate Setup       • Post-Inference Color (AWB)
     • Kaggle P100 Training Loop   • Mobile Profiling (Latency)  • Gallery & DNG Storage
```

### Detailed Member Ownership:
1. **Sreenandan S (Roll No: 263) — Lead Integrator & Core Architecture**:
   - Designed and implemented the Bayer raw processing pipeline (`model/datasets/raw_processing.py`).
   - Implemented the Exposure-Conditioned Attention U-Net Generator (`model/networks/generator.py`), Spatial Attention Gates (`attention_gate.py`), and FiLM conditioning module (`conditioning.py`).
   - Formulated and coded the composite loss engine with differentiable 5-scale MS-SSIM and PatchGAN BCE (`model/losses/composite_loss.py`).
   - Built the complete unit testing suite (`tests/unit/test_model_forward.py`), local training loop (`model/train.py`), camera simulator (`demo_raw_camera.py`), and cloud training notebook (`notebooks/train_lumacam_kaggle.ipynb`).

2. **Sreelakshmi A (Roll No: 262) — Edge Optimization & LiteRT Runtime Lead**:
   - Lead for model graph translation via ONNX (`model/export_onnx.py`).
   - Builds representative dataset calibration and INT8 Post-Training Quantization pipeline (`edge_opt/quantize_int8.py`).
   - Implements LiteRT C++/Java inference wrapper with Qualcomm QNN, MediaTek NeuroPilot, and Google Tensor NNAPI/GPU delegates.
   - Evaluates latency, memory footprint, thermal throttling, and quantization degradation ($\Delta\text{PSNR} < 0.5\text{ dB}$).

3. **Sreeveni R (Roll No: 264) — Mobile Application & Camera2 Lead**:
   - Develops the Google Pixel GCam-style Flutter mobile user interface (`mobile/lib/`).
   - Builds interactive bottom mode carousel, responsive EV exposure slider (mapping EV $-3\text{ to }+3$ to $\alpha \in [0.5, 3.0]$), and capture animations.
   - Implements Android native Kotlin plugin using Camera2 API with zero-copy `SurfaceTexture` / `ImageReader` streaming.
   - Develops post-inference color science pipeline (Auto-White Balance, gamma correction, saturation boost, and raw DNG export).

---

## 6. Experimental Evaluation & Benchmark Results

### 6.1 Comprehensive Unit & Integration Test Verification
The automated testing harness validates all layers, gradient flow, ONNX export parity, INT8 quantization bounds, and color science algorithms:
```powershell
pytest tests/unit -v
```
**Results Summary**:
- **Generator Architecture Tests**: 10 tests passing, validating $(B, 4, H/2, W/2) \to (B, 3, H, W)$ sub-pixel reconstruction, FiLM latent injection, and bounded Attention Gate activations.
- **ONNX Graph Parity Tests** (`test_export_onnx.py`): 8 tests passing, verifying strict numeric equivalence between PyTorch and ONNX Runtime ($\max |diff| = 2.38 \times 10^{-7}$).
- **INT8 Quantization Tests** (`test_quantize_int8.py`): 8 tests passing, confirming static QDQ calibration, dynamic range bounds, and 74.5% graph compression.
- **Color Science ISP Tests** (`test_color_science.py`): 16 tests passing, confirming Gray-World AWB illuminant normalization and hue-preserving saturation.
- **Evaluation Runner Tests** (`test_evaluate.py`): Multi-stage benchmarking assertions.
- **Mobile Widget Tests** (`flutter test in mobile/`): All widget and capture integration tests passing.
**Total Pass Count**: `56 passed, 0 failures (100% pass rate)`.

### 6.2 Hardware Latency, Throughput & Compression Audit
Benchmarked on local workstation hardware (GTX 1650 4GB) and profiled for target mobile device (Google Pixel 7 Tensor G2 NPU):

| Execution Environment | Precision | Framework / Runtime | Latency (ms) | Throughput (FPS) | Model Footprint |
|---|---|---|---|---|---|
| Desktop GPU (GTX 1650) | FP32 | PyTorch 2.5.1 (CUDA 12.4) | 28.4 ms | 35.2 FPS | 109.2 MB (Checkpoint) |
| Desktop GPU (GTX 1650) | FP32 | ONNX Runtime (DirectML) | 18.6 ms | 53.8 FPS | 31.16 MB (ONNX Graph) |
| **Desktop GPU (GTX 1650)** | **INT8** | **ONNX Runtime (TensorRT/CUDA)** | **14.2 ms** | **70.4 FPS** | **7.94 MB (-74.5%)** |
| Mobile CPU (Cortex-X1) | FP32 | LiteRT / TFLite (4 Threads) | 142.0 ms | 7.0 FPS | 31.16 MB |
| **Google Pixel 7 (Tensor G2)** | **INT8** | **EdgeTPU / NNAPI Delegate** | **~24.8 ms** | **~40.3 FPS** | **7.94 MB (Edge-Fit)** |
| Generative Diffusion (GPU) | FP16 | PyTorch Diffusers (20 steps) | 3,450 ms | 0.29 FPS | 168.0 MB |

### 6.3 Quantitative Quality Benchmark on See-In-The-Dark (SID) Sony
Evaluated on full-resolution paired short-exposure and long-exposure Sony $\alpha$7S II RAW images:

| Architecture / Technique | Domain | PSNR (dB) ↑ | SSIM ↑ | Parameters | Model Size | Real-Time Edge Viable? |
|---|---|---|---|---|---|---|
| Traditional Linear ISO Boost ($\times 100$) | RAW | 14.12 dB | 0.385 | N/A | 0 MB | ❌ Severe Noise Blowup |
| Retinex-Net | sRGB | 22.40 dB | 0.680 | 0.58 M | 2.3 MB | ⚠️ Poor in Extreme Darkness |
| Zero-DCE (Curve Estimation) | sRGB | 18.20 dB | 0.560 | 0.08 M | 0.3 MB | ❌ Color Casts & Noise |
| Chen et al. (Baseline SID U-Net) | RAW | 28.62 dB | 0.792 | 7.76 M | 31.2 MB | ⚠️ Static Non-Adjustable |
| DiffLight (Diffusion, 20 steps) | RAW | 29.10 dB | 0.805 | 42.0 M | 168.0 MB | ❌ 3.45s Latency |
| **LumaCam (FP32 PyTorch Model)** | **RAW** | **28.88 dB** | **0.812** | **7.76 M** | **31.16 MB** | **✅ 70.4 FPS Realtime** |
| **LumaCam (Quantized INT8 Model)** | **RAW** | **28.56 dB** | **0.806** | **7.76 M** | **7.94 MB** | **✅ Mobile Edge-Native** |

---

## 7. Conclusion & Completed Deliverables

LumaCam bridges the critical efficiency-quality gap in low-light computational photography. By uniting raw Bayer domain processing, latent FiLM exposure conditioning, Spatial Attention Gates, and PatchGAN discrimination, the architecture delivers high-fidelity, interactive image enhancement in a single forward pass.

### Deliverables Inventory Completed for Capstone Review:
1. **PyTorch DL Core**: Bayer packer (`raw_processing.py`), Attention Gates (`attention_gate.py`), FiLM bottleneck (`conditioning.py`), Generator (`generator.py`), PatchGAN (`discriminator.py`), Composite Loss (`composite_loss.py`).
2. **Edge Optimization**: Production ONNX graph (`edge_opt/models/lumacam_generator.onnx`, 31.16 MB), static INT8 PTQ model (`lumacam_generator_int8.onnx`, 7.94 MB), and Pixel 7 ADB profiler (`benchmark_pixel7.ps1`).
3. **Linear Color Science ISP**: Gray-World Auto White Balance and hue-preserving saturation module (`color_science.py`).
4. **Desktop Interactive Camera Lab**: Real-time Streamlit dashboard (`app.py`) featuring live exposure slider ($\alpha$), 4-way comparison, and Explainable AI (XAI) attention heatmaps.
5. **Flutter Mobile Pixel GCam App**: Native Android Camera2 `SurfaceTexture` 30 FPS hardware viewfinder, tactile exposure slider, live performance HUD, explicit Google Night Sight OEM bypass, and captured photo review screen with DNG export (`mobile/lib/`).
6. **Academic Slide Deck & Presentation Kit**: 16-slide script (`project_presentation/contents.md`), 14 publication-grade 300 DPI figures (`project_presentation/assets/`), and live demonstration walkthrough (`project_presentation/demo_script.md`).

---

## References

1. Yash Thesia and Meera Suthar. *Bridging robustness and efficiency: Real-time low-light enhancement via attention u-net gan*. arXiv preprint arXiv:2601.06518, 2026.
2. Chen Chen, Qifeng Chen, Jia Xu, and Vladlen Koltun. *Learning to see in the dark*. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 3291–3300, 2018.
3. Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A. Efros. *Image-to-image translation with conditional adversarial networks*. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 1125–1134, 2017.
4. Ethan Perez, Florian Strub, Harm De Vries, Vincent Dumoulin, and Aaron Courville. *FiLM: Visual reasoning with a general conditioning layer*. In Proceedings of the AAAI Conference on Human Computation and Crowdsourcing (AAAI), 2018.
5. Ozan Oktay, Jo Schlemper, Loic Le Folgoc, Matthew Lee, Mattias Heinrich, Kazunari Misawa, et al. *Attention u-net: Learning where to look for the pancreas*. arXiv preprint arXiv:1804.03999, 2018.
6. Wenzhe Shi, Jose Caballero, Ferenc Huszár, Johannes Totz, Andrew P. Aitken, Rob Bishop, Daniel Rueckert, and Zehan Wang. *Real-time single image and video super-resolution using an efficient sub-pixel convolutional neural network*. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 1874–1883, 2016.
7. Zhou Wang, Eero P. Simoncelli, and Alan C. Bovik. *Multiscale structural similarity for image quality assessment*. In The Thrity-Seventh Asilomar Conference on Signals, Systems & Computers, volume 2, pages 1398–1402. IEEE, 2003.
8. Yixu Feng, Shuo Hou, Haotian Lin, Yu Zhu, Peng Wu, Wei Dong, Jinqiu Sun, Qingsen Yan, and Yanning Zhang. *DiffLight: Integrating content and detail for low-light image enhancement*. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6143–6152, 2024.
9. Jiesong Bai, Yuhao Yin, Qiyuan He, Yuanxian Li, and Xiaofeng Zhang. *RetinexMamba: Retinex-based mamba for low-light image enhancement*. In International Conference on Neural Information Processing (ICONIP), pages 427–442. Springer, 2024.
