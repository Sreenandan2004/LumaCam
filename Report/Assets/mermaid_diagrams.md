# LumaCam: Mermaid Diagrams & Flowcharts for Report

This document contains all formal flowcharts and diagrams designed for the LumaCam Phase 1 Project Report. These can be viewed directly in markdown previewers or converted into TikZ / LaTeX figures.

---

## 1. High-Level End-to-End System Architecture

```mermaid
flowchart TD
    subgraph Sensor ["1. Acquisition Domain (Google Pixel 7)"]
        A["Physical Low-Light Scene (< 1 lux)"] --> B["Camera Sensor (14-bit CFA)"]
        B --> C["Android Camera2 HAL\n(OEM Night Sight BYPASSED)"]
        C --> D["Pure Bayer RAW Stream\n(Level 3 RAW_SENSOR)"]
    end

    subgraph NativeBridge ["2. Native Android Preprocessing (Kotlin / C++)"]
        D --> E["Black Pedestal Subtraction\n(b0 = 512, w0 = 16383)"]
        E --> F["Bayer Quad Packing\n(H, W, 1) -> (H/2, W/2, 4)"]
        F --> G["Analytical Amplification\n(Input = alpha * X_pack)"]
    end

    subgraph UserControl ["3. Interactive Exposure Control"]
        SL["Tactile Viewfinder Slider\n(EV -3 to +3 / alpha 1 to 300)"] --> G
        SL --> H["FiLM Conditioning Latent\nMLP(alpha) -> (gamma, beta)"]
    end

    subgraph NeuralCore ["4. Quantized Neural Processing (LiteRT / Tensor G2)"]
        G --> ENC["Attention U-Net Encoder\n(4 Downsampling Stages)"]
        ENC --> BN["Bottleneck Representation\n(512 x 16 x 16)"]
        H --> MOD["FiLM Feature Modulation\nF_hat = (1 + gamma)*F + beta"]
        BN --> MOD
        MOD --> DEC["Attention U-Net Decoder\n(4 Upsampling Stages)"]
        ENC -.->|"Skip Connections"| AG["Spatial Attention Gates\n(Noise Suppression)"]
        AG -.-> DEC
        DEC --> PS["Sub-Pixel PixelShuffle Head\n(12 -> 3 channels, 2x)"]
    end

    subgraph PostISP ["5. Linear Color Science & Display"]
        PS --> AWB["Gray-World Auto White Balance\n(Illuminant Normalization)"]
        AWB --> SAT["Hue-Preserving Saturation Boost\n(Delta-E Minimization)"]
        SAT --> TM["Gamma Transfer Function\n(Linear to sRGB Encoding)"]
        TM --> VIEW["30 FPS Zero-Copy Flutter Viewfinder\n(Texture Widget)"]
        TM --> DNG["High-Fidelity DNG / JPEG Storage"]
    end
```

---

## 2. Zero-Copy Android Camera2 SurfaceTexture Frame Pipeline

```mermaid
sequenceDiagram
    autonumber
    participant Sensor as Camera Sensor (Hardware)
    participant HAL as Camera2 HAL (Android OS)
    participant Kotlin as Camera2RawController (Kotlin)
    participant NPU as LiteRT Engine (Tensor G2)
    participant Flutter as Flutter GCam UI (Dart)

    Flutter->>Kotlin: initializeCamera(previewWidth, previewHeight)
    Kotlin->>HAL: openCamera("0") [Probe RAW_SENSOR Capability]
    HAL-->>Kotlin: CameraDevice opened
    Kotlin->>Flutter: registerSurfaceTexture() -> textureId
    Flutter->>Flutter: Render Texture(textureId: id)
    Kotlin->>HAL: createCaptureSession(SurfaceTexture, ImageReader)
    HAL-->>Kotlin: CaptureSession active (Repeating Request)
    
    loop Every Frame (~33 ms / 30 FPS)
        Sensor->>HAL: Capture 14-bit Bayer photons
        HAL->>Kotlin: onImageAvailable(ImageReader RAW_SENSOR)
        Kotlin->>Kotlin: Bayer Quad Packing + Pedestal Subtraction
        Flutter->>Kotlin: updateExposureFactor(alpha)
        Kotlin->>NPU: forward(packed_quad, alpha)
        NPU-->>Kotlin: restored_linear_rgb (24.8 ms)
        Kotlin->>Kotlin: Fast Color Science (AWB + Saturation)
        Kotlin->>HAL: writeFrameToSurfaceTexture() [Zero-Copy]
        HAL-->>Flutter: Hardware Buffer swapped to Viewfinder
        Flutter->>Flutter: Update HUD Overlay (FPS: 30, Latency: 24.8ms)
    end

    opt User Taps Shutter Button
        Flutter->>Kotlin: captureStillFrame(alpha)
        Kotlin->>HAL: Capture Single Full-Res RAW
        Kotlin->>NPU: High-Fidelity Neural Reconstruction
        Kotlin-->>Flutter: Return Captured Review Screen (DNG + JPEG)
        Flutter->>Flutter: Display Before/After Split Lens Comparison
    end
```

---

## 3. Attention U-Net Generator with FiLM Latent Injection

```mermaid
flowchart LR
    subgraph Input ["Input Tensor"]
        IN["Packed Bayer Quad\n(B, 4, H/2, W/2)"]
    end

    subgraph Encoder ["Contracting Path (Encoder)"]
        E1["ConvBlock 1\n(32, H/2, W/2)"]
        E2["ConvBlock 2\n(64, H/4, W/4)"]
        E3["ConvBlock 3\n(128, H/8, W/8)"]
        E4["ConvBlock 4\n(256, H/16, W/16)"]
        IN --> E1 --> E2 --> E3 --> E4
    end

    subgraph Bottleneck ["Bottleneck & Conditioning"]
        E4 --> B1["Bottleneck Conv\n(512, H/32, W/32)"]
        ALPHA["Exposure Alpha\n(B, 1)"] --> MLP["FiLM MLP\nLinear -> LeakyReLU -> Linear"]
        MLP --> PARAMS["gamma, beta in R^(512)"]
        B1 --> AFFINE["Affine Modulation\n(1 + gamma)*F + beta"]
        PARAMS --> AFFINE
    end

    subgraph Decoder ["Expanding Path (Decoder)"]
        AFFINE --> D4["UpConv 4\n(256, H/16, W/16)"]
        D4 --> D3["UpConv 3\n(128, H/8, W/8)"]
        D3 --> D2["UpConv 2\n(64, H/4, W/4)"]
        D2 --> D1["UpConv 1\n(32, H/2, W/2)"]
    end

    subgraph AttentionGates ["Spatial Attention Gates"]
        E4 -.-> AG4["Attention Gate 4"] -.-> D4
        D4 -.->|"Gating Signal g"| AG4
        E3 -.-> AG3["Attention Gate 3"] -.-> D3
        D3 -.->|"Gating Signal g"| AG3
        E2 -.-> AG2["Attention Gate 2"] -.-> D2
        D2 -.->|"Gating Signal g"| AG2
        E1 -.-> AG1["Attention Gate 1"] -.-> D1
        D1 -.->|"Gating Signal g"| AG1
    end

    subgraph OutputHead ["Sub-Pixel Reconstruction"]
        D1 --> SH["Sub-Pixel Head\nConv2d (32 -> 12)"]
        SH --> PS["PixelShuffle (r=2)\n(B, 12, H/2, W/2) -> (B, 3, H, W)"]
        PS --> SIG["Sigmoid Activation\nOutput in [0, 1]"]
    end
```

---

## 4. Composite Objective Optimization Workflow

```mermaid
flowchart TD
    DARK["Dark Raw Input (x)"] --> GEN["Generator G(x, alpha)"]
    ALPHA["Conditioning Factor (alpha)"] --> GEN
    GEN --> FAKE["Generated Image G(x)"]
    REAL["Ground Truth Long Exposure (y)"]

    subgraph Losses ["Composite Loss Formulation"]
        FAKE & REAL --> L1["Pixel Loss:\nL1 = ||G(x) - y||_1"]
        FAKE & REAL --> MSSSIM["Structural Loss:\nL_MS-SSIM (5 Scales)"]
        
        FAKE & DARK --> DISC["PatchGAN Discriminator D(x, G(x))"]
        REAL & DARK --> DISC_R["PatchGAN Discriminator D(x, y)"]
        
        DISC --> LGAN["cGAN Adversarial Loss:\nE[(D(x, G(x)) - 1)^2]"]
    end

    L1 --> TOTAL["Total Generator Loss:\nL_G = L1 + 0.2*L_MS-SSIM + 0.05*L_cGAN"]
    MSSSIM --> TOTAL
    LGAN --> TOTAL

    TOTAL --> OPT_G["Adam Optimizer (G)\nlr = 1e-4, beta1 = 0.5, beta2 = 0.999"]
    DISC & DISC_R --> OPT_D["Adam Optimizer (D)\nlr = 1e-4, beta1 = 0.5, beta2 = 0.999"]
```

---

## 5. Team Work Breakdown Structure (3-Member Ownership)

```mermaid
flowchart TD
    ROOT["LumaCam Capstone Project (Team 21)"]

    subgraph Sreenandan ["Sreenandan S (Roll 263)\nLead Integrator & Core DL Lead"]
        M1_1["Raw Bayer Processing & Quad Packing"]
        M1_2["Attention U-Net Architecture Design"]
        M1_3["FiLM Latent Conditioning Module"]
        M1_4["PatchGAN Discriminator & Composite Loss"]
        M1_5["PyTorch Training Loop & Kaggle P100 Pipeline"]
        M1_6["Automated Unit Test Suite (56 tests)"]
    end

    subgraph Sreelakshmi ["Sreelakshmi A (Roll 262)\nEdge Optimization & LiteRT Runtime Lead"]
        M2_1["ONNX Dynamic & Static Graph Export"]
        M2_2["Representative Calibration Dataset Extraction"]
        M2_3["Post-Training INT8 Quantization (PTQ)"]
        M2_4["Quantization PSNR/SSIM Degradation Audit"]
        M2_5["Pixel 7 ADB Latency & Memory Profiler"]
        M2_6["LiteRT NNAPI / EdgeTPU Delegate Harness"]
    end

    subgraph Sreeveni ["Sreeveni R (Roll 264)\nMobile Application & Camera2 Lead"]
        M3_1["Pixel GCam Flutter UI & Viewfinder Screen"]
        M3_2["Tactile Real-Time Exposure Slider (alpha)"]
        M3_3["Android Native Camera2 RAW_SENSOR Bridge"]
        M3_4["Zero-Copy SurfaceTexture Rendering"]
        M3_5["OEM Google Night Sight Bypass Flags"]
        M3_6["Post-Inference Color Science (AWB & Saturation)"]
    end

    ROOT --> Sreenandan
    ROOT --> Sreelakshmi
    ROOT --> Sreeveni
```
