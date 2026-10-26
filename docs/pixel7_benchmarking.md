# Google Pixel 7 On-Device Benchmarking Guide
> **Device**: Google Pixel 7 (Chipset: Google Tensor G2 with EdgeTPU / NPU)  
> **Model Target**: `edge_opt/models/lumacam_generator_int8.onnx` (7.94 MB) / `lumacam_int8.tflite`  
> **Execution Provider**: Android NNAPI / LiteRT GPU Delegate

---

## 1. Why the Pixel 7 Tensor G2 Excels at LumaCam

The Google Pixel 7 is powered by the **Tensor G2** SoC, which features:
* **Custom Google EdgeTPU / NPU**: Purpose-built for INT8 quantized neural operations (matrix multiplication, fused conv-relu, attention soft-masking).
* **Mali-G710 MP7 GPU**: Accelerates float16/INT8 pipelines through the Android Neural Networks API (NNAPI).
* **Camera2 Level 3 Hardware Support**: Full access to uncompressed `RAW_SENSOR` 14-bit Bayer frames.

---

## 2. Model Compression Audit

| Model Format | Precision | File Size | Memory Footprint Reduction | Target Platform |
|---|:---:|:---:|:---:|---|
| PyTorch (`.pt`) | FP32 | 109.09 MB | Baseline (1.0x) | Desktop CUDA Training (GTX 1650 / RTX 3090) |
| ONNX (`.onnx`) | FP32 | 31.16 MB | -71.4% | Universal Inference Engine |
| **Quantized INT8 (`.onnx` / `.tflite`)** | **INT8 (QDQ)** | **7.94 MB** | **-92.7% (-74.5% vs ONNX)** | **Google Pixel 7 (Tensor G2 NPU / EdgeTPU)** |

---

## 3. How to Run On-Device Profiling via ADB (Step-by-Step)

### Step 1: Enable Developer Options on Your Pixel 7
1. Go to **Settings** $\to$ **About phone**.
2. Tap **Build number** 7 times until you see *"You are now a developer!"*.
3. Go to **Settings** $\to$ **System** $\to$ **Developer options** $\to$ Enable **USB debugging**.

### Step 2: Connect Phone to Laptop via USB
In your PowerShell terminal, verify the phone is detected:
```powershell
adb devices
```
*(Your Pixel 7 will show a pop-up saying "Allow USB debugging?". Tap "Always allow from this computer" $\to$ OK).*

### Step 3: Run the Automated Benchmark
Run the provided benchmarking script:
```powershell
powershell -ExecutionPolicy Bypass -File edge_opt/benchmark_pixel7.ps1
```

The script will:
1. Verify device connection (`Google Pixel 7`).
2. Push the lightweight 7.94 MB model to `/data/local/tmp/lumacam/`.
3. Execute the benchmark measuring exact on-device milliseconds per frame and RAM consumption.
4. Output a verified on-device latency score to show in your presentation!
