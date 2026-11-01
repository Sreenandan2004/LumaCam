# LumaCam Live Demo Playbook & Presentation Walkthrough
> **For Capstone Team 21 (SCTCE) — Demo Day Presentation Guide**

---

## 1. Quick Setup Instructions (Before the Evaluators Arrive)

### Station A: Laptop Interactive Desktop Demo (Streamlit)
Open a terminal in `D:\Development\LumaCam` and activate the virtual environment:
```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```
This opens `http://localhost:8501` in your browser.
- Have this window maximized on your laptop / projector screen.
- Pre-load a sample image (e.g. from `mobile/assets/samples/` or default).
- Verify the slider moves smoothly and the 4 panels render properly.

### Station B: Mobile GCam Interface Showcase (Flutter)
You have two great options to demonstrate the Flutter mobile app:

#### Option 1: On your Pixel 7 (Connected via USB)
1. Enable USB Debugging on your Pixel 7: `Settings -> System -> Developer options -> USB debugging`.
2. Connect Pixel 7 via USB-C cable to your laptop.
3. In a new PowerShell terminal:
   ```powershell
   cd D:\Development\LumaCam\mobile
   flutter devices
   # Note the device ID for your Pixel 7
   flutter run -d <pixel7_device_id>
   ```
4. Hold the Pixel 7 in your hands or place it on the presenter table on a phone stand.

#### Option 2: Live Browser/Desktop Preview (Instant Fallback)
If phone connection is inconvenient or during quick projector transitions:
```powershell
cd D:\Development\LumaCam\mobile
flutter run -d edge   # Or -d chrome
```
This runs the exact Pixel GCam UI in an interactive mobile-aspect ratio window directly on your laptop screen!

---

## 2. Minute-by-Minute Live Demo Script (2–3 Minutes)

### Step 1: The Problem Visualized (30 Seconds)
- **Action**: In the Streamlit app or Flutter app, show the **Dark Raw** input.
- **Presenter Spoke**: 
  > *"Respected evaluators, notice this first frame. This is what a smartphone camera captures in sub-1-lux lighting. It is essentially pitch black. If we perform conventional digital brightening—like raising ISO to 12,800—we get this second image: severe color speckles, banding, and complete loss of contrast."*

### Step 2: LumaCam Neural Restoration (45 Seconds)
- **Action**: Switch the view or drag the split-screen slider to reveal **LumaCam Enhanced**.
- **Presenter Spoke**:
  > *"Now observe what happens when LumaCam processes that same raw Bayer frame. Using our 4-channel Attention U-Net and sub-pixel reconstruction, the noise is filtered out in raw Bayer space before demosaicing. Edges and textures remain crisp without any waxy blur, because our PatchGAN discriminator actively penalized synthetic smoothing during training."*

### Step 3: Interactive Exposure Slider ($\alpha$) (45 Seconds)
- **Action**: On the Flutter GCam UI (or Streamlit slider), smoothly drag the exposure slider from $50\times$ up to $250\times$.
- **Presenter Spoke**:
  > *"Notice this vertical GCam exposure dial on the right. In existing deep learning solutions, the amplification factor is fixed. In LumaCam, our bottleneck FiLM layer dynamically modulates the latent features. As I drag this slider, the exposure responds continuously and naturally, allowing the user to adapt to scene brightness in real time."*

### Step 4: The Hardware Metrics HUD (30 Seconds)
- **Action**: Tap or point to the **Performance HUD** in the Flutter app / benchmark graphs.
- **Presenter Spoke**:
  > *"Finally, looking at our live hardware telemetry: on our local GPU, inference takes just 14.2 milliseconds, equivalent to 70.4 FPS. With our static INT8 quantization, the model weighs only 7.94 MB—a 74.5% reduction—allowing it to run comfortably within the 33ms real-time threshold on a mobile processor like Google's Tensor G2."*

---

## 3. Presentation Slides & Asset Mapping

All presentation visual assets are conveniently located in `project_presentation/assets/`:
1. `architecture_block_diagram.png`: Main pipeline block diagram (Slides 1, 5, 6, 16).
2. `pipeline_comparison.png`: 4-stage visual comparison from raw to ISP (Slides 2, 11).
3. `attention_gate_heatmaps.png`: 4-scale Explainable AI attention maps (Slides 7, 12).
4. `training_curves.png`: 500-epoch Generator/Discriminator loss & validation PSNR (Slides 8, 13).
5. `quality_metrics_comparison.png`: PSNR & SSIM vs literature baselines (Slide 13).
6. `latency_throughput_benchmark.png`: 14.2ms / 70.4 FPS latency vs Diffusion models (Slides 3, 9, 14).
7. `dark_raw.png`, `linear_iso.png`, `lumacam_enhanced.png`, `lumacam_isp.png`: Individual high-res crops for granular detail slides.
