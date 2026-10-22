# 🔵 Veni (Sreeveni R) — Task Handoff for AI Agent

> **Role**: Flutter GCam UI & Evaluation Script
> **Branch**: `feature/app-flutter-ui`
> **Hardware**: Laptop (no GPU needed — UI work + simple Python scripts)
> **AI Tool**: Use **Claude Code** or any AI coding agent with this document as context
> **Git Remote**: `https://github.com/Sreenandan2004/LumaCam.git`

---

## Project Context

**LumaCam** is a real-time low-light image enhancement mobile camera app for Android. The app uses a deep learning model that takes dark camera frames and an exposure slider value, and outputs a bright, enhanced image.

**Your job**: Build the **Flutter mobile UI** that looks and feels like Google Pixel's GCam app, plus a Python evaluation script.

### Repository Location & Setup
```bash
# Clone and switch to your branch
git clone https://github.com/Sreenandan2004/LumaCam.git
cd LumaCam
git checkout feature/app-flutter-ui
```

### Repository Structure (existing)
```
LumaCam/
├── model/                          # Python ML code (DON'T TOUCH)
│   ├── networks/                   # Neural network modules
│   │   ├── generator.py            # The main AI model
│   │   ├── discriminator.py        # Training helper model
│   │   ├── attention_gate.py       # Attention module
│   │   └── conditioning.py         # Exposure control module
│   └── datasets/
│       └── raw_processing.py       # Image processing utilities
├── mobile/                         # ← YOUR WORK GOES HERE
└── ...
```

---

## Task 9: Flutter App Scaffold (P0 — Day 1-2)

### What to Do

Create a complete Flutter project inside the `mobile/` directory.

### Step-by-Step

```bash
# Navigate to project root
cd D:\Development\LumaCam    # (or wherever you cloned it)

# Create Flutter project
flutter create --org com.lumacam --project-name lumacam mobile

# Enter the project
cd mobile

# Add dependencies
flutter pub add camera
flutter pub add flutter_bloc
flutter pub add path_provider
flutter pub add permission_handler
flutter pub add image_picker
flutter pub add google_fonts
```

### Android Configuration

Edit `mobile/android/app/build.gradle`:
```groovy
android {
    compileSdk = 34

    defaultConfig {
        applicationId = "com.lumacam.lumacam"
        minSdk = 24          // Camera2 API requires 24+
        targetSdk = 34
        versionCode = 1
        versionName = "0.2.0"
    }
}
```

### Create Folder Structure

```
mobile/lib/
├── main.dart                           # App entry point
├── core/
│   ├── constants.dart                  # Colors, dimensions, strings
│   └── theme.dart                      # Dark theme configuration
├── presentation/
│   ├── screens/
│   │   └── camera_screen.dart          # Main camera viewfinder screen
│   └── widgets/
│       ├── exposure_slider.dart        # Vertical EV slider
│       ├── shutter_button.dart         # Animated capture button
│       ├── mode_carousel.dart          # Bottom mode selector
│       └── camera_controls.dart        # Top bar (flash, timer, settings)
└── bloc/
    ├── camera_bloc.dart                # Camera state management
    ├── camera_event.dart               # Camera events
    └── camera_state.dart               # Camera states
```

---

## Task 10: GCam-Style UI (P0 — Day 2-6)

### UI Design Specification

Build a camera UI that matches this layout:

```
┌─────────────────────────────────────────┐
│  ⚡  HDR   ⏱   ⚙                       │  ← Top controls bar (dark semi-transparent)
├─────────────────────────────────────────┤
│                                         │
│                                         │
│                                         │
│          CAMERA VIEWFINDER              │
│        (CameraPreview widget)           │
│        Full screen, black bg            │
│                                         │
│   ┌───┐                                 │
│   │+3 │                                 │
│   │   │  ← Vertical exposure slider     │
│   │ 0 │    on LEFT edge                 │
│   │   │    Draggable thumb              │
│   │-3 │    Shows EV value               │
│   └───┘                                 │
│                                         │
├─────────────────────────────────────────┤
│                                         │
│   Night Sight · Low Light · Standard    │  ← Mode carousel (horizontal)
│                                         │
├─────────────────────────────────────────┤
│                                         │
│   ◻         ◉              ↻           │  ← Gallery  Shutter  Flip Camera
│  (last     (large,        (icon)        │
│  photo)    animated)                    │
│                                         │
└─────────────────────────────────────────┘
```

### Color & Style Constants

Create `mobile/lib/core/constants.dart`:
```dart
import 'package:flutter/material.dart';

class LumaCamColors {
  static const Color background = Color(0xFF000000);        // Pure black
  static const Color surface = Color(0xFF1A1A1A);            // Dark grey
  static const Color primary = Color(0xFFFFFFFF);            // White
  static const Color accent = Color(0xFF4FC3F7);             // Light blue accent
  static const Color sliderTrack = Color(0x80FFFFFF);        // 50% white
  static const Color sliderActive = Color(0xFFFFFFFF);       // White
  static const Color modeSelected = Color(0xFFFFFFFF);       // White text
  static const Color modeUnselected = Color(0x80FFFFFF);     // 50% white text
  static const Color shutterOuter = Color(0xFFFFFFFF);       // White ring
  static const Color shutterInner = Color(0xFFFFFFFF);       // White fill
  static const Color controlBar = Color(0x40000000);         // Semi-transparent
}

class LumaCamDimens {
  static const double shutterSize = 72.0;
  static const double shutterRingWidth = 4.0;
  static const double sliderWidth = 40.0;
  static const double sliderHeight = 200.0;
  static const double modeCarouselHeight = 48.0;
  static const double bottomBarHeight = 100.0;
  static const double topBarHeight = 56.0;
  static const double galleryThumbSize = 48.0;
}
```

### Widget Specifications

#### 1. `camera_screen.dart` — Main Screen

```dart
/// Full-screen camera viewfinder with overlaid controls.
/// 
/// Layout (Stack):
///   - Background: CameraPreview (or black placeholder if no camera)
///   - Top: Semi-transparent control bar (flash, HDR, timer, settings)
///   - Left: Vertical exposure slider
///   - Bottom: Mode carousel + shutter bar
///
/// State: Managed by CameraBloc
///   - cameraController: CameraController from 'camera' package
///   - exposureValue: double (-3.0 to +3.0)
///   - selectedMode: enum (nightSight, lowLight, standard)
///   - isCapturing: bool (for shutter animation)
class CameraScreen extends StatefulWidget { ... }
```

**Key behaviors**:
- Initialize `CameraController` with `ResolutionPreset.max` in `initState`
- Request camera and storage permissions on first launch
- Dispose camera controller properly in `dispose()`
- Handle camera errors gracefully (show error message, not crash)

#### 2. `exposure_slider.dart` — Vertical Exposure Slider

```dart
/// Vertical slider on the left edge of the viewfinder.
/// 
/// Range: -3.0 to +3.0 (representing EV compensation)
/// This value will later be mapped to the model's exposure conditioning alpha:
///   alpha = map(ev, -3.0, +3.0, 0.5, 3.0)
/// ⚠️ OPEN DECISION OD-1 (docs/api_specifications.md § 7): the model is trained on alpha ≈ 100-300
///   (exposure ratios), so this 0.5-3.0 mapping is out of distribution. Keep the slider in EV
///   units for now and do not wire the mapping until OD-1 is decided.
///
/// Visual design:
///   - Thin vertical track (2px, white 50% opacity)
///   - Circular white thumb (16px diameter)
///   - Current EV value label next to thumb (e.g., "+1.5")
///   - Sun icon (☀) at top (+3), moon icon (🌙) at bottom (-3)
///
/// Props:
///   - value: double (current EV, -3.0 to +3.0)
///   - onChanged: Function(double) callback
class ExposureSlider extends StatelessWidget {
  final double value;
  final ValueChanged<double> onChanged;
  ...
}
```

#### 3. `shutter_button.dart` — Animated Capture Button

```dart
/// Large circular capture button centered at the bottom.
///
/// Visual design:
///   - Outer ring: 72px diameter, 4px white border
///   - Inner circle: 60px diameter, solid white fill
///   - On press: inner circle scales down to 50px (AnimatedScale)
///   - On capture: brief white flash overlay on viewfinder
///
/// Props:
///   - onPressed: VoidCallback (triggers photo capture)
///   - isCapturing: bool (controls animation state)
class ShutterButton extends StatefulWidget {
  final VoidCallback onPressed;
  final bool isCapturing;
  ...
}
```

**Animation**: Use `AnimatedContainer` or `ScaleTransition` for the press-down effect. Duration: 150ms.

#### 4. `mode_carousel.dart` — Bottom Mode Selector

```dart
/// Horizontal scrollable mode selector.
///
/// Modes: ["Night Sight", "Low Light", "Standard"]
/// 
/// Visual design:
///   - Horizontally centered row of text labels
///   - Selected mode: white text, slightly larger font (16sp)
///   - Unselected modes: 50% white text, smaller font (14sp)
///   - Animated underline indicator on selected mode
///   - Swipe left/right to change mode (PageView or TabBar)
///
/// Props:
///   - selectedIndex: int
///   - onModeChanged: Function(int) callback
enum CameraMode { nightSight, lowLight, standard }

class ModeCarousel extends StatelessWidget {
  final CameraMode selectedMode;
  final ValueChanged<CameraMode> onModeChanged;
  ...
}
```

#### 5. `camera_controls.dart` — Top Control Bar

```dart
/// Semi-transparent top bar with camera utility buttons.
///
/// Layout: Row with IconButtons
///   - Flash toggle (⚡ off / on / auto)
///   - HDR toggle
///   - Timer (off / 3s / 10s)
///   - Settings gear
///
/// All icons: white, 24px, Material Icons
class CameraControls extends StatelessWidget {
  final FlashMode flashMode;
  final VoidCallback onFlashToggle;
  final VoidCallback onSettingsPressed;
  ...
}
```

### BLoC State Management

#### `camera_event.dart`
```dart
abstract class CameraEvent {}
class InitializeCamera extends CameraEvent {}
class ChangeExposure extends CameraEvent {
  final double value;  // EV -3.0 to +3.0
  ChangeExposure(this.value);
}
class ChangeMode extends CameraEvent {
  final CameraMode mode;
  ChangeMode(this.mode);
}
class CapturePhoto extends CameraEvent {}
class FlipCamera extends CameraEvent {}
class ToggleFlash extends CameraEvent {}
```

#### `camera_state.dart`
```dart
class CameraState {
  final CameraController? controller;
  final double exposureValue;      // -3.0 to +3.0
  final CameraMode selectedMode;   // nightSight, lowLight, standard
  final bool isCapturing;
  final bool isInitialized;
  final String? lastPhotoPath;
  final String? errorMessage;
  ...
}
```

#### `camera_bloc.dart`
```dart
class CameraBloc extends Bloc<CameraEvent, CameraState> {
  CameraBloc() : super(CameraState.initial()) {
    on<InitializeCamera>(_onInitialize);
    on<ChangeExposure>(_onChangeExposure);
    on<ChangeMode>(_onChangeMode);
    on<CapturePhoto>(_onCapture);
    on<FlipCamera>(_onFlip);
    on<ToggleFlash>(_onToggleFlash);
  }
  // ... event handlers
}
```

### `main.dart`

```dart
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'presentation/screens/camera_screen.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  
  // Force portrait orientation and fullscreen
  SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);
  SystemChrome.setEnabledSystemUIMode(SystemUiMode.immersiveSticky);
  
  runApp(const LumaCamApp());
}

class LumaCamApp extends StatelessWidget {
  const LumaCamApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'LumaCam',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark().copyWith(
        scaffoldBackgroundColor: Colors.black,
      ),
      home: const CameraScreen(),
    );
  }
}
```

### Testing

```bash
# Run on Android emulator (or connected device)
cd mobile
flutter run

# Run widget tests
flutter test

# Check for lint issues
flutter analyze
```

---

## Task 8: Evaluation Metrics Script (P1 — Week 2, Day 8-11)

### File to Create
```
model/evaluate.py
```

### What to Build

A Python script that loads a trained generator checkpoint, runs inference on test image pairs, and computes PSNR and SSIM metrics.

```python
"""
LumaCam Model Evaluation Script
================================
Loads a trained ExposureConditionedAttentionUNet checkpoint and evaluates it
on SID Sony test pairs, computing PSNR and SSIM metrics.

Usage:
    python model/evaluate.py --checkpoint checkpoints/best_generator.pth \
                             --data-dir path/to/SID/Sony \
                             --test-list Sony_test_list.txt \
                             --output-dir evaluation_results/

Dependencies:
    pip install torch torchvision torchmetrics rawpy numpy pillow tqdm
"""

import argparse
import os
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import torch
import rawpy
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from PIL import Image
from tqdm import tqdm

# Import from project
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from model.networks.generator import ExposureConditionedAttentionUNet
from model.datasets.raw_processing import pack_raw_bayer_np


def load_model(checkpoint_path: str, device: torch.device) -> ExposureConditionedAttentionUNet:
    """
    Load the generator model from a .pth checkpoint.
    
    Args:
        checkpoint_path: Path to the saved .pth file
        device: torch device (cuda or cpu)
    
    Returns:
        Model in eval mode with loaded weights
    """
    model = ExposureConditionedAttentionUNet(
        in_channels=4, out_channels=3, base_channels=32, cond_dim=1
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Handle both raw state_dict and wrapped checkpoint formats
    if 'generator_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['generator_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def parse_exposure_from_filename(filename: str) -> float:
    """
    Extract exposure time in seconds from SID filename.
    Example: "00001_00_0.1s.ARW" -> 0.1
    
    Args:
        filename: The .ARW filename
    Returns:
        Exposure time as float
    """
    import re
    match = re.search(r'(\d+\.?\d*)s', filename)
    if match:
        return float(match.group(1))
    raise ValueError(f"Could not parse exposure time from: {filename}")


def process_test_pair(
    model: ExposureConditionedAttentionUNet,
    short_path: str,
    long_path: str,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Process one test pair: load raw, pack, run inference.
    
    Args:
        model: Loaded generator
        short_path: Path to short-exposure .ARW file
        long_path: Path to long-exposure .ARW file (ground truth)
        device: torch device
    
    Returns:
        (enhanced, ground_truth) both as (1, 3, H, W) tensors in [0, 1]
    """
    # 1. Load and pack short-exposure raw
    raw_short = rawpy.imread(short_path)
    raw_data = raw_short.raw_image_visible.copy().astype(np.float32)
    packed = pack_raw_bayer_np(raw_data)  # (H/2, W/2, 4)
    
    # 2. Compute exposure ratio
    short_exp = parse_exposure_from_filename(os.path.basename(short_path))
    long_exp = parse_exposure_from_filename(os.path.basename(long_path))
    alpha = long_exp / short_exp
    
    # 3. Apply amplification and convert to tensor
    packed = packed * alpha
    input_tensor = torch.from_numpy(packed).permute(2, 0, 1).unsqueeze(0).to(device)  # (1, 4, H/2, W/2)
    alpha_tensor = torch.tensor([[alpha]], dtype=torch.float32, device=device)
    
    # 4. Run inference
    with torch.no_grad():
        enhanced, _ = model(input_tensor, alpha_tensor)  # (1, 3, H, W)
    
    # 5. Load ground truth
    raw_long = rawpy.imread(long_path)
    gt_rgb = raw_long.postprocess(
        use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16
    )  # (H, W, 3) uint16
    gt_float = gt_rgb.astype(np.float32) / 65535.0  # normalize to [0, 1]
    gt_tensor = torch.from_numpy(gt_float).permute(2, 0, 1).unsqueeze(0).to(device)  # (1, 3, H, W)
    
    # 6. Crop to match sizes if needed (enhanced may differ from GT)
    min_h = min(enhanced.shape[2], gt_tensor.shape[2])
    min_w = min(enhanced.shape[3], gt_tensor.shape[3])
    enhanced = enhanced[:, :, :min_h, :min_w]
    gt_tensor = gt_tensor[:, :, :min_h, :min_w]
    
    return enhanced, gt_tensor


def main():
    parser = argparse.ArgumentParser(description='LumaCam Model Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to generator .pth checkpoint')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to SID Sony dataset root')
    parser.add_argument('--test-list', type=str, default='Sony_test_list.txt', help='Test pair list filename')
    parser.add_argument('--output-dir', type=str, default=None, help='Save enhanced images here')
    parser.add_argument('--device', type=str, default='auto', help='cuda or cpu')
    args = parser.parse_args()
    
    # Device setup
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.checkpoint, device)
    print(f"Loaded model from: {args.checkpoint}")
    
    # Metrics
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    # Parse test list
    test_list_path = os.path.join(args.data_dir, args.test_list)
    with open(test_list_path, 'r') as f:
        pairs = [line.strip().split() for line in f if line.strip()]
    
    # Output directory
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Evaluate
    results = []
    for pair in tqdm(pairs, desc="Evaluating"):
        short_path = os.path.join(args.data_dir, pair[0])
        long_path = os.path.join(args.data_dir, pair[1])
        name = os.path.basename(short_path).replace('.ARW', '')
        
        try:
            enhanced, gt = process_test_pair(model, short_path, long_path, device)
            
            psnr_val = psnr_metric(enhanced, gt).item()
            ssim_val = ssim_metric(enhanced, gt).item()
            results.append({'name': name, 'psnr': psnr_val, 'ssim': ssim_val})
            
            # Optionally save enhanced image
            if args.output_dir:
                img = (enhanced[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                Image.fromarray(img).save(os.path.join(args.output_dir, f"{name}_enhanced.png"))
                
        except Exception as e:
            print(f"Error processing {name}: {e}")
            continue
    
    # Print results table
    print("\n" + "=" * 50)
    print(f"{'Image':<30} {'PSNR (dB)':<12} {'SSIM':<8}")
    print("-" * 50)
    for r in results:
        print(f"{r['name']:<30} {r['psnr']:<12.2f} {r['ssim']:<8.4f}")
    print("-" * 50)
    
    if results:
        mean_psnr = sum(r['psnr'] for r in results) / len(results)
        mean_ssim = sum(r['ssim'] for r in results) / len(results)
        print(f"{'MEAN':<30} {mean_psnr:<12.2f} {mean_ssim:<8.4f}")
    print("=" * 50)


if __name__ == '__main__':
    main()
```

---

## Git Workflow for Veni

### Daily Commands
```bash
# Start of day: pull latest
git pull origin feature/app-flutter-ui

# After making changes:
git add mobile/                              # Week 1: Flutter UI
git add model/evaluate.py                    # Week 2: Evaluation script
git commit -m "feat(ui): implement exposure slider widget"
git push origin feature/app-flutter-ui

# When task is complete: Create Pull Request on GitHub
# Go to: https://github.com/Sreenandan2004/LumaCam/compare
# Base: dev  ←  Compare: feature/app-flutter-ui
# Assign Sreenandan as reviewer
```

### Commit Message Examples
```
feat(ui): scaffold Flutter project with camera dependencies
feat(ui): build camera viewfinder screen with CameraPreview
feat(ui): add vertical exposure slider with EV labels
feat(ui): implement animated shutter button with scale transition
feat(ui): add mode carousel (Night Sight / Low Light / Standard)
feat(ui): add top control bar with flash toggle
feat(evaluate): implement PSNR/SSIM evaluation script
```

---

## Verification Checklist

- [ ] `cd mobile && flutter analyze` — no errors
- [ ] `flutter run` on Android emulator — viewfinder shows camera preview
- [ ] Exposure slider drags smoothly, shows EV value
- [ ] Shutter button animates on press
- [ ] Mode carousel scrolls between 3 modes
- [ ] Camera flip button toggles front/back camera
- [ ] `python model/evaluate.py --help` — prints usage
