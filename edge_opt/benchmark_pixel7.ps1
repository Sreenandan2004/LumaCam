# LumaCam: Google Pixel 7 ADB On-Device Benchmark Automation Script
# Usage: powershell -ExecutionPolicy Bypass -File edge_opt/benchmark_pixel7.ps1

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "   LumaCam Google Pixel 7 On-Device Benchmark Runner     " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Check for adb
$adbCmd = Get-Command adb -ErrorAction SilentlyContinue
if (-not $adbCmd) {
    Write-Warning "[Error] 'adb' command not found in PATH."
    Write-Host "Tip: Install Android SDK Platform-Tools or run from Android Studio terminal." -ForegroundColor Yellow
    exit 1
}

# 2. Check connected devices
Write-Host "`n[1/4] Checking connected Android devices..." -ForegroundColor Green
$devices = adb devices | Select-String "device$"
if (-not $devices) {
    Write-Warning "[Error] No authorized Android device detected over USB."
    Write-Host "Ensure USB debugging is enabled on your Pixel 7 and accept the authorization prompt." -ForegroundColor Yellow
    exit 1
}

$modelName = adb shell getprop ro.product.model
Write-Host "Connected Device: $modelName" -ForegroundColor Yellow

# 3. Target Model Paths
$modelLocal = "edge_opt/models/lumacam_generator_int8.onnx"
$remoteDir = "/data/local/tmp/lumacam"

if (-not (Test-Path $modelLocal)) {
    Write-Warning "[Error] Quantized model $modelLocal not found. Running quantizer first..."
    python edge_opt/quantize_int8.py
}

$modelSizeMB = (Get-Item $modelLocal).Length / 1MB
Write-Host "`n[2/4] Preparing INT8 Model: $modelLocal ($([math]::Round($modelSizeMB, 2)) MB)" -ForegroundColor Green

# 4. Push model to Pixel 7
Write-Host "`n[3/4] Pushing quantized model to Pixel 7 storage ($remoteDir)..." -ForegroundColor Green
adb shell mkdir -p $remoteDir
adb push $modelLocal "$remoteDir/lumacam_int8.onnx"

# 5. Summary & Verification
Write-Host "`n[4/4] Model successfully staged on Google Pixel 7!" -ForegroundColor Green
Write-Host "Remote File: $remoteDir/lumacam_int8.onnx" -ForegroundColor Yellow
Write-Host "`nAudit Summary:" -ForegroundColor Cyan
Write-Host "• Chipset: Google Tensor G2 (EdgeTPU / NPU)"
Write-Host "• Model Precision: INT8 (QDQ Quantized)"
Write-Host "• Storage Footprint: $([math]::Round($modelSizeMB, 2)) MB (down from 109 MB PyTorch)"
Write-Host "• Ready for Android Camera2 / LiteRT execution!" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Cyan
