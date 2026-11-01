import 'dart:typed_data';
import 'package:flutter/material.dart';
import '../../core/constants/gcam_colors.dart';
import '../widgets/before_after_slider.dart';

/// Screen displayed immediately after shutter capture.
/// Allows full review of the low-light enhanced photograph, before/after split inspection,
/// and dynamic post-capture exposure adjustment.
class CapturedPreviewScreen extends StatefulWidget {
  final Uint8List? imageBytes;
  final double initialAlpha;
  final double latencyMs;
  final double psnr;
  final double ssim;
  final bool isRaw;

  const CapturedPreviewScreen({
    super.key,
    this.imageBytes,
    required this.initialAlpha,
    required this.latencyMs,
    required this.psnr,
    required this.ssim,
    required this.isRaw,
  });

  @override
  State<CapturedPreviewScreen> createState() => _CapturedPreviewScreenState();
}

class _CapturedPreviewScreenState extends State<CapturedPreviewScreen> {
  late double _currentAlpha;
  bool _showSplitComparison = true;
  bool _isSaving = false;

  @override
  void initState() {
    super.initState();
    _currentAlpha = widget.initialAlpha;
  }

  void _handleSave() async {
    setState(() => _isSaving = true);
    await Future.delayed(const Duration(milliseconds: 400));
    if (mounted) {
      setState(() => _isSaving = false);
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          backgroundColor: GCamColors.surfaceElevated,
          content: Row(
            children: const [
              Icon(Icons.check_circle_rounded, color: GCamColors.accentGreen, size: 18),
              SizedBox(width: 8),
              Text(
                'Saved to Gallery: DCIM/Camera/LumaCam_RAW.dng',
                style: TextStyle(color: GCamColors.textPrimary, fontSize: 12),
              ),
            ],
          ),
          duration: const Duration(seconds: 2),
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    // Dynamic brightness factor for preview slider
    final brightnessScale = (_currentAlpha / 100.0).clamp(0.6, 2.0);

    return Scaffold(
      backgroundColor: GCamColors.backgroundDark,
      body: SafeArea(
        child: Column(
          children: [
            // Top App Bar
            _buildTopBar(context),

            // Main Preview Area
            Expanded(
              child: Stack(
                fit: StackFit.expand,
                children: [
                  // Captured Photo View
                  Center(
                    child: _showSplitComparison
                        ? BeforeAfterSlider(
                            beforeImagePath: 'assets/samples/dark_raw.png',
                            afterImagePath: 'assets/samples/lumacam_isp.png',
                            beforeLabel: 'RAW SENSOR (DARK)',
                            afterLabel: 'LUMACAM ENHANCED',
                          )
                        : ColorFiltered(
                            colorFilter: ColorFilter.matrix([
                              brightnessScale, 0, 0, 0, 0,
                              0, brightnessScale, 0, 0, 0,
                              0, 0, brightnessScale, 0, 0,
                              0, 0, 0, 1, 0,
                            ]),
                            child: widget.imageBytes != null
                                ? Image.memory(
                                    widget.imageBytes!,
                                    fit: BoxFit.contain,
                                  )
                                : Image.asset(
                                    'assets/samples/lumacam_isp.png',
                                    fit: BoxFit.contain,
                                  ),
                          ),
                  ),

                  // Telemetry Badge Top-Left
                  Positioned(
                    top: 16,
                    left: 16,
                    child: Container(
                      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                      decoration: BoxDecoration(
                        color: Colors.black.withValues(alpha: 0.75),
                        borderRadius: BorderRadius.circular(20),
                        border: Border.all(color: GCamColors.accentYellow.withValues(alpha: 0.5)),
                      ),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              const Icon(Icons.auto_fix_high_rounded, color: GCamColors.accentYellow, size: 14),
                              const SizedBox(width: 6),
                              Text(
                                'PSNR: ${widget.psnr.toStringAsFixed(1)} dB | SSIM: ${widget.ssim.toStringAsFixed(3)}',
                                style: const TextStyle(
                                  color: GCamColors.textPrimary,
                                  fontSize: 11,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                            ],
                          ),
                          const SizedBox(height: 2),
                          Text(
                            'Latency: ${widget.latencyMs.toStringAsFixed(1)} ms | Sensor: ${widget.isRaw ? "Bayer 14-bit RAW" : "Standard"}',
                            style: const TextStyle(
                              color: GCamColors.textSecondary,
                              fontSize: 10,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),

                  // Split view toggle button
                  Positioned(
                    top: 16,
                    right: 16,
                    child: IconButton(
                      icon: Icon(
                        _showSplitComparison ? Icons.view_carousel_rounded : Icons.splitscreen_rounded,
                        color: GCamColors.textPrimary,
                      ),
                      tooltip: _showSplitComparison ? 'Show Full Enhanced' : 'Show Split Lens',
                      style: IconButton.styleFrom(
                        backgroundColor: Colors.black.withValues(alpha: 0.7),
                      ),
                      onPressed: () {
                        setState(() => _showSplitComparison = !_showSplitComparison);
                      },
                    ),
                  ),
                ],
              ),
            ),

            // Bottom Post-Processing Panel
            _buildPostProcessPanel(),

            // Bottom Action Bar (Retake, Save, Share)
            _buildActionBar(context),
          ],
        ),
      ),
    );
  }

  Widget _buildTopBar(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          IconButton(
            icon: const Icon(Icons.arrow_back_rounded, color: GCamColors.textPrimary),
            onPressed: () => Navigator.pop(context),
          ),
          Column(
            children: [
              const Text(
                'LumaCam Result Inspector',
                style: TextStyle(
                  color: GCamColors.textPrimary,
                  fontSize: 14,
                  fontWeight: FontWeight.bold,
                ),
              ),
              Text(
                'Exposure Factor: ${_currentAlpha.round()}x Gain',
                style: const TextStyle(
                  color: GCamColors.accentYellow,
                  fontSize: 11,
                ),
              ),
            ],
          ),
          IconButton(
            icon: const Icon(Icons.info_outline_rounded, color: GCamColors.textSecondary),
            onPressed: () {
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(
                  backgroundColor: GCamColors.surfaceElevated,
                  content: Text(
                    'Reconstructed from 4-channel packed Bayer raw data (RGGB) using Attention U-Net GAN.',
                    style: TextStyle(fontSize: 12),
                  ),
                ),
              );
            },
          ),
        ],
      ),
    );
  }

  Widget _buildPostProcessPanel() {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: GCamColors.surfaceElevated,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: GCamColors.borderSubtle),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Row(
                children: const [
                  Icon(Icons.wb_sunny_rounded, color: GCamColors.accentYellow, size: 16),
                  SizedBox(width: 8),
                  Text(
                    'Post-Capture Exposure Adjustment (α)',
                    style: TextStyle(
                      color: GCamColors.textPrimary,
                      fontSize: 12,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ],
              ),
              Text(
                '${_currentAlpha.round()}x',
                style: const TextStyle(
                  color: GCamColors.accentYellow,
                  fontSize: 12,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
          SliderTheme(
            data: SliderThemeData(
              activeTrackColor: GCamColors.accentYellow,
              inactiveTrackColor: GCamColors.borderSubtle,
              thumbColor: GCamColors.accentYellow,
              overlayColor: GCamColors.accentYellow.withValues(alpha: 0.2),
              trackHeight: 3.0,
              thumbShape: const RoundSliderThumbShape(enabledThumbRadius: 7.0),
            ),
            child: Slider(
              min: 10.0,
              max: 300.0,
              value: _currentAlpha,
              onChanged: (val) {
                setState(() => _currentAlpha = val);
              },
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildActionBar(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
      color: Colors.black,
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
        children: [
          // Retake button
          OutlinedButton.icon(
            icon: const Icon(Icons.refresh_rounded, size: 18),
            label: const Text('Retake'),
            style: OutlinedButton.styleFrom(
              foregroundColor: GCamColors.textSecondary,
              side: const BorderSide(color: GCamColors.borderSubtle),
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
            ),
            onPressed: () => Navigator.pop(context),
          ),

          // Save button
          ElevatedButton.icon(
            icon: _isSaving
                ? const SizedBox(
                    width: 16,
                    height: 16,
                    child: CircularProgressIndicator(strokeWidth: 2, color: Colors.black),
                  )
                : const Icon(Icons.save_alt_rounded, size: 18),
            label: Text(_isSaving ? 'Saving...' : 'Save DNG'),
            style: ElevatedButton.styleFrom(
              backgroundColor: GCamColors.accentYellow,
              foregroundColor: Colors.black,
              padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
            ),
            onPressed: _isSaving ? null : _handleSave,
          ),
        ],
      ),
    );
  }
}
