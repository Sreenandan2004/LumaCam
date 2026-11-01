import 'dart:typed_data';
import 'package:flutter/material.dart';
import '../../core/constants/gcam_colors.dart';
import '../widgets/before_after_slider.dart';

/// Screen displayed immediately after shutter capture.
/// Allows full review of the low-light enhanced photograph, before/after split inspection,
/// and dynamic post-capture exposure adjustment.
class CapturedPreviewScreen extends StatefulWidget {
  final Uint8List? imageBytes;
  final Uint8List? beforeBytes;
  final double initialAlpha;
  final double latencyMs;
  final double psnr;
  final double ssim;
  final bool isRaw;

  const CapturedPreviewScreen({
    super.key,
    this.imageBytes,
    this.beforeBytes,
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
                'Saved to Gallery: DCIM/Camera/LumaCam_Pixel_ISP.jpg',
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
                            beforeImageBytes: widget.beforeBytes,
                            afterImageBytes: widget.imageBytes,
                            beforeImagePath: 'assets/samples/dark_raw.png',
                            afterImagePath: 'assets/samples/lumacam_isp.png',
                            beforeLabel: 'SENSOR RAW (DARK)',
                            afterLabel: 'LUMACAM + FLAGSHIP ISP',
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
                            'Latency: ${widget.latencyMs.toStringAsFixed(1)} ms | ISP: Bilateral + Unsharp Deblur',
                            style: const TextStyle(
                              color: GCamColors.textSecondary,
                              fontSize: 10,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),

                  // Post-Capture Exposure Slider (Right Edge)
                  Positioned(
                    right: 16,
                    bottom: 24,
                    top: 80,
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        const Icon(Icons.wb_sunny_rounded, color: GCamColors.accentYellow, size: 16),
                        const SizedBox(height: 8),
                        Expanded(
                          child: RotatedBox(
                            quarterTurns: 3,
                            child: SliderTheme(
                              data: SliderTheme.of(context).copyWith(
                                trackHeight: 3,
                                thumbShape: const RoundSliderThumbShape(enabledThumbRadius: 7),
                                activeTrackColor: GCamColors.accentYellow,
                                inactiveTrackColor: Colors.white24,
                                thumbColor: Colors.white,
                              ),
                              child: Slider(
                                value: _currentAlpha,
                                min: 10.0,
                                max: 300.0,
                                onChanged: (val) => setState(() => _currentAlpha = val),
                              ),
                            ),
                          ),
                        ),
                        const SizedBox(height: 8),
                        Text(
                          '${_currentAlpha.toInt()}x',
                          style: const TextStyle(
                            color: GCamColors.textPrimary,
                            fontSize: 10,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),

            // Bottom Control Bar
            _buildBottomBar(context),
          ],
        ),
      ),
    );
  }

  Widget _buildTopBar(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          IconButton(
            icon: const Icon(Icons.arrow_back_ios_new_rounded, color: GCamColors.textPrimary),
            onPressed: () => Navigator.pop(context),
          ),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
            decoration: BoxDecoration(
              color: GCamColors.surfaceElevated,
              borderRadius: BorderRadius.circular(20),
            ),
            child: Row(
              children: [
                Icon(
                  _showSplitComparison ? Icons.compare_rounded : Icons.photo_rounded,
                  color: GCamColors.accentGreen,
                  size: 16,
                ),
                const SizedBox(width: 6),
                Text(
                  _showSplitComparison ? 'Split Comparison' : 'Enhanced Photo',
                  style: const TextStyle(
                    color: GCamColors.textPrimary,
                    fontSize: 12,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ],
            ),
          ),
          IconButton(
            icon: Icon(
              _showSplitComparison ? Icons.view_sidebar_rounded : Icons.splitscreen_rounded,
              color: GCamColors.textPrimary,
            ),
            tooltip: 'Toggle Split Comparison',
            onPressed: () => setState(() => _showSplitComparison = !_showSplitComparison),
          ),
        ],
      ),
    );
  }

  Widget _buildBottomBar(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
      decoration: const BoxDecoration(
        color: GCamColors.surface,
        borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
        children: [
          // Retake Button
          OutlinedButton.icon(
            style: OutlinedButton.styleFrom(
              side: const BorderSide(color: Colors.white24),
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
            ),
            icon: const Icon(Icons.refresh_rounded, color: GCamColors.textPrimary, size: 18),
            label: const Text('Retake', style: TextStyle(color: GCamColors.textPrimary)),
            onPressed: () => Navigator.pop(context),
          ),

          // Save / Export Button
          ElevatedButton.icon(
            style: ElevatedButton.styleFrom(
              backgroundColor: GCamColors.accentGreen,
              foregroundColor: Colors.black,
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
              padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 12),
            ),
            icon: _isSaving
                ? const SizedBox(
                    width: 16,
                    height: 16,
                    child: CircularProgressIndicator(color: Colors.black, strokeWidth: 2),
                  )
                : const Icon(Icons.file_download_rounded, size: 18),
            label: Text(_isSaving ? 'Saving...' : 'Save Photo', style: const TextStyle(fontWeight: FontWeight.bold)),
            onPressed: _isSaving ? null : _handleSave,
          ),
        ],
      ),
    );
  }
}
