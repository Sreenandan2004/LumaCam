import 'package:flutter/material.dart';
import '../../core/constants/gcam_colors.dart';
import '../../core/models/benchmark_metrics.dart';
import '../widgets/before_after_slider.dart';
import '../widgets/gcam_exposure_slider.dart';
import '../widgets/gcam_mode_carousel.dart';
import '../widgets/gcam_shutter_button.dart';
import '../widgets/gcam_top_bar.dart';
import '../widgets/metrics_hud_overlay.dart';
import '../widgets/training_insights_sheet.dart';

/// Main Google Pixel GCam Interface Screen
class GCamCameraScreen extends StatefulWidget {
  const GCamCameraScreen({super.key});

  @override
  State<GCamCameraScreen> createState() => _GCamCameraScreenState();
}

class _GCamCameraScreenState extends State<GCamCameraScreen> {
  // Camera & Model State
  double _exposureAlpha = 100.0;
  bool _isHudVisible = true;
  bool _isCapturing = false;
  int _selectedModeIndex = 1; // Default: 'ADAPTIVE LOW LIGHT'
  bool _useIspPipeline = true;
  int _comparisonSampleIndex = 0;

  final List<String> _cameraModes = [
    'NIGHT SIGHT',
    'ADAPTIVE LOW LIGHT',
    'PRO RAW',
    'CAMERA',
  ];

  // Sample comparison pairs
  final List<Map<String, String>> _samplePairs = [
    {
      'before': 'assets/samples/dark_raw.png',
      'after': 'assets/samples/lumacam_isp.png',
      'beforeLabel': 'RAW BAYER (DARK)',
      'afterLabel': 'LUMACAM + GCAM ISP',
    },
    {
      'before': 'assets/samples/linear_iso.png',
      'after': 'assets/samples/lumacam_enhanced.png',
      'beforeLabel': 'LINEAR ISO BOOST',
      'afterLabel': 'ATTN U-NET GAN',
    },
    {
      'before': 'assets/samples/lumacam_enhanced.png',
      'after': 'assets/samples/lumacam_isp.png',
      'beforeLabel': 'RAW GAN (NO ISP)',
      'afterLabel': 'GAN + COLOR SCIENCE',
    },
  ];

  void _handleShutterPress() async {
    if (_isCapturing) return;

    setState(() => _isCapturing = true);

    // Simulate real Camera2 raw frame capture & neural inference cycle
    await Future.delayed(const Duration(milliseconds: 600));

    if (mounted) {
      setState(() {
        _isCapturing = false;
        // Cycle to next sample on capture to demonstrate multiple scenes
        _comparisonSampleIndex = (_comparisonSampleIndex + 1) % _samplePairs.length;
      });

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          backgroundColor: GCamColors.surfaceElevated,
          content: Row(
            children: [
              const Icon(Icons.check_circle_rounded, color: GCamColors.accentGreen, size: 18),
              const SizedBox(width: 8),
              Text(
                'Captured & Enhanced (${_exposureAlpha.round()}x Gain, ISP Tone-Mapped)',
                style: const TextStyle(color: GCamColors.textPrimary, fontSize: 12),
              ),
            ],
          ),
          duration: const Duration(seconds: 2),
        ),
      );
    }
  }

  void _openTrainingInsights() {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (_) => const TrainingInsightsSheet(),
    );
  }

  @override
  Widget build(BuildContext context) {
    final currentPair = _samplePairs[_comparisonSampleIndex];

    // Compute live metrics based on exposure slider
    final currentMetrics = BenchmarkMetrics(
      exposureAlpha: _exposureAlpha,
      latencyMs: 14.20 + (_exposureAlpha > 150 ? 1.5 : 0.0),
      fps: 1000.0 / (14.20 + (_exposureAlpha > 150 ? 1.5 : 0.0)),
      psnrGainDb: 2.70 + (_exposureAlpha / 300.0) * 1.5,
      currentPsnrDb: 12.13 + (_exposureAlpha / 300.0) * 1.5,
      ispStage: _useIspPipeline
          ? 'Linear Gray-World AWB + Hue Saturation (1.15x)'
          : 'Bypassed (Raw Model Output)',
    );

    return Scaffold(
      backgroundColor: GCamColors.background,
      body: SafeArea(
        child: Column(
          children: [
            // 1. Top GCam Bar
            GCamTopBar(
              isHudVisible: _isHudVisible,
              onToggleHud: () => setState(() => _isHudVisible = !_isHudVisible),
              onOpenTrainingInsights: _openTrainingInsights,
            ),

            // 2. Main Viewfinder (Card with rounded corners)
            Expanded(
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(24),
                  child: Container(
                    color: Colors.black,
                    child: Stack(
                      children: [
                        // Viewfinder Interactive Split Slider
                        Positioned.fill(
                          child: BeforeAfterSlider(
                            key: ValueKey('sample_$_comparisonSampleIndex'),
                            beforeImagePath: currentPair['before']!,
                            afterImagePath: currentPair['after']!,
                            beforeLabel: currentPair['beforeLabel']!,
                            afterLabel: currentPair['afterLabel']!,
                          ),
                        ),

                        // Performance Matrix HUD Overlay
                        if (_isHudVisible)
                          Positioned(
                            top: 12,
                            left: 0,
                            right: 0,
                            child: MetricsHudOverlay(metrics: currentMetrics),
                          ),

                        // Tactile Vertical Exposure Slider (Right Edge)
                        Positioned(
                          right: 12,
                          bottom: 24,
                          child: GCamExposureSlider(
                            alphaValue: _exposureAlpha,
                            onChanged: (val) => setState(() => _exposureAlpha = val),
                          ),
                        ),

                        // Zoom Pills (1x, 2x) at bottom center of viewfinder
                        Positioned(
                          bottom: 12,
                          left: 0,
                          right: 0,
                          child: Center(
                            child: Container(
                              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                              decoration: BoxDecoration(
                                color: Colors.black.withValues(alpha: 0.6),
                                borderRadius: BorderRadius.circular(16),
                              ),
                              child: Row(
                                mainAxisSize: MainAxisSize.min,
                                children: [
                                  _buildZoomPill('1x', true),
                                  const SizedBox(width: 4),
                                  _buildZoomPill('2x', false),
                                ],
                              ),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ),

            // 3. Camera Mode Carousel
            const SizedBox(height: 6),
            GCamModeCarousel(
              modes: _cameraModes,
              selectedIndex: _selectedModeIndex,
              onModeSelected: (idx) => setState(() => _selectedModeIndex = idx),
            ),
            const SizedBox(height: 12),

            // 4. Bottom Shutter Controls Bar
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  // Gallery Thumbnail
                  GestureDetector(
                    onTap: () {
                      setState(() {
                        _comparisonSampleIndex = (_comparisonSampleIndex + 1) % _samplePairs.length;
                      });
                    },
                    child: Container(
                      width: 52,
                      height: 52,
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        border: Border.all(color: Colors.white30, width: 2),
                        image: DecorationImage(
                          image: AssetImage(currentPair['after']!),
                          fit: BoxFit.cover,
                        ),
                      ),
                    ),
                  ),

                  // Center GCam Shutter Button
                  GCamShutterButton(
                    onPressed: _handleShutterPress,
                    isProcessing: _isCapturing,
                  ),

                  // ISP Toggle / Scene Switcher Button
                  IconButton(
                    iconSize: 32,
                    icon: Icon(
                      _useIspPipeline ? Icons.auto_fix_high_rounded : Icons.auto_fix_normal_rounded,
                      color: _useIspPipeline ? GCamColors.accentAmber : GCamColors.textSecondary,
                    ),
                    tooltip: 'Toggle GCam ISP Color Science',
                    onPressed: () {
                      setState(() => _useIspPipeline = !_useIspPipeline);
                      ScaffoldMessenger.of(context).showSnackBar(
                        SnackBar(
                          content: Text(
                            _useIspPipeline
                                ? 'GCam ISP: Gray-World AWB & Saturation Active'
                                : 'GCam ISP: Bypassed (Raw Neural Output)',
                          ),
                          duration: const Duration(seconds: 1),
                        ),
                      );
                    },
                  ),
                ],
              ),
            ),
            const SizedBox(height: 8),
          ],
        ),
      ),
    );
  }

  Widget _buildZoomPill(String label, bool isSelected) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
        color: isSelected ? Colors.white24 : Colors.transparent,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Text(
        label,
        style: TextStyle(
          color: isSelected ? Colors.white : Colors.white60,
          fontSize: 11,
          fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
        ),
      ),
    );
  }
}
