import 'package:flutter/material.dart';
import '../../core/constants/gcam_colors.dart';

/// Minimalist Pixel GCam Top Navigation Bar
class GCamTopBar extends StatelessWidget {
  final bool isHudVisible;
  final VoidCallback onToggleHud;
  final VoidCallback onOpenTrainingInsights;
  final bool isSafeDemoMode;
  final VoidCallback onToggleSafeDemo;
  final VoidCallback? onRotateOrientation;

  const GCamTopBar({
    super.key,
    required this.isHudVisible,
    required this.onToggleHud,
    required this.onOpenTrainingInsights,
    this.isSafeDemoMode = false,
    required this.onToggleSafeDemo,
    this.onRotateOrientation,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          // Night Sight Indicator
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
            decoration: BoxDecoration(
              color: GCamColors.surfaceGlass,
              borderRadius: BorderRadius.circular(20),
              border: Border.all(color: Colors.white12),
            ),
            child: const Row(
              children: [
                Icon(Icons.bedtime_rounded, color: GCamColors.accentAmber, size: 16),
                SizedBox(width: 6),
                Text(
                  'LUMACAM AI',
                  style: TextStyle(
                    color: GCamColors.textPrimary,
                    fontSize: 11,
                    fontWeight: FontWeight.bold,
                    letterSpacing: 0.8,
                  ),
                ),
              ],
            ),
          ),

          // Action Buttons
          Row(
            children: [
              // Fast Live Orientation Adjuster
              if (onRotateOrientation != null)
                IconButton(
                  icon: const Icon(Icons.screen_rotation_rounded, color: GCamColors.accentAmber, size: 20),
                  tooltip: 'Adjust Viewfinder Orientation (Rotate 90°)',
                  onPressed: onRotateOrientation,
                ),

              // Safe Presentation Demo Toggle
              IconButton(
                icon: Icon(
                  isSafeDemoMode ? Icons.shield_rounded : Icons.videocam_rounded,
                  color: isSafeDemoMode ? GCamColors.accentGreen : GCamColors.accentAmber,
                  size: 22,
                ),
                tooltip: isSafeDemoMode
                    ? 'Safe Benchmark Mode (Bypassing OEM Night Sight)'
                    : 'Live Hardware Camera Stream Active',
                onPressed: onToggleSafeDemo,
              ),

              // Training Insights Button
              IconButton(
                icon: const Icon(Icons.insights_rounded, color: GCamColors.accentBlue, size: 22),
                tooltip: 'Kaggle Training & Architecture Insights',
                onPressed: onOpenTrainingInsights,
              ),

              // HUD Metrics Toggle
              IconButton(
                icon: Icon(
                  isHudVisible ? Icons.speed_rounded : Icons.speed_outlined,
                  color: isHudVisible ? GCamColors.accentGreen : GCamColors.textSecondary,
                  size: 22,
                ),
                tooltip: 'Toggle Performance Matrix HUD',
                onPressed: onToggleHud,
              ),

              // Settings Gear
              IconButton(
                icon: const Icon(Icons.settings_outlined, color: GCamColors.textSecondary, size: 22),
                onPressed: () {
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(
                      content: Text('LumaCam Settings: Google Tensor G2 NPU Delegate Active'),
                      duration: Duration(seconds: 2),
                    ),
                  );
                },
              ),
            ],
          ),
        ],
      ),
    );
  }
}
