import 'package:flutter/material.dart';
import '../../core/constants/gcam_colors.dart';

/// Pixel GCam Style Vertical Exposure Slider with dynamic tactile feedback.
class GCamExposureSlider extends StatelessWidget {
  final double alphaValue; // Range e.g. 1.0 to 300.0
  final ValueChanged<double> onChanged;

  const GCamExposureSlider({
    super.key,
    required this.alphaValue,
    required this.onChanged,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 52,
      height: 240,
      padding: const EdgeInsets.symmetric(vertical: 12),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.55),
        borderRadius: BorderRadius.circular(30),
        border: Border.all(color: Colors.white.withOpacity(0.15), width: 1.0),
      ),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          // Exposure Value Badge
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
            decoration: BoxDecoration(
              color: GCamColors.accentAmber.withOpacity(0.2),
              borderRadius: BorderRadius.circular(8),
            ),
            child: Text(
              '${alphaValue.round()}x',
              style: const TextStyle(
                color: GCamColors.accentAmber,
                fontSize: 10,
                fontWeight: FontWeight.bold,
              ),
            ),
          ),

          // Vertical Rotated Slider
          Expanded(
            child: RotatedBox(
              quarterTurns: 3,
              child: SliderTheme(
                data: SliderThemeData(
                  trackHeight: 3.0,
                  activeTrackColor: GCamColors.accentAmber,
                  inactiveTrackColor: Colors.white.withOpacity(0.2),
                  thumbColor: Colors.white,
                  thumbShape: const RoundSliderThumbShape(enabledThumbRadius: 7.0),
                  overlayColor: GCamColors.accentAmber.withOpacity(0.2),
                  overlayShape: const RoundSliderOverlayShape(overlayRadius: 16.0),
                ),
                child: Slider(
                  value: alphaValue.clamp(1.0, 300.0),
                  min: 1.0,
                  max: 300.0,
                  onChanged: onChanged,
                ),
              ),
            ),
          ),

          // Sun Icon
          const Icon(
            Icons.wb_sunny_rounded,
            color: GCamColors.accentAmber,
            size: 18,
          ),
        ],
      ),
    );
  }
}
