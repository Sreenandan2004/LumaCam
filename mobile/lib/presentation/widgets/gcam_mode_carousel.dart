import 'package:flutter/material.dart';
import '../../core/constants/gcam_colors.dart';

/// Pixel GCam Camera Mode Selector Carousel
class GCamModeCarousel extends StatelessWidget {
  final List<String> modes;
  final int selectedIndex;
  final ValueChanged<int> onModeSelected;

  const GCamModeCarousel({
    super.key,
    required this.modes,
    required this.selectedIndex,
    required this.onModeSelected,
  });

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: 40,
      child: Center(
        child: ListView.separated(
          shrinkWrap: true,
          scrollDirection: Axis.horizontal,
          itemCount: modes.length,
          separatorBuilder: (context, index) => const SizedBox(width: 8),
          itemBuilder: (context, index) {
            final isSelected = index == selectedIndex;
            return GestureDetector(
              onTap: () => onModeSelected(index),
              child: AnimatedContainer(
                duration: const Duration(milliseconds: 180),
                padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
                decoration: BoxDecoration(
                  color: isSelected ? Colors.white.withValues(alpha: 0.18) : Colors.transparent,
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Text(
                  modes[index],
                  style: TextStyle(
                    color: isSelected ? GCamColors.accentAmber : GCamColors.textSecondary,
                    fontSize: 12,
                    fontWeight: isSelected ? FontWeight.w800 : FontWeight.w600,
                    letterSpacing: 0.8,
                  ),
                ),
              ),
            );
          },
        ),
      ),
    );
  }
}
