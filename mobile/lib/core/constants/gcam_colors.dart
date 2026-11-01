import 'package:flutter/material.dart';

/// Pixel GCam Design Tokens and Color Palette
class GCamColors {
  GCamColors._();

  static const Color background = Color(0xFF0F0F12);
  static const Color surface = Color(0xFF1B1B1F);
  static const Color surfaceElevated = Color(0xFF26262B);
  static const Color surfaceGlass = Color(0xCC1B1B1F);

  // Pixel Accent Palette
  static const Color accentAmber = Color(0xFFFDD835); // Exposure sun / slider
  static const Color accentBlue = Color(0xFF8AB4F8);  // Google Material Blue
  static const Color accentGreen = Color(0xFF81C995); // Good performance / Success
  static const Color accentCoral = Color(0xFFF28B82); // Low light warning
  static const Color accentPurple = Color(0xFFD7AEFB); // Attention Gate XAI

  // Typography Colors
  static const Color textPrimary = Color(0xFFFFFFFF);
  static const Color textSecondary = Color(0xFF9AA0A6);
  static const Color textTertiary = Color(0xFF5F6368);

  // Shutter & Controls
  static const Color shutterRing = Color(0xFFFFFFFF);
  static const Color shutterInner = Color(0xFFEEEEEE);
  static const Color sliderTrack = Color(0x40FFFFFF);
  static const Color sliderActive = Color(0xFFFDD835);
}
