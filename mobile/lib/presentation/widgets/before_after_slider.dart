import 'dart:typed_data';
import 'package:flutter/material.dart';
import '../../core/constants/gcam_colors.dart';

/// Interactive draggable comparison slider between before and after image states.
/// Supports both pre-rendered assets and real-time in-memory captured frames (Uint8List).
class BeforeAfterSlider extends StatefulWidget {
  final String beforeImagePath;
  final String afterImagePath;
  final Uint8List? beforeImageBytes;
  final Uint8List? afterImageBytes;
  final String beforeLabel;
  final String afterLabel;

  const BeforeAfterSlider({
    super.key,
    this.beforeImagePath = 'assets/samples/dark_raw.png',
    this.afterImagePath = 'assets/samples/lumacam_isp.png',
    this.beforeImageBytes,
    this.afterImageBytes,
    this.beforeLabel = 'RAW SENSOR (DARK)',
    this.afterLabel = 'LUMACAM AI + ISP',
  });

  @override
  State<BeforeAfterSlider> createState() => _BeforeAfterSliderState();
}

class _BeforeAfterSliderState extends State<BeforeAfterSlider> {
  double _splitRatio = 0.5; // Range: 0.0 to 1.0

  Widget _buildAfterImage() {
    if (widget.afterImageBytes != null) {
      return Image.memory(
        widget.afterImageBytes!,
        fit: BoxFit.cover,
        errorBuilder: (context, error, stackTrace) => _buildPlaceholder('After Image'),
      );
    }
    return Image.asset(
      widget.afterImagePath,
      fit: BoxFit.cover,
      errorBuilder: (context, error, stackTrace) => _buildPlaceholder('After Image'),
    );
  }

  Widget _buildBeforeImage() {
    if (widget.beforeImageBytes != null) {
      return Image.memory(
        widget.beforeImageBytes!,
        fit: BoxFit.cover,
        errorBuilder: (context, error, stackTrace) => _buildPlaceholder('Before Image'),
      );
    }
    return Image.asset(
      widget.beforeImagePath,
      fit: BoxFit.cover,
      errorBuilder: (context, error, stackTrace) => _buildPlaceholder('Before Image'),
    );
  }

  Widget _buildPlaceholder(String text) {
    return Container(
      color: GCamColors.surfaceElevated,
      child: Center(
        child: Text(text, style: const TextStyle(color: GCamColors.textSecondary)),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final width = constraints.maxWidth;
        final height = constraints.maxHeight;

        return GestureDetector(
          onHorizontalDragUpdate: (details) {
            setState(() {
              _splitRatio = (_splitRatio + details.delta.dx / width).clamp(0.05, 0.95);
            });
          },
          child: Stack(
            fit: StackFit.expand,
            children: [
              // 1. Bottom Layer: After Image (LumaCam Enhanced + Flagship ISP)
              _buildAfterImage(),

              // 2. Top Layer: Before Image (Raw Dark / Noisy) clipped by splitRatio
              ClipRect(
                clipper: _HorizontalSplitClipper(_splitRatio),
                child: _buildBeforeImage(),
              ),

              // 3. Vertical Divider Line
              Positioned(
                left: width * _splitRatio - 1.5,
                top: 0,
                bottom: 0,
                child: Container(
                  width: 3.0,
                  color: Colors.white.withValues(alpha: 0.9),
                ),
              ),

              // 4. Center Drag Handle Pill
              Positioned(
                left: width * _splitRatio - 20,
                top: height / 2 - 20,
                child: Container(
                  width: 40,
                  height: 40,
                  decoration: BoxDecoration(
                    color: Colors.white,
                    shape: BoxShape.circle,
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withValues(alpha: 0.5),
                        blurRadius: 8,
                        offset: const Offset(0, 2),
                      ),
                    ],
                  ),
                  child: const Icon(
                    Icons.compare_arrows_rounded,
                    color: Colors.black87,
                    size: 24,
                  ),
                ),
              ),

              // 5. Left Label Pill (Before)
              Positioned(
                top: 16,
                left: 16,
                child: _buildBadge(widget.beforeLabel, GCamColors.accentCoral),
              ),

              // 6. Right Label Pill (After)
              Positioned(
                top: 16,
                right: 16,
                child: _buildBadge(widget.afterLabel, GCamColors.accentGreen),
              ),
            ],
          ),
        );
      },
    );
  }

  Widget _buildBadge(String label, Color accentColor) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.65),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: accentColor.withValues(alpha: 0.6), width: 1.0),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 6,
            height: 6,
            decoration: BoxDecoration(
              color: accentColor,
              shape: BoxShape.circle,
            ),
          ),
          const SizedBox(width: 6),
          Text(
            label,
            style: const TextStyle(
              color: GCamColors.textPrimary,
              fontSize: 10,
              fontWeight: FontWeight.w700,
              letterSpacing: 0.6,
            ),
          ),
        ],
      ),
    );
  }
}

class _HorizontalSplitClipper extends CustomClipper<Rect> {
  final double ratio;
  _HorizontalSplitClipper(this.ratio);

  @override
  Rect getClip(Size size) {
    return Rect.fromLTRB(0, 0, size.width * ratio, size.height);
  }

  @override
  bool shouldReclip(_HorizontalSplitClipper oldClipper) => oldClipper.ratio != ratio;
}
