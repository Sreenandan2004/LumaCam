import 'package:flutter/material.dart';
import '../../core/constants/gcam_colors.dart';

/// Iconic Google Pixel GCam Shutter Button with tactile press animation.
class GCamShutterButton extends StatefulWidget {
  final VoidCallback onPressed;
  final bool isProcessing;

  const GCamShutterButton({
    super.key,
    required this.onPressed,
    this.isProcessing = false,
  });

  @override
  State<GCamShutterButton> createState() => _GCamShutterButtonState();
}

class _GCamShutterButtonState extends State<GCamShutterButton>
    with SingleTickerProviderStateMixin {
  bool _isPressed = false;

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTapDown: (_) => setState(() => _isPressed = true),
      onTapUp: (_) {
        setState(() => _isPressed = false);
        widget.onPressed();
      },
      onTapCancel: () => setState(() => _isPressed = false),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 120),
        width: 82,
        height: 82,
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          border: Border.all(
            color: widget.isProcessing ? GCamColors.accentAmber : GCamColors.shutterRing,
            width: 4.0,
          ),
          boxShadow: widget.isProcessing
              ? [
                  BoxShadow(
                    color: GCamColors.accentAmber.withValues(alpha: 0.5),
                    blurRadius: 16,
                    spreadRadius: 2,
                  )
                ]
              : null,
        ),
        child: Center(
          child: AnimatedContainer(
            duration: const Duration(milliseconds: 120),
            width: _isPressed ? 60 : (widget.isProcessing ? 34 : 68),
            height: _isPressed ? 60 : (widget.isProcessing ? 34 : 68),
            decoration: BoxDecoration(
              color: widget.isProcessing ? GCamColors.accentAmber : GCamColors.shutterInner,
              shape: BoxShape.circle,
            ),
            child: widget.isProcessing
                ? const Padding(
                    padding: EdgeInsets.all(6),
                    child: CircularProgressIndicator(
                      strokeWidth: 2.5,
                      valueColor: AlwaysStoppedAnimation<Color>(Colors.black),
                    ),
                  )
                : null,
          ),
        ),
      ),
    );
  }
}
