import 'package:flutter/material.dart';
import '../../core/constants/gcam_colors.dart';
import '../../core/models/benchmark_metrics.dart';

/// Real-time Glassmorphism HUD Overlay showing Hardware Metrics and ISP Stages.
class MetricsHudOverlay extends StatelessWidget {
  final BenchmarkMetrics metrics;

  const MetricsHudOverlay({
    super.key,
    required this.metrics,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 16),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.72),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.white.withValues(alpha: 0.18), width: 1.0),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.4),
            blurRadius: 10,
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          // Header with Device & NPU status
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Row(
                children: [
                  Container(
                    width: 8,
                    height: 8,
                    decoration: const BoxDecoration(
                      color: GCamColors.accentGreen,
                      shape: BoxShape.circle,
                    ),
                  ),
                  const SizedBox(width: 8),
                  Text(
                    metrics.activeDevice,
                    style: const TextStyle(
                      color: GCamColors.textPrimary,
                      fontSize: 11,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ],
              ),
              Text(
                '${metrics.modelSizeMb} MB (${metrics.modelPrecision})',
                style: const TextStyle(
                  color: GCamColors.accentBlue,
                  fontSize: 10,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          const Divider(height: 1, color: Colors.white12),
          const SizedBox(height: 8),

          // Metric Grid
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceAround,
            children: [
              _buildStat('LATENCY', '${metrics.latencyMs.toStringAsFixed(1)} ms', GCamColors.accentAmber),
              _buildStat('FPS', metrics.fps.toStringAsFixed(0), GCamColors.accentGreen),
              _buildStat('GAIN (α)', '${metrics.exposureAlpha.round()}x', GCamColors.accentAmber),
              _buildStat('SNR GAIN', '+${metrics.psnrGainDb.toStringAsFixed(2)} dB', GCamColors.accentBlue),
            ],
          ),

          const SizedBox(height: 8),
          // ISP Pipeline Stage Tracker
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
            decoration: BoxDecoration(
              color: Colors.white.withValues(alpha: 0.08),
              borderRadius: BorderRadius.circular(8),
            ),
            child: Row(
              children: [
                const Icon(Icons.tune_rounded, color: GCamColors.accentAmber, size: 14),
                const SizedBox(width: 6),
                Expanded(
                  child: Text(
                    'GCam ISP: ${metrics.ispStage}',
                    style: const TextStyle(
                      color: GCamColors.textSecondary,
                      fontSize: 10,
                      fontWeight: FontWeight.w500,
                    ),
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildStat(String label, String value, Color color) {
    return Column(
      children: [
        Text(
          label,
          style: const TextStyle(
            color: GCamColors.textTertiary,
            fontSize: 9,
            fontWeight: FontWeight.w700,
            letterSpacing: 0.5,
          ),
        ),
        const SizedBox(height: 2),
        Text(
          value,
          style: TextStyle(
            color: color,
            fontSize: 13,
            fontWeight: FontWeight.w800,
          ),
        ),
      ],
    );
  }
}
