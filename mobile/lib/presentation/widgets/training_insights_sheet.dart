import 'package:flutter/material.dart';
import '../../core/constants/gcam_colors.dart';
import '../../core/models/benchmark_metrics.dart';

/// Modal bottom sheet showcasing Kaggle Cloud Training Data, Attention Gate XAI, and Architecture Comparison.
class TrainingInsightsSheet extends StatelessWidget {
  const TrainingInsightsSheet({super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      height: MediaQuery.of(context).size.height * 0.85,
      decoration: const BoxDecoration(
        color: GCamColors.surface,
        borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
      ),
      child: Column(
        children: [
          // Drag Handle
          const SizedBox(height: 12),
          Container(
            width: 44,
            height: 4,
            decoration: BoxDecoration(
              color: Colors.white24,
              borderRadius: BorderRadius.circular(2),
            ),
          ),
          const SizedBox(height: 16),

          // Header
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 20),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'LumaCam Academic Insights',
                      style: TextStyle(
                        color: GCamColors.textPrimary,
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    SizedBox(height: 2),
                    Text(
                      'Capstone Project Team 21 • SCTCE',
                      style: TextStyle(
                        color: GCamColors.textSecondary,
                        fontSize: 12,
                      ),
                    ),
                  ],
                ),
                IconButton(
                  icon: const Icon(Icons.close_rounded, color: Colors.white70),
                  onPressed: () => Navigator.pop(context),
                ),
              ],
            ),
          ),
          const Divider(color: Colors.white12, height: 24),

          // Content List
          Expanded(
            child: ListView(
              padding: const EdgeInsets.symmetric(horizontal: 20),
              children: [
                // 1. Kaggle Training Specifications Card
                _buildSectionHeader('1. KAGGLE CLOUD TRAINING METRICS', Icons.cloud_done_rounded),
                Container(
                  padding: const EdgeInsets.all(14),
                  decoration: BoxDecoration(
                    color: GCamColors.surfaceElevated,
                    borderRadius: BorderRadius.circular(14),
                    border: Border.all(color: Colors.white12),
                  ),
                  child: Column(
                    children: [
                      _buildMetricRow('Dataset', BenchmarkMetrics.trainingSummary['dataset']),
                      _buildMetricRow('Training Pairs', '${BenchmarkMetrics.trainingSummary['trainingPairs']} raw Sony pairs'),
                      _buildMetricRow('Total Epochs', '${BenchmarkMetrics.trainingSummary['epochs']} (Alternating GAN Loop)'),
                      _buildMetricRow('Composite Loss', BenchmarkMetrics.trainingSummary['lossObjective']),
                      _buildMetricRow('Target PSNR', BenchmarkMetrics.trainingSummary['finalValidationPsnr']),
                      _buildMetricRow('Target SSIM', BenchmarkMetrics.trainingSummary['finalValidationSsim']),
                      _buildMetricRow('LPIPS Perceptual', BenchmarkMetrics.trainingSummary['finalLpips']),
                      _buildMetricRow('Training Hardware', BenchmarkMetrics.trainingSummary['trainingHardware']),
                    ],
                  ),
                ),
                const SizedBox(height: 20),

                // 2. Explainable AI: Attention Gates
                _buildSectionHeader('2. EXPLAINABLE AI (XAI): ATTENTION GATES', Icons.visibility_rounded),
                const Text(
                  'Spatial Attention Gates dynamically filter extreme Poisson shot noise on U-Net skip connections while preserving high-frequency edges and contours:',
                  style: TextStyle(color: GCamColors.textSecondary, fontSize: 12),
                ),
                const SizedBox(height: 10),
                ClipRRect(
                  borderRadius: BorderRadius.circular(12),
                  child: Image.asset(
                    'assets/figures/attention_gate_heatmaps.png',
                    fit: BoxFit.cover,
                    errorBuilder: (context, error, stackTrace) => const SizedBox.shrink(),
                  ),
                ),
                const SizedBox(height: 20),

                // 3. Latency & Real-Time Edge Feasibility
                _buildSectionHeader('3. LATENCY COMPARISON VS DIFFUSION', Icons.speed_rounded),
                const Text(
                  'While diffusion models take 2-5 seconds per image, LumaCam operates in a single forward pass (~14.2 ms / 70 FPS), well within the 33.3 ms (30 FPS) mobile camera threshold:',
                  style: TextStyle(color: GCamColors.textSecondary, fontSize: 12),
                ),
                const SizedBox(height: 10),
                ClipRRect(
                  borderRadius: BorderRadius.circular(12),
                  child: Image.asset(
                    'assets/figures/latency_throughput_benchmark.png',
                    fit: BoxFit.cover,
                    errorBuilder: (context, error, stackTrace) => const SizedBox.shrink(),
                  ),
                ),
                const SizedBox(height: 20),

                // 4. Quality Benchmark Comparison
                _buildSectionHeader('4. SIGNAL FIDELITY (PSNR & SSIM)', Icons.bar_chart_rounded),
                ClipRRect(
                  borderRadius: BorderRadius.circular(12),
                  child: Image.asset(
                    'assets/figures/quality_metrics_comparison.png',
                    fit: BoxFit.cover,
                    errorBuilder: (context, error, stackTrace) => const SizedBox.shrink(),
                  ),
                ),
                const SizedBox(height: 30),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSectionHeader(String title, IconData icon) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Row(
        children: [
          Icon(icon, color: GCamColors.accentAmber, size: 16),
          const SizedBox(width: 8),
          Text(
            title,
            style: const TextStyle(
              color: GCamColors.accentAmber,
              fontSize: 12,
              fontWeight: FontWeight.bold,
              letterSpacing: 0.8,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildMetricRow(String key, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(
            key,
            style: const TextStyle(color: GCamColors.textSecondary, fontSize: 11),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              value,
              textAlign: TextAlign.end,
              style: const TextStyle(
                color: GCamColors.textPrimary,
                fontSize: 11,
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
