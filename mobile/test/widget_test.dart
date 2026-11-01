import 'package:flutter_test/flutter_test.dart';
import 'package:lumacam/main.dart';
import 'package:lumacam/presentation/widgets/gcam_shutter_button.dart';
import 'package:lumacam/presentation/widgets/gcam_exposure_slider.dart';
import 'package:lumacam/presentation/widgets/metrics_hud_overlay.dart';

void main() {
  testWidgets('LumaCam GCam UI Smoke Test and Interactive Controls', (WidgetTester tester) async {
    // Build LumaCam application
    await tester.pumpWidget(const LumaCamApp());
    await tester.pump();

    // 1. Verify Top Bar Branding
    expect(find.text('LUMACAM AI'), findsOneWidget);

    // 2. Verify Camera Modes in Carousel
    expect(find.text('ADAPTIVE LOW LIGHT'), findsOneWidget);
    expect(find.text('NIGHT SIGHT'), findsOneWidget);
    expect(find.text('PRO RAW'), findsOneWidget);

    // 3. Verify GCam Shutter Button and Exposure Slider exist
    expect(find.byType(GCamShutterButton), findsOneWidget);
    expect(find.byType(GCamExposureSlider), findsOneWidget);

    // 4. Verify Metrics HUD Overlay exists initially
    expect(find.byType(MetricsHudOverlay), findsOneWidget);
    expect(find.textContaining('LATENCY'), findsOneWidget);

    // 5. Test Tapping Mode in Carousel
    await tester.tap(find.text('NIGHT SIGHT'));
    await tester.pump();

    // 6. Test Tapping HUD Toggle Button
    final hudButton = find.byTooltip('Toggle Performance Matrix HUD');
    expect(hudButton, findsOneWidget);
    await tester.tap(hudButton);
    await tester.pump();

    // Verify HUD hidden
    expect(find.byType(MetricsHudOverlay), findsNothing);

    // Toggle HUD back on
    await tester.tap(hudButton);
    await tester.pump();
    expect(find.byType(MetricsHudOverlay), findsOneWidget);
  });
}
