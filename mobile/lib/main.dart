import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'core/constants/gcam_colors.dart';
import 'presentation/screens/gcam_camera_screen.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();

  // Enforce immersive edge-to-edge dark system overlay
  SystemChrome.setSystemUIOverlayStyle(
    const SystemUiOverlayStyle(
      statusBarColor: Colors.transparent,
      statusBarIconBrightness: Brightness.light,
      systemNavigationBarColor: GCamColors.background,
      systemNavigationBarIconBrightness: Brightness.light,
    ),
  );

  runApp(const LumaCamApp());
}

class LumaCamApp extends StatelessWidget {
  const LumaCamApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'LumaCam',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        brightness: Brightness.dark,
        scaffoldBackgroundColor: GCamColors.background,
        colorScheme: const ColorScheme.dark(
          primary: GCamColors.accentAmber,
          secondary: GCamColors.accentBlue,
          surface: GCamColors.surface,
        ),
        useMaterial3: true,
      ),
      home: const GCamCameraScreen(),
    );
  }
}
