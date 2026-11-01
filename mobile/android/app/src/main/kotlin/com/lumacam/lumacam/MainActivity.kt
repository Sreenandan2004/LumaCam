package com.lumacam.lumacam

import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel

class MainActivity : FlutterActivity() {
    private val CHANNEL = "com.lumacam/camera2"
    private var camera2RawController: Camera2RawController? = null

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        camera2RawController = Camera2RawController(
            context = applicationContext,
            textureRegistry = flutterEngine.renderer
        )

        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL).setMethodCallHandler { call, result ->
            when (call.method) {
                "initializeCamera" -> {
                    camera2RawController?.initialize(result)
                }
                "setExposureAlpha" -> {
                    val alpha = call.argument<Double>("alpha") ?: 100.0
                    camera2RawController?.setExposureAlpha(alpha, result)
                }
                "captureImage" -> {
                    val alpha = call.argument<Double>("alpha") ?: 100.0
                    camera2RawController?.captureImage(alpha, result)
                }
                "dispose" -> {
                    camera2RawController?.dispose(result)
                }
                else -> {
                    result.notImplemented()
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        // Ensure background threads and camera resources are released
        camera2RawController?.dispose(object : MethodChannel.Result {
            override fun success(result: Any?) {}
            override fun error(errorCode: String, errorMessage: String?, errorDetails: Any?) {}
            override fun notImplemented() {}
        })
    }
}
