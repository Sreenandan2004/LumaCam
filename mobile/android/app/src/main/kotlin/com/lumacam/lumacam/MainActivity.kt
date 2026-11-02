package com.lumacam.lumacam

import android.Manifest
import android.content.pm.PackageManager
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel

class MainActivity : FlutterActivity() {
    companion object {
        private const val CHANNEL = "com.lumacam/camera2"
        private const val CAMERA_PERMISSION_REQUEST_CODE = 1001
    }

    private var camera2RawController: Camera2RawController? = null
    private var pendingPermissionResult: MethodChannel.Result? = null

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        camera2RawController = Camera2RawController(
            activity = this,
            context = applicationContext,
            textureRegistry = flutterEngine.renderer
        )

        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL).setMethodCallHandler { call, result ->
            when (call.method) {
                "initializeCamera" -> {
                    checkAndRequestCameraPermission(result)
                }
                "setExposureAlpha" -> {
                    val alpha = call.argument<Double>("alpha") ?: 100.0
                    camera2RawController?.setExposureAlpha(alpha, result)
                }
                "captureImage" -> {
                    val alpha = call.argument<Double>("alpha") ?: 100.0
                    val quarterTurns = call.argument<Int>("quarterTurns") ?: 0
                    camera2RawController?.captureImage(alpha, quarterTurns, result)
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

    private fun checkAndRequestCameraPermission(result: MethodChannel.Result) {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            camera2RawController?.initialize(result)
        } else {
            pendingPermissionResult = result
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.CAMERA),
                CAMERA_PERMISSION_REQUEST_CODE
            )
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == CAMERA_PERMISSION_REQUEST_CODE) {
            val pending = pendingPermissionResult
            pendingPermissionResult = null
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                if (pending != null) {
                    camera2RawController?.initialize(pending)
                }
            } else {
                pending?.error("PERMISSION_DENIED", "Camera permission is required to stream sensor feed", null)
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        camera2RawController?.dispose(object : MethodChannel.Result {
            override fun success(result: Any?) {}
            override fun error(errorCode: String, errorMessage: String?, errorDetails: Any?) {}
            override fun notImplemented() {}
        })
    }
}
