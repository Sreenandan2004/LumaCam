package com.lumacam.lumacam

import android.Manifest
import android.content.ContentValues
import android.content.pm.PackageManager
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel

class MainActivity : FlutterActivity() {
    companion object {
        private const val TAG = "MainActivity"
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
                    val mode = call.argument<String>("mode") ?: "NIGHT SIGHT"
                    camera2RawController?.captureImage(alpha, quarterTurns, mode, result)
                }
                "saveImageToGallery" -> {
                    val bytes = call.argument<ByteArray>("imageBytes")
                    val title = call.argument<String>("title") ?: "LumaCam_${System.currentTimeMillis()}"
                    if (bytes != null) {
                        val savedPath = saveImageToMediaStore(bytes, title)
                        if (savedPath != null) {
                            result.success(savedPath)
                        } else {
                            result.error("SAVE_FAILED", "Failed to write image to storage", null)
                        }
                    } else {
                        result.error("INVALID_ARGS", "Image bytes are null", null)
                    }
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

    private fun saveImageToMediaStore(bytes: ByteArray, title: String): String? {
        return try {
            val contentValues = ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, "$title.jpg")
                put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/LumaCam")
                    put(MediaStore.Images.Media.IS_PENDING, 1)
                }
            }

            val uri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues)
            if (uri != null) {
                contentResolver.openOutputStream(uri)?.use { outputStream ->
                    outputStream.write(bytes)
                    outputStream.flush()
                }
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    contentValues.clear()
                    contentValues.put(MediaStore.Images.Media.IS_PENDING, 0)
                    contentResolver.update(uri, contentValues, null, null)
                }
                Log.i(TAG, "Successfully saved photo to MediaStore: $uri")
                uri.toString()
            } else {
                null
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save photo to MediaStore", e)
            null
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
