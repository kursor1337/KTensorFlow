package dev.kursor.vision.core.camera

import android.util.Log
import androidx.camera.core.AspectRatio
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import dev.kursor.ktensorflow.vision.AndroidImage
import dev.kursor.ktensorflow.vision.Image
import dev.kursor.ktensorflow.vision.PixelFormat
import dev.kursor.ktensorflow.vision.rotate

@Composable
actual fun LiveCameraUi(
    modifier: Modifier,
    onFrame: (Image) -> Unit
) {
    val lifecycleOwner = LocalLifecycleOwner.current

    AndroidView(
        modifier = modifier,
        factory = { ctx ->
            val previewView = PreviewView(ctx)

            val cameraProviderFuture = ProcessCameraProvider.getInstance(ctx)
            cameraProviderFuture.addListener({
                val cameraProvider = cameraProviderFuture.get()

                val preview = Preview.Builder()
                    .setTargetAspectRatio(AspectRatio.RATIO_16_9)
                    .setTargetRotation(previewView.display.rotation)
                    .build()
                    .apply {
                        surfaceProvider = previewView.surfaceProvider
                    }

                Log.d("kursor1337", "previewView.display.rotation: ${previewView.display.rotation}")

                val analyzer = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()

                analyzer.setAnalyzer(ContextCompat.getMainExecutor(ctx)) { imageProxy ->
                    try {
                        val rotation = imageProxy.imageInfo.rotationDegrees
                        val image = imageProxy
                            .toBitmap()
                            .let {
                                AndroidImage(
                                    it,
                                    PixelFormat.ARGB
                                )
                            }
                            .let {
                                if (rotation != 0) {
                                    it.rotate(rotation.toFloat())
                                } else {
                                    it
                                }
                            }
                        onFrame(image)
                    } catch (e: Exception) {
                        Log.e("LiveCameraUi", "Frame error: ${e.message}")
                    } finally {
                        imageProxy.close()
                    }
                }

                try {
                    cameraProvider.unbindAll()
                    cameraProvider.bindToLifecycle(
                        lifecycleOwner,
                        CameraSelector.DEFAULT_BACK_CAMERA,
                        preview,
                        analyzer
                    )
                } catch (e: Exception) {
                    Log.e("LiveCameraUi", "Camera binding failed: ${e.message}")
                }
            }, ContextCompat.getMainExecutor(ctx))
            previewView
        },
        update = {

        }
    )
}
