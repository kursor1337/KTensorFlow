package dev.kursor.vision.core.camera

import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.viewinterop.UIKitView
import dev.kursor.ktensorflow.vision.Image
import dev.kursor.ktensorflow.vision.IosImage
import dev.kursor.ktensorflow.vision.PixelFormat
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.refTo
import platform.AVFoundation.AVCaptureConnection
import platform.AVFoundation.AVCaptureDevice
import platform.AVFoundation.AVCaptureDeviceInput
import platform.AVFoundation.AVCaptureOutput
import platform.AVFoundation.AVCaptureSession
import platform.AVFoundation.AVCaptureSessionPreset640x480
import platform.AVFoundation.AVCaptureVideoDataOutput
import platform.AVFoundation.AVCaptureVideoDataOutputSampleBufferDelegateProtocol
import platform.AVFoundation.AVCaptureVideoOrientationPortrait
import platform.AVFoundation.AVCaptureVideoPreviewLayer
import platform.AVFoundation.AVLayerVideoGravityResizeAspectFill
import platform.AVFoundation.AVMediaTypeVideo
import platform.CoreGraphics.CGBitmapContextCreate
import platform.CoreGraphics.CGColorSpaceCreateDeviceRGB
import platform.CoreGraphics.CGContextDrawImage
import platform.CoreGraphics.CGImageAlphaInfo
import platform.CoreGraphics.CGImageGetHeight
import platform.CoreGraphics.CGImageGetWidth
import platform.CoreGraphics.CGImageRelease
import platform.CoreGraphics.CGRectGetHeight
import platform.CoreGraphics.CGRectGetWidth
import platform.CoreGraphics.CGRectMake
import platform.CoreGraphics.kCGBitmapByteOrder32Little
import platform.CoreImage.CIContext
import platform.CoreImage.CIImage
import platform.CoreImage.createCGImage
import platform.CoreMedia.CMSampleBufferGetImageBuffer
import platform.CoreMedia.CMSampleBufferRef
import platform.CoreVideo.CVPixelBufferLockBaseAddress
import platform.CoreVideo.CVPixelBufferUnlockBaseAddress
import platform.CoreVideo.kCVPixelBufferPixelFormatTypeKey
import platform.CoreVideo.kCVPixelFormatType_32BGRA
import platform.darwin.NSObject
import platform.darwin.dispatch_async
import platform.darwin.dispatch_queue_create

@OptIn(ExperimentalForeignApi::class)
@Composable
actual fun LiveCameraUi(
    modifier: Modifier,
    onFrame: (Image) -> Unit
) {
    UIKitView(
        modifier = modifier,
        factory = {
            val view = CameraView()
            val session = AVCaptureSession().apply {
                sessionPreset = AVCaptureSessionPreset640x480
            }

            val device = AVCaptureDevice.defaultDeviceWithMediaType(AVMediaTypeVideo)
            val input = AVCaptureDeviceInput.deviceInputWithDevice(device!!, error = null)
            if (input != null && session.canAddInput(input)) session.addInput(input)

            val output = AVCaptureVideoDataOutput().apply {
                setAlwaysDiscardsLateVideoFrames(true)
                videoSettings = mapOf(kCVPixelBufferPixelFormatTypeKey to kCVPixelFormatType_32BGRA)
                val connection = connectionWithMediaType(AVMediaTypeVideo)
                connection?.videoOrientation = AVCaptureVideoOrientationPortrait

            }

            val queue = dispatch_queue_create("camera.frame.queue", null)
            val delegate =
                object : NSObject(), AVCaptureVideoDataOutputSampleBufferDelegateProtocol {
                    override fun captureOutput(
                        output: AVCaptureOutput,
                        didOutputSampleBuffer: CMSampleBufferRef?,
                        fromConnection: AVCaptureConnection
                    ) {
                        val pixelBuffer = CMSampleBufferGetImageBuffer(didOutputSampleBuffer)!!
                        CVPixelBufferLockBaseAddress(pixelBuffer, 0U)

                        val ciImage = CIImage.imageWithCVPixelBuffer(pixelBuffer).imageByApplyingOrientation(6)
                        val context = CIContext()
                        val cgImage = context.createCGImage(ciImage, ciImage.extent)!!

                        val width = CGImageGetWidth(cgImage)
                        val height = CGImageGetHeight(cgImage)
                        val bytesPerPixel = 4u
                        val out = ByteArray((width * height * bytesPerPixel).toInt())

                        val colorSpace = CGColorSpaceCreateDeviceRGB()
                        val bitmapContext = CGBitmapContextCreate(
                            out.refTo(0),
                            width,
                            height,
                            8u,                              // bits per component
                            width * bytesPerPixel,          // bytes per row (tight)
                            colorSpace,
                            CGImageAlphaInfo.kCGImageAlphaPremultipliedFirst.value or kCGBitmapByteOrder32Little
                        )!!

                        CGContextDrawImage(
                            bitmapContext,
                            CGRectMake(
                                0.0,
                                0.0,
                                width.toDouble(),
                                height.toDouble()
                            ),
                            cgImage
                        )

                        CGImageRelease(cgImage)
                        CVPixelBufferUnlockBaseAddress(pixelBuffer, 0U)

                        // ✅ Call onFrame with the proper image
                        onFrame(
                            IosImage(
                                width = width.toInt(),
                                height = height.toInt(),
                                pixelFormat = PixelFormat.BGRA, // matches kCGImageAlphaPremultipliedLast
                                platformImage = out
                            )
                        )
                    }
                }

            output.setSampleBufferDelegate(delegate, queue)
            if (session.canAddOutput(output)) session.addOutput(output)

            val previewLayer = AVCaptureVideoPreviewLayer(session = session).apply {
                videoGravity = AVLayerVideoGravityResizeAspectFill
                frame = view.bounds

                val width = CGRectGetWidth(view.bounds)
                val height = CGRectGetHeight(view.bounds)
                println("view.bounds: ${width}x$height")
            }

            view.layer.addSublayer(previewLayer)

            val sessionQueue = dispatch_queue_create("camera.session.queue", null)
            dispatch_async(sessionQueue) {
                session.startRunning()
            }

            view.session = session
            view.output = output
            view.delegate = delegate
            view.previewLayer = previewLayer

            // ✅ Cleanup on dispose
            view.layer.setNeedsLayout()
            view.layoutIfNeeded()

            view
        },
        onRelease = { view ->
            val layers = view.layer.sublayers?.filterIsInstance<AVCaptureVideoPreviewLayer>()
                ?: return@UIKitView
            layers.forEach { layer ->
                layer.session?.stopRunning()
                layer.removeFromSuperlayer()
            }
        }
    )
}
