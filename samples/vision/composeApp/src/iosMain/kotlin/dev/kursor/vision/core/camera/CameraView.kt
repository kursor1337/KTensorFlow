package dev.kursor.vision.core.camera

import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.readValue
import platform.AVFoundation.AVCaptureSession
import platform.AVFoundation.AVCaptureVideoDataOutput
import platform.AVFoundation.AVCaptureVideoPreviewLayer
import platform.CoreGraphics.CGRectZero
import platform.UIKit.UIView
import platform.darwin.NSObject

@OptIn(ExperimentalForeignApi::class)
class CameraView : UIView {
    lateinit var session: AVCaptureSession
    lateinit var output: AVCaptureVideoDataOutput
    lateinit var delegate: NSObject

    var previewLayer: AVCaptureVideoPreviewLayer? = null

    constructor() : super(frame = CGRectZero.readValue())

    override fun layoutSubviews() {
        super.layoutSubviews()
        // Теперь frame previewLayer будет равен bounds, когда UIKitView получает размер от Compose
        previewLayer?.frame = bounds
    }
}
