package dev.kursor.ktensorflow.vision

import kotlinx.cinterop.addressOf
import kotlinx.cinterop.alloc
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.plus
import kotlinx.cinterop.ptr
import kotlinx.cinterop.reinterpret
import kotlinx.cinterop.usePinned
import kotlinx.cinterop.value
import platform.CoreGraphics.CGContextDrawImage
import platform.CoreGraphics.CGImageGetHeight
import platform.CoreGraphics.CGImageGetWidth
import platform.CoreGraphics.CGImageRef
import platform.CoreGraphics.CGImageRefVar
import platform.CoreGraphics.CGImageRelease
import platform.CoreGraphics.CGRectMake
import platform.CoreVideo.CVPixelBufferGetBaseAddress
import platform.CoreVideo.CVPixelBufferGetBytesPerRow
import platform.CoreVideo.CVPixelBufferGetHeight
import platform.CoreVideo.CVPixelBufferGetPixelFormatType
import platform.CoreVideo.CVPixelBufferGetWidth
import platform.CoreVideo.CVPixelBufferIsPlanar
import platform.CoreVideo.CVPixelBufferLockBaseAddress
import platform.CoreVideo.CVPixelBufferRef
import platform.CoreVideo.CVPixelBufferUnlockBaseAddress
import platform.CoreVideo.kCVPixelBufferLock_ReadOnly
import platform.CoreVideo.kCVPixelFormatType_32BGRA
import platform.CoreVideo.kCVReturnSuccess
import platform.VideoToolbox.VTCreateCGImageFromCVPixelBuffer
import platform.darwin.ByteVar
import platform.darwin.noErr
import platform.posix.memcpy

/**
 * Creates an [Image] from a CoreGraphics image.
 *
 * The pixels are copied, so the returned image does not depend on [cgImage]: the caller keeps
 * ownership of it and releases it as usual, before or after using the result. Any CoreGraphics
 * image is accepted, whatever its color space, bit depth or alpha layout: it is drawn into a
 * device RGB (or device gray) buffer in [pixelFormat].
 *
 * [PixelFormat.BGRA] is the native layout of CoreGraphics and of the camera and is the cheapest.
 * [PixelFormat.Grayscale] converts the image with the same ITU-R 601 weights as [grayscale]; a
 * gray image keeps every level unchanged.
 *
 * `UIImage.imageOrientation` is not part of a `CGImage`: pass `uiImage.CGImage` and [rotate] the
 * result if the image is not upright.
 *
 * @param cgImage the image to copy.
 * @param pixelFormat storage format of the new image.
 */
fun Image(
    cgImage: CGImageRef,
    pixelFormat: PixelFormat = PixelFormat.BGRA
): Image {
    val width = CGImageGetWidth(cgImage).toInt()
    val height = CGImageGetHeight(cgImage).toInt()

    return when (pixelFormat) {
        is PixelFormat.RGBA, is PixelFormat.RGB -> draw(cgImage, width, height, pixelFormat)

        // Через grayscale (vImage, Rec.601), а не рисованием в серый контекст: CoreGraphics
        // при этом применяет управление цветом, и яркость расходилась бы с Android. Серое
        // изображение проходит этот путь без потерь - каждый из 256 уровней остаётся собой.
        PixelFormat.Grayscale ->
            draw(cgImage, width, height, PixelFormat.BGRA).grayscale(closeOriginal = true)
    }
}

/**
 * Creates an [Image] from a CoreVideo pixel buffer, for example a camera frame from
 * `AVCaptureVideoDataOutput` or `CMSampleBufferGetImageBuffer`.
 *
 * The pixels are copied, so the returned image does not depend on [pixelBuffer]: the caller
 * keeps ownership of it, and the frame can be returned to the capture pool right away. The
 * buffer does not need to be locked.
 *
 * A `kCVPixelFormatType_32BGRA` buffer converted to [PixelFormat.BGRA] is copied row by row
 * without any conversion; its color is treated as premultiplied by alpha, as CoreVideo does, which
 * makes no difference for opaque camera frames. Any other buffer format that VideoToolbox can
 * convert to a `CGImage`, including the camera's YpCbCr formats, goes through the [Image]
 * `CGImage` factory above, with the same rules for [pixelFormat].
 *
 * Camera frames come in the sensor orientation: [rotate] the result to make it upright.
 *
 * @param pixelBuffer the frame to copy.
 * @param pixelFormat storage format of the new image.
 * @throws IllegalArgumentException if the buffer format cannot be converted.
 */
fun Image(
    pixelBuffer: CVPixelBufferRef,
    pixelFormat: PixelFormat = PixelFormat.BGRA
): Image {
    if (pixelFormat == PixelFormat.BGRA &&
        CVPixelBufferGetPixelFormatType(pixelBuffer) == kCVPixelFormatType_32BGRA &&
        !CVPixelBufferIsPlanar(pixelBuffer)
    ) {
        return copyBgra(pixelBuffer)
    }

    val cgImage = memScoped {
        val out = alloc<CGImageRefVar>()
        val status = VTCreateCGImageFromCVPixelBuffer(pixelBuffer, null, out.ptr)
        require(status == noErr.toInt() && out.value != null) {
            "CVPixelBuffer of format ${CVPixelBufferGetPixelFormatType(pixelBuffer).fourCC()} " +
                "cannot be converted to an image (OSStatus $status)"
        }
        out.value!!
    }
    try {
        return Image(cgImage, pixelFormat)
    } finally {
        CGImageRelease(cgImage)
    }
}

private fun draw(
    cgImage: CGImageRef,
    width: Int,
    height: Int,
    pixelFormat: PixelFormat
): Image {
    val out = ByteArray(width * height * pixelFormat.channels)
    withBitmapContext(out, width, height, pixelFormat) { ctx ->
        CGContextDrawImage(
            c = ctx,
            rect = CGRectMake(
                x = 0.0,
                y = 0.0,
                width = width.toDouble(),
                height = height.toDouble()
            ),
            image = cgImage
        )
    }
    return IosImage(width, height, pixelFormat, out)
}

private fun copyBgra(pixelBuffer: CVPixelBufferRef): Image {
    val status = CVPixelBufferLockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly)
    require(status == kCVReturnSuccess) { "Failed to lock CVPixelBuffer (CVReturn $status)" }
    try {
        val width = CVPixelBufferGetWidth(pixelBuffer).toInt()
        val height = CVPixelBufferGetHeight(pixelBuffer).toInt()
        val srcRowBytes = CVPixelBufferGetBytesPerRow(pixelBuffer).toLong()
        val src = checkNotNull(CVPixelBufferGetBaseAddress(pixelBuffer)) {
            "CVPixelBuffer has no base address"
        }.reinterpret<ByteVar>()

        // Строки CVPixelBuffer обычно выровнены (bytesPerRow > width * 4), а буфер IosImage
        // плотный: копировать приходится построчно
        val rowBytes = width * PixelFormat.BGRA.channels
        val out = ByteArray(height * rowBytes)
        out.usePinned { pinned ->
            for (y in 0 until height) {
                memcpy(pinned.addressOf(y * rowBytes), src + y * srcRowBytes, rowBytes.toULong())
            }
        }
        return IosImage(width, height, PixelFormat.BGRA, out)
    } finally {
        CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly)
    }
}

private fun UInt.fourCC(): String =
    listOf(24, 16, 8, 0).map { ((this shr it) and 0xFFu).toInt().toChar() }.joinToString("")
