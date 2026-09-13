package dev.kursor.ktensorflow.vision

import kotlinx.cinterop.Pinned
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.pin
import platform.CoreGraphics.CGColorRenderingIntent
import platform.CoreGraphics.CGColorSpaceCreateDeviceGray
import platform.CoreGraphics.CGColorSpaceCreateDeviceRGB
import platform.CoreGraphics.CGDataProviderCreateWithData
import platform.CoreGraphics.CGImageCreate
import platform.CoreGraphics.CGImageRef
import platform.CoreGraphics.CGImageRelease

class CGImageScope(
    val cgImage: CGImageRef,
    private val pinnedData: Pinned<ByteArray>
) : AutoCloseable {

    override fun close() {
        CGImageRelease(cgImage)
        pinnedData.unpin()
    }
}

fun Image.asCGImage(): CGImageScope {
    val pinnedData = platformImage.pin()
    return CGImageScope(createCGImage(pinnedData), pinnedData)
}

fun <T> Image.withCGImage(block: (CGImageRef) -> T): T {
    val pinnedData = platformImage.pin()
    val cgImage = createCGImage(pinnedData)
    try {
        return block(cgImage)
    } finally {
        CGImageRelease(cgImage)
        pinnedData.unpin()
    }
}

// Данные должны оставаться запиненными (pinned) до тех пор, пока CGImage реально
// не прочитан CoreGraphics (например, внутри CGContextDrawImage) - если распиновать
// сразу после CGImageCreate, GC Kotlin/Native может переместить память ДО чтения,
// и CGImage окажется битым/пустым (было воспроизведено на неоднородных изображениях).
private fun Image.createCGImage(pinnedData: Pinned<ByteArray>): CGImageRef {
    val colorSpace =
        if (pixelFormat == PixelFormat.Grayscale)
            CGColorSpaceCreateDeviceGray()
        else
            CGColorSpaceCreateDeviceRGB()

    val provider = CGDataProviderCreateWithData(
        null,
        pinnedData.addressOf(0),
        platformImage.size.toULong(),
        null
    )!!

    return CGImageCreate(
        width.toULong(),
        height.toULong(),
        8u,
        (pixelFormat.channels * 8).toULong(),
        (width * pixelFormat.channels).toULong(),
        colorSpace,
        pixelFormat.cgBitmapInfo,
        provider,
        null,
        false,
        CGColorRenderingIntent.kCGRenderingIntentDefault
    )!!
}
