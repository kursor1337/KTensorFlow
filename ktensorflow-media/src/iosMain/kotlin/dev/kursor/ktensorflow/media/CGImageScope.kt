package dev.kursor.ktensorflow.media

import kotlinx.cinterop.addressOf
import kotlinx.cinterop.usePinned
import platform.CoreGraphics.CGColorRenderingIntent
import platform.CoreGraphics.CGColorSpaceCreateDeviceGray
import platform.CoreGraphics.CGColorSpaceCreateDeviceRGB
import platform.CoreGraphics.CGDataProviderCreateWithData
import platform.CoreGraphics.CGImageAlphaInfo
import platform.CoreGraphics.CGImageCreate
import platform.CoreGraphics.CGImageRef
import platform.CoreGraphics.CGImageRelease
import platform.CoreGraphics.kCGBitmapByteOrder32Little

class CGImageScope(
    val cgImage: CGImageRef
) : AutoCloseable {

    override fun close() {
        CGImageRelease(cgImage)
    }
}

fun Image.asCGImage(): CGImageScope =
    CGImageScope(toCGImage())

fun <T> Image.withCGImage(block: (CGImageRef) -> T) {
    val cgImage = toCGImage()
    try {
        block(cgImage)
    } finally {
        CGImageRelease(cgImage)
    }
}

private fun Image.toCGImage(): CGImageRef {
    val colorSpace =
        if (pixelFormat == PixelFormat.Grayscale)
            CGColorSpaceCreateDeviceGray()
        else
            CGColorSpaceCreateDeviceRGB()

    val provider = platformImage.usePinned {
        CGDataProviderCreateWithData(
            null,
            it.addressOf(0),
            platformImage.size.toULong(),
            null
        )
    }!!

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


