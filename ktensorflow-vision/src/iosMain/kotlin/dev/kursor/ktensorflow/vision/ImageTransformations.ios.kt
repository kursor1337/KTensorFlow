package dev.kursor.ktensorflow.vision

import kotlinx.cinterop.addressOf
import kotlinx.cinterop.usePinned
import platform.CoreGraphics.CGBitmapContextCreate
import platform.CoreGraphics.CGColorSpaceCreateDeviceGray
import platform.CoreGraphics.CGColorSpaceCreateDeviceRGB
import platform.CoreGraphics.CGColorSpaceRelease
import platform.CoreGraphics.CGContextDrawImage
import platform.CoreGraphics.CGContextRef
import platform.CoreGraphics.CGContextRelease
import platform.CoreGraphics.CGContextRotateCTM
import platform.CoreGraphics.CGContextTranslateCTM
import platform.CoreGraphics.CGImageCreateWithImageInRect
import platform.CoreGraphics.CGImageRelease
import platform.CoreGraphics.CGRectMake
import kotlin.math.PI

actual fun Image.resize(
    newWidth: Int,
    newHeight: Int,
    closeOriginal: Boolean
): Image {
    val out = ByteArray(newWidth * newHeight * pixelFormat.channels)
    withBitmapContext(out, newWidth, newHeight, pixelFormat) { ctx ->
        withCGImage { cgImage ->
            CGContextDrawImage(
                c = ctx,
                rect = CGRectMake(
                    x = 0.0,
                    y = 0.0,
                    width = newWidth.toDouble(),
                    height = newHeight.toDouble()
                ),
                image = cgImage
            )
        }
    }

    if (closeOriginal) close()

    return IosImage(
        width = newWidth,
        height = newHeight,
        pixelFormat = pixelFormat,
        platformImage = out
    )
}

actual fun Image.crop(
    rect: Rect,
    closeOriginal: Boolean
): Image {
    val newWidth = rect.right - rect.left
    val newHeight = rect.bottom - rect.top
    val out = ByteArray(newWidth * newHeight * pixelFormat.channels)

    withBitmapContext(out, newWidth, newHeight, pixelFormat) { ctx ->
        withCGImage { cgImage ->
            val cropRect = CGRectMake(
                x = rect.left.toDouble(),
                y = rect.top.toDouble(),
                width = newWidth.toDouble(),
                height = newHeight.toDouble()
            )
            val cropped = CGImageCreateWithImageInRect(
                image = cgImage,
                rect = cropRect
            ) ?: error("Failed to crop CGImage")
            CGContextDrawImage(
                c = ctx,
                rect = CGRectMake(
                    x = 0.0,
                    y = 0.0,
                    width = newWidth.toDouble(),
                    height = newHeight.toDouble()
                ),
                image = cropped)
            CGImageRelease(cropped)
        }
    }

    if (closeOriginal) close()

    return IosImage(
        width = newWidth,
        height = newHeight,
        pixelFormat = pixelFormat,
        platformImage = out
    )
}

actual fun Image.rotate(
    degrees: Float,
    closeOriginal: Boolean
): Image {
    val radians = degrees * PI / 180.0
    val sin = kotlin.math.abs(kotlin.math.sin(radians))
    val cos = kotlin.math.abs(kotlin.math.cos(radians))
    val newWidth = (width * cos + height * sin).toInt()
    val newHeight = (width * sin + height * cos).toInt()

    val out = ByteArray(newWidth * newHeight * pixelFormat.channels)
    withBitmapContext(out, newWidth, newHeight, pixelFormat) { ctx ->
        CGContextTranslateCTM(
            c = ctx,
            tx = newWidth / 2.0,
            ty = newHeight / 2.0
        )
        CGContextRotateCTM(
            c = ctx,
            angle = radians
        )
        CGContextTranslateCTM(
            c = ctx,
            tx = -width / 2.0,
            ty = -height / 2.0
        )
        withCGImage { cgImage ->
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
    }

    if (closeOriginal) close()

    return IosImage(
        width = newWidth,
        height = newHeight,
        pixelFormat = pixelFormat,
        platformImage = out
    )
}

actual fun Image.grayscale(
    closeOriginal: Boolean
): Image {
    val out = ByteArray(width * height)
    val rgba = pixelFormat as? PixelFormat.RGBA
        ?: error("Grayscale supports only RGBA formats")
    var src = 0
    var dst = 0
    repeat(width * height) {
        val r = platformImage[src + rgba.rIndex].toInt() and 0xFF
        val g = platformImage[src + rgba.gIndex].toInt() and 0xFF
        val b = platformImage[src + rgba.bIndex].toInt() and 0xFF
        out[dst++] = (0.299 * r + 0.587 * g + 0.114 * b).toInt().coerceIn(0, 255).toByte()
        src += rgba.channels
    }

    if (closeOriginal) close()

    return IosImage(
        width = width,
        height = height,
        pixelFormat = PixelFormat.Grayscale,
        platformImage = out
    )
}


private fun <R> withBitmapContext(
    out: ByteArray,
    width: Int,
    height: Int,
    pixelFormat: PixelFormat,
    block: (CGContextRef?) -> R
): R {
    val colorSpace =
        if (pixelFormat == PixelFormat.Grayscale)
            CGColorSpaceCreateDeviceGray()
        else
            CGColorSpaceCreateDeviceRGB()

    return out.usePinned {
        val ctx = CGBitmapContextCreate(
            data = it.addressOf(0),
            width = width.toULong(),
            height = height.toULong(),
            bitsPerComponent = 8u,
            bytesPerRow = (width * pixelFormat.channels).toULong(),
            space = colorSpace,
            bitmapInfo = pixelFormat.cgBitmapInfo
        )
        try {
            block(ctx)
        } finally {
            CGContextRelease(ctx)
            CGColorSpaceRelease(colorSpace)
        }
    }
}



