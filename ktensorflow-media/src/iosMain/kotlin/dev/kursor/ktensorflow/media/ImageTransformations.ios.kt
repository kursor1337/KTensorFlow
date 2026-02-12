package dev.kursor.ktensorflow.media

import dev.kursor.ktensorflow.media.camera.Rect
import kotlinx.cinterop.BetaInteropApi
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.convert
import kotlinx.cinterop.refTo
import kotlinx.cinterop.usePinned
import platform.CoreFoundation.CFDataRef
import platform.CoreGraphics.CGBitmapContextCreate
import platform.CoreGraphics.CGBitmapContextCreateImage
import platform.CoreGraphics.CGColorRenderingIntent
import platform.CoreGraphics.CGColorSpaceCreateDeviceGray
import platform.CoreGraphics.CGColorSpaceCreateDeviceRGB
import platform.CoreGraphics.CGContextDrawImage
import platform.CoreGraphics.CGContextRef
import platform.CoreGraphics.CGContextRotateCTM
import platform.CoreGraphics.CGContextTranslateCTM
import platform.CoreGraphics.CGDataProviderCreateWithCFData
import platform.CoreGraphics.CGImageAlphaInfo
import platform.CoreGraphics.CGImageCreate
import platform.CoreGraphics.CGImageCreateWithImageInRect
import platform.CoreGraphics.CGRectMake
import platform.CoreGraphics.kCGBitmapByteOrder32Big
import platform.CoreGraphics.kCGBitmapByteOrder32Little
import platform.Foundation.NSData
import platform.Foundation.create
import kotlin.math.PI

actual fun Image.resize(newWidth: Int, newHeight: Int): Image {
    val out = ByteArray(newWidth * newHeight * pixelFormat.channels)

    val ctx = createBitmapContext(out, newWidth, newHeight)

    withCGImage { cgImage ->
        CGContextDrawImage(
            ctx,
            CGRectMake(0.0, 0.0, newWidth.toDouble(), newHeight.toDouble()),
            cgImage
        )
    }

    return Image(newWidth, newHeight, pixelFormat, out)
}

actual fun Image.crop(rect: Rect): Image {
    val newWidth = rect.right - rect.left
    val newHeight = rect.bottom - rect.top

    val out = ByteArray(newWidth * newHeight * pixelFormat.channels)
    val ctx = createBitmapContext(out, newWidth, newHeight)

    withCGImage { cgImage ->
        val cropRect = CGRectMake(
            rect.left.toDouble(),
            rect.top.toDouble(),
            newWidth.toDouble(),
            newHeight.toDouble()
        )

        val cropped = CGImageCreateWithImageInRect(cgImage, cropRect)
            ?: error("Failed to crop CGImage")

        CGContextDrawImage(
            ctx,
            CGRectMake(0.0, 0.0, newWidth.toDouble(), newHeight.toDouble()),
            cropped
        )
    }

    return Image(newWidth, newHeight, pixelFormat, out)
}

actual fun Image.rotate(degrees: Float): Image {
    val radians = degrees * PI / 180.0

    val sin = kotlin.math.abs(kotlin.math.sin(radians))
    val cos = kotlin.math.abs(kotlin.math.cos(radians))

    val newWidth = (width * cos + height * sin).toInt()
    val newHeight = (width * sin + height * cos).toInt()

    val out = ByteArray(newWidth * newHeight * pixelFormat.channels)
    val ctx = createBitmapContext(out, newWidth, newHeight)

    CGContextTranslateCTM(ctx, newWidth / 2.0, newHeight / 2.0)
    CGContextRotateCTM(ctx, radians)
    CGContextTranslateCTM(ctx, -width / 2.0, -height / 2.0)

    withCGImage { cgImage ->
        CGContextDrawImage(
            ctx,
            CGRectMake(0.0, 0.0, width.toDouble(), height.toDouble()),
            cgImage
        )
    }

    return Image(newWidth, newHeight, pixelFormat, out)
}

actual fun Image.grayscale(): Image {
    val out = ByteArray(width * height)

    val rgba = pixelFormat as? PixelFormat.RGBA
        ?: error("Grayscale supports only RGBA formats")

    var src = 0
    var dst = 0

    repeat(width * height) {
        val r = data[src + rgba.rIndex].toInt() and 0xFF
        val g = data[src + rgba.gIndex].toInt() and 0xFF
        val b = data[src + rgba.bIndex].toInt() and 0xFF

        out[dst++] =
            (0.299 * r + 0.587 * g + 0.114 * b)
                .toInt()
                .coerceIn(0, 255)
                .toByte()

        src += rgba.channels
    }

    return Image(width, height, PixelFormat.Grayscale, out)
}


private fun Image.createBitmapContext(
    out: ByteArray,
    width: Int,
    height: Int
): CGContextRef {
    val colorSpace =
        if (pixelFormat == PixelFormat.Grayscale)
            CGColorSpaceCreateDeviceGray()
        else
            CGColorSpaceCreateDeviceRGB()

    return out.usePinned {
        CGBitmapContextCreate(
            it.addressOf(0),
            width.toULong(),
            height.toULong(),
            8u,
            (width * pixelFormat.channels).toULong(),
            colorSpace,
            pixelFormat.cgBitmapInfo
        ) ?: error("Failed to create bitmap context")
    }
}



