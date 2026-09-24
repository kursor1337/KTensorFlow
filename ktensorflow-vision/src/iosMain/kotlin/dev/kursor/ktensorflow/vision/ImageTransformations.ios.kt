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
        pixels = out
    )
}

actual fun Image.crop(
    rect: Rect,
    closeOriginal: Boolean
): Image {
    requireInsideImage(rect)

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
        pixels = out
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
            // Знак инвертирован намеренно: у CoreGraphics ось Y направлена вверх, и
            // положительный угол выглядел бы поворотом ПРОТИВ часовой стрелки. На Android
            // Matrix.postRotate, как и EXIF с CameraX, считает положительный угол поворотом
            // ПО часовой, и rotate(90f) обязан давать одну и ту же картинку на обеих платформах.
            angle = -radians
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
        pixels = out
    )
}

actual fun Image.grayscale(
    closeOriginal: Boolean
): Image {
    val out = ByteArray(width * height)
    val bytes = platformImage

    // Поддержаны все форматы: на Android grayscale работает с любым изображением.
    // Яркость считает vImage (SIMD), как на Android её считает ColorMatrix
    when (val format = pixelFormat) {
        PixelFormat.Grayscale -> bytes.copyInto(out, endIndex = out.size)

        is PixelFormat.RGB -> {
            val fourChannels = expandToFourChannels(bytes, width, height)
            lumaRec601(fourChannels, format.withAlpha, out, width, height)
        }

        // Буфер premultiplied: без обратного умножения полупрозрачные пиксели потемнели бы,
        // и результат разошёлся бы с Android, где ColorMatrix работает с прямым цветом
        is PixelFormat.RGBA -> {
            val straight = unpremultiply(bytes, format, width, height)
            lumaRec601(straight, format, out, width, height)
        }
    }

    if (closeOriginal) close()

    return IosImage(
        width = width,
        height = height,
        pixelFormat = PixelFormat.Grayscale,
        pixels = out
    )
}

internal fun <R> withBitmapContext(
    out: ByteArray,
    width: Int,
    height: Int,
    pixelFormat: PixelFormat,
    block: (CGContextRef?) -> R
): R {
    // CoreGraphics не рисует в 3-канальный буфер: рисуем в RGBA и упаковываем обратно
    // Цвет остаётся premultiplied, то есть полупрозрачные края после поворота ложатся на
    // чёрный фон - у RGB нет альфы, чтобы их хранить
    if (pixelFormat is PixelFormat.RGB) {
        val rgba = ByteArray(width * height * 4)
        val result = withBitmapContext(rgba, width, height, pixelFormat.coreGraphicsFormat, block)
        packToThreeChannels(rgba, out, width, height)
        return result
    }

    val colorSpace =
        if (pixelFormat == PixelFormat.Grayscale) {
            CGColorSpaceCreateDeviceGray()
        } else {
            CGColorSpaceCreateDeviceRGB()
        }

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
