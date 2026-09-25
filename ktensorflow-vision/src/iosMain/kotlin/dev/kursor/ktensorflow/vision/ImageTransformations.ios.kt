package dev.kursor.ktensorflow.vision

import kotlinx.cinterop.addressOf
import kotlinx.cinterop.usePinned
import platform.CoreGraphics.CGBitmapContextCreate
import platform.CoreGraphics.CGBlendMode
import platform.CoreGraphics.CGColorSpaceCreateDeviceGray
import platform.CoreGraphics.CGColorSpaceCreateDeviceRGB
import platform.CoreGraphics.CGColorSpaceRelease
import platform.CoreGraphics.CGContextDrawImage
import platform.CoreGraphics.CGContextFillRect
import platform.CoreGraphics.CGContextRef
import platform.CoreGraphics.CGContextRelease
import platform.CoreGraphics.CGContextRotateCTM
import platform.CoreGraphics.CGContextSetBlendMode
import platform.CoreGraphics.CGContextSetGrayFillColor
import platform.CoreGraphics.CGContextSetRGBFillColor
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
    requirePositiveSize(newWidth, newHeight)

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

internal actual fun Image.drawLetterboxed(
    scaledWidth: Int,
    scaledHeight: Int,
    targetWidth: Int,
    targetHeight: Int,
    padX: Int,
    padY: Int,
    padColorArgb: Int
): Image {
    val out = ByteArray(targetWidth * targetHeight * pixelFormat.channels)
    withBitmapContext(out, targetWidth, targetHeight, pixelFormat) { ctx ->
        // Copy, а не обычное наложение: и заливка, и картинка заменяют пиксели, как при
        // копировании массива, иначе полупрозрачные пиксели смешались бы с цветом полей
        CGContextSetBlendMode(ctx, CGBlendMode.kCGBlendModeCopy)

        val r = ((padColorArgb shr 16) and 0xFF) / 255.0
        val g = ((padColorArgb shr 8) and 0xFF) / 255.0
        val b = (padColorArgb and 0xFF) / 255.0
        // У RGB нет альфы: поля хранят цвет как есть, как и фабрика Image для этого формата
        val a = if (pixelFormat is PixelFormat.RGB) 1.0 else ((padColorArgb ushr 24) and 0xFF) / 255.0
        if (pixelFormat == PixelFormat.Grayscale) {
            CGContextSetGrayFillColor(ctx, b, a)
        } else {
            CGContextSetRGBFillColor(ctx, r, g, b, a)
        }
        CGContextFillRect(ctx, CGRectMake(0.0, 0.0, targetWidth.toDouble(), targetHeight.toDouble()))

        withCGImage { cgImage ->
            CGContextDrawImage(
                c = ctx,
                // Ось Y у CoreGraphics направлена вверх: отступ сверху превращается в отступ снизу
                rect = CGRectMake(
                    x = padX.toDouble(),
                    y = (targetHeight - padY - scaledHeight).toDouble(),
                    width = scaledWidth.toDouble(),
                    height = scaledHeight.toDouble()
                ),
                image = cgImage
            )
        }
    }

    return IosImage(targetWidth, targetHeight, pixelFormat, out)
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
