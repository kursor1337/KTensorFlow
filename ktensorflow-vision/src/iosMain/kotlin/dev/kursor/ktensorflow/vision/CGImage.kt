package dev.kursor.ktensorflow.vision

import kotlinx.cinterop.Pinned
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.usePinned
import platform.CoreGraphics.CGColorRenderingIntent
import platform.CoreGraphics.CGColorSpaceCreateDeviceGray
import platform.CoreGraphics.CGColorSpaceCreateDeviceRGB
import platform.CoreGraphics.CGColorSpaceRelease
import platform.CoreGraphics.CGDataProviderCreateWithData
import platform.CoreGraphics.CGDataProviderRelease
import platform.CoreGraphics.CGImageCreate
import platform.CoreGraphics.CGImageRef
import platform.CoreGraphics.CGImageRelease

/**
 * Runs [block] with a CoreGraphics view of this image's pixels.
 *
 * The `CGImage` is only valid inside [block]: it reads the pixel buffer of the image directly and
 * is released right after the block returns, so it must not be retained or used afterwards.
 */
fun <T> Image.withCGImage(block: (CGImageRef) -> T): T {
    // CoreGraphics не знает 3-канальных раскладок: RGB и BGR отдаются ей развёрнутыми в 4 канала
    val bytes = if (pixelFormat is PixelFormat.RGB) {
        expandToFourChannels(platformImage, width, height)
    } else {
        platformImage
    }
    val cgFormat = pixelFormat.coreGraphicsFormat

    // Данные должны оставаться запиненными (pinned) до тех пор, пока CGImage реально
    // не прочитан CoreGraphics (например, внутри CGContextDrawImage) - если распиновать
    // сразу после CGImageCreate, GC Kotlin/Native может переместить память ДО чтения,
    // и CGImage окажется битым/пустым (было воспроизведено на неоднородных изображениях).
    return bytes.usePinned { pinned ->
        val cgImage = createCGImage(pinned, bytes.size, width, height, cgFormat)
        try {
            block(cgImage)
        } finally {
            CGImageRelease(cgImage)
        }
    }
}

private fun createCGImage(
    pinnedData: Pinned<ByteArray>,
    size: Int,
    width: Int,
    height: Int,
    pixelFormat: PixelFormat
): CGImageRef {
    val colorSpace =
        if (pixelFormat == PixelFormat.Grayscale) {
            CGColorSpaceCreateDeviceGray()
        } else {
            CGColorSpaceCreateDeviceRGB()
        }

    val provider = CGDataProviderCreateWithData(
        null,
        pinnedData.addressOf(0),
        size.toULong(),
        null
    )!!

    // CGImage удерживает provider и цветовое пространство сам, наши ссылки можно отпустить
    val cgImage = CGImageCreate(
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
    CGDataProviderRelease(provider)
    CGColorSpaceRelease(colorSpace)
    return cgImage
}
