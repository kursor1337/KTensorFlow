package dev.kursor.ktensorflow.vision

import kotlin.math.roundToInt

/**
 * [Image] on iOS, backed by a raw pixel buffer.
 *
 * This is the entry point for frames that come from the platform, for example a camera
 * `CVPixelBuffer` drawn into a `CGBitmapContext`. The buffer must follow this layout exactly,
 * otherwise the pixels are read incorrectly:
 * - 8 bits per channel, `pixelFormat.channels` bytes per pixel;
 * - rows packed tightly: `width * pixelFormat.channels` bytes per row, no padding;
 * - channels of each pixel in the order given by the [PixelFormat] indices, so a context created
 *   with `kCGImageAlphaPremultipliedFirst | kCGBitmapByteOrder32Little` matches [PixelFormat.BGRA];
 * - for formats with alpha, color channels premultiplied by alpha, as CoreGraphics stores them.
 *
 * To build an image from plain packed ARGB values use the `Image(width, height, pixelFormat, pixels)`
 * factory instead.
 *
 * @throws IllegalArgumentException if the size is not positive or [pixels] does not hold exactly
 * `width * height` pixels.
 */
class IosImage(
    override val width: Int,
    override val height: Int,
    override val pixelFormat: PixelFormat,
    pixels: PlatformImage
) : Image {
    private val bytesPerPixel = pixelFormat.channels
    private val bytesPerRow = width * bytesPerPixel

    init {
        // Как на Android, где Bitmap отказывается от такого размера: иначе пустое или даже
        // отрицательное изображение создавалось и падало позже, выходом за массив в resize
        require(width > 0 && height > 0) { "Image size must be positive, got ${width}x$height" }
        // Неверная длина раньше проявлялась далеко отсюда: выходом за массив при чтении
        // пикселей или молча неверными цветами при построчном сдвиге
        require(pixels.size == height * bytesPerRow) {
            "Pixel buffer of ${width}x$height ${pixelFormat.channels}-channel image must hold " +
                "${height * bytesPerRow} bytes, got ${pixels.size}"
        }
    }

    private var pixelBuffer: ByteArray? = pixels

    /**
     * Pixel buffer backing this image.
     *
     * @throws IllegalStateException if the image has already been closed.
     */
    override val platformImage: PlatformImage
        get() = checkNotNull(pixelBuffer) {
            "Image ${width}x$height has already been closed"
        }

    private val data: ByteArray get() = platformImage

    override operator fun get(x: Int, y: Int): Int {
        if (x !in 0 until width || y !in 0 until height) return 0
        val o = y * bytesPerRow + x * bytesPerPixel
        return when (pixelFormat) {
            PixelFormat.Grayscale -> {
                // Контракт Image: наружу всегда упакованный ARGB, у серого R=G=B.
                // Без маски байт >= 128 приходил наружу отрицательным числом.
                val v = data[o].toInt() and 0xFF
                (0xFF shl 24) or (v shl 16) or (v shl 8) or v
            }
            is PixelFormat.RGB -> {
                val b = data[o + pixelFormat.bIndex].toInt() and 0xFF
                val g = data[o + pixelFormat.gIndex].toInt() and 0xFF
                val r = data[o + pixelFormat.rIndex].toInt() and 0xFF
                val a = 0xFF
                (a shl 24) or (r shl 16) or (g shl 8) or b
            }
            is PixelFormat.RGBA -> {
                // Буфер хранит premultiplied alpha, поэтому здесь нужно то же обратное
                // умножение, что и в getPixels: иначе один и тот же полупрозрачный пиксель
                // читался бы по-разному в зависимости от способа доступа.
                val a = data[o + pixelFormat.aIndex].toInt() and 0xFF
                if (a == 0xFF) {
                    val b = data[o + pixelFormat.bIndex].toInt() and 0xFF
                    val g = data[o + pixelFormat.gIndex].toInt() and 0xFF
                    val r = data[o + pixelFormat.rIndex].toInt() and 0xFF
                    (a shl 24) or (r shl 16) or (g shl 8) or b
                } else {
                    val b = data[o + pixelFormat.bIndex].toInt().unpremultiplyAlpha(a) and 0xFF
                    val g = data[o + pixelFormat.gIndex].toInt().unpremultiplyAlpha(a) and 0xFF
                    val r = data[o + pixelFormat.rIndex].toInt().unpremultiplyAlpha(a) and 0xFF
                    (a shl 24) or (r shl 16) or (g shl 8) or b
                }
            }
        }
    }

    override fun getPixels(): IntArray {
        val out = IntArray(width * height)
        getPixels(out)
        return out
    }

    override fun getPixels(buffer: IntArray) {
        // data - это геттер поверх platformImage, а индексы каналов лежат в data-классе:
        // в горячем цикле и то, и другое поднято в локальные переменные.
        val bytes = data

        when (pixelFormat) {
            PixelFormat.Grayscale -> {
                for (i in buffer.indices) {
                    val v = bytes[i].toInt() and 0xFF
                    buffer[i] = (0xFF shl 24) or (v shl 16) or (v shl 8) or v
                }
            }
            is PixelFormat.RGB -> {
                val rIndex = pixelFormat.rIndex
                val gIndex = pixelFormat.gIndex
                val bIndex = pixelFormat.bIndex
                for (i in buffer.indices) {
                    val o = 3 * i
                    val b = bytes[o + bIndex].toInt() and 0xFF
                    val g = bytes[o + gIndex].toInt() and 0xFF
                    val r = bytes[o + rIndex].toInt() and 0xFF
                    buffer[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                }
            }
            is PixelFormat.RGBA -> {
                val rIndex = pixelFormat.rIndex
                val gIndex = pixelFormat.gIndex
                val bIndex = pixelFormat.bIndex
                val aIndex = pixelFormat.aIndex
                for (i in buffer.indices) {
                    val o = 4 * i
                    val a = bytes[o + aIndex].toInt() and 0xFF
                    if (a == 0xFF) {
                        // Непрозрачный пиксель - обратное умножение на альфу тождественно,
                        // поэтому три float-деления с округлением можно не делать вовсе.
                        val b = bytes[o + bIndex].toInt() and 0xFF
                        val g = bytes[o + gIndex].toInt() and 0xFF
                        val r = bytes[o + rIndex].toInt() and 0xFF
                        buffer[i] = (a shl 24) or (r shl 16) or (g shl 8) or b
                    } else {
                        val b = bytes[o + bIndex].toInt().unpremultiplyAlpha(a) and 0xFF
                        val g = bytes[o + gIndex].toInt().unpremultiplyAlpha(a) and 0xFF
                        val r = bytes[o + rIndex].toInt().unpremultiplyAlpha(a) and 0xFF
                        buffer[i] = (a shl 24) or (r shl 16) or (g shl 8) or b
                    }
                }
            }
        }
    }

    /**
     * Releases the pixel buffer and marks the image as closed.
     *
     * On iOS there is no native handle to free, but dropping the buffer both makes it
     * collectable right away and keeps the lifecycle contract identical to Android, where
     * touching a closed image fails loudly. Without that symmetry a use-after-close slips
     * through iOS tests and only shows up on a device. Closing twice is safe.
     */
    override fun close() {
        pixelBuffer = null
    }
}

internal fun Int.unpremultiplyAlpha(a: Int): Int = if (a == 0) {
    0
} else {
    (this.toFloat() / a * 255).roundToInt()
}

actual fun Image(
    width: Int,
    height: Int,
    pixelFormat: PixelFormat,
    pixels: IntArray
): Image {
    val bytes = ByteArray(width * height * pixelFormat.channels)

    when (pixelFormat) {
        PixelFormat.Grayscale -> {
            for (i in pixels.indices) {
                bytes[i] = (pixels[i] and 0xFF).toByte()
            }
        }
        is PixelFormat.RGB -> {
            for (i in pixels.indices) {
                val p = pixels[i]
                val o = i * 3
                bytes[o + pixelFormat.rIndex] = ((p shr 16) and 0xFF).toByte()
                bytes[o + pixelFormat.gIndex] = ((p shr 8) and 0xFF).toByte()
                bytes[o + pixelFormat.bIndex] = (p and 0xFF).toByte()
            }
        }
        is PixelFormat.RGBA -> {
            for (i in pixels.indices) {
                val p = pixels[i]
                val o = i * 4
                val a = (p shr 24) and 0xFF

                val r = (((p shr 16) and 0xFF) * a / 255)
                val g = (((p shr 8) and 0xFF) * a / 255)
                val b = ((p and 0xFF) * a / 255)

                bytes[o + pixelFormat.aIndex] = a.toByte()
                bytes[o + pixelFormat.rIndex] = r.toByte()
                bytes[o + pixelFormat.gIndex] = g.toByte()
                bytes[o + pixelFormat.bIndex] = b.toByte()
            }
        }
    }
    return IosImage(width, height, pixelFormat, bytes)
}
