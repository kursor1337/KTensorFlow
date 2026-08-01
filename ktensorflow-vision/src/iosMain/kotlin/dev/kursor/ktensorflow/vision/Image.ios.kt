package dev.kursor.ktensorflow.vision

import kotlin.math.roundToInt

/**
 * Actual implementation of Image for iOS, wrapping a UIImage.
 * It uses CoreGraphics to extract and manage the pixel buffer in ARGB_8888 format
 * for efficient pixel access.
 */
class IosImage(
    override val width: Int,
    override val height: Int,
    override val pixelFormat: PixelFormat,
    override val platformImage: PlatformImage // alpha always premultiplied
) : Image {
    private val bytesPerPixel = pixelFormat.channels
    private val bytesPerRow = width * bytesPerPixel

    private val data: ByteArray get() = platformImage

    override operator fun get(x: Int, y: Int): Int {
        if (x !in 0 until width || y !in 0 until height) return 0
        val o = y * bytesPerRow + x * bytesPerPixel
        return when (pixelFormat) {
            PixelFormat.Grayscale -> {
                data[o].toInt()
            }
            is PixelFormat.RGB -> {
                val b = data[o + pixelFormat.bIndex].toInt() and 0xFF
                val g = data[o + pixelFormat.gIndex].toInt() and 0xFF
                val r = data[o + pixelFormat.rIndex].toInt() and 0xFF
                val a = 0xFF
                (a shl 24) or (r shl 16) or (g shl 8) or b
            }
            is PixelFormat.RGBA -> {
                val b = data[o + pixelFormat.bIndex].toInt() and 0xFF
                val g = data[o + pixelFormat.gIndex].toInt() and 0xFF
                val r = data[o + pixelFormat.rIndex].toInt() and 0xFF
                val a = data[o + pixelFormat.aIndex].toInt() and 0xFF
                (a shl 24) or (r shl 16) or (g shl 8) or b
            }
        }
    }

    override fun getPixels(): IntArray {
        val out = IntArray(width * height)
        getPixels(out)
        return out
    }

    override fun getPixels(buffer: IntArray) {
        when (pixelFormat) {
            PixelFormat.Grayscale -> {
                for (i in buffer.indices) {
                    buffer[i] = data[i].toInt() and 0xFF
                }
            }
            is PixelFormat.RGB -> {
                for (i in buffer.indices) {
                    val o = 3 * i
                    val b = data[o + pixelFormat.bIndex].toInt() and 0xFF
                    val g = data[o + pixelFormat.gIndex].toInt() and 0xFF
                    val r = data[o + pixelFormat.rIndex].toInt() and 0xFF
                    val a = 0xFF
                    buffer[i] = (a shl 24) or (r shl 16) or (g shl 8) or b
                }
            }
            is PixelFormat.RGBA -> {
                for (i in buffer.indices) {
                    val o = 4 * i
                    val a = data[o + pixelFormat.aIndex].toInt() and 0xFF
                    val b = data[o + pixelFormat.bIndex].toInt().unpremultiplyAlpha(a) and 0xFF
                    val g = data[o + pixelFormat.gIndex].toInt().unpremultiplyAlpha(a) and 0xFF
                    val r = data[o + pixelFormat.rIndex].toInt().unpremultiplyAlpha(a) and 0xFF

                    buffer[i] = (a shl 24) or (r shl 16) or (g shl 8) or b
                }
            }
        }
    }

    override fun close() {
        // do nothing
    }
}

private fun Int.unpremultiplyAlpha(a: Int): Int = if (a == 0) {
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
