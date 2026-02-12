package dev.kursor.ktensorflow.media

import kotlin.math.roundToInt

/**
 * Actual implementation of Image for iOS, wrapping a UIImage.
 * It uses CoreGraphics to extract and manage the pixel buffer in ARGB_8888 format
 * for efficient pixel access.
 */
actual class Image(
    actual val width: Int,
    actual val height: Int,
    actual val pixelFormat: PixelFormat,
    val data: ByteArray // alpha always premultiplied
) {
    private val bytesPerPixel = pixelFormat.channels
    private val bytesPerRow = width * bytesPerPixel

    actual operator fun get(x: Int, y: Int): Int {
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

    actual fun getPixels(): IntArray {
        val out = IntArray(width * height)

        when (pixelFormat) {
            PixelFormat.Grayscale -> {
                for (i in out.indices) {
                    out[i] = data[i].toInt() and 0xFF
                }
            }
            is PixelFormat.RGB -> {
                for (i in out.indices) {
                    val b = data[pixelFormat.bIndex].toInt() and 0xFF
                    val g = data[pixelFormat.gIndex].toInt() and 0xFF
                    val r = data[pixelFormat.rIndex].toInt() and 0xFF
                    val a = 0xFF
                    out[i] = (a shl 24) or (r shl 16) or (g shl 8) or b
                }
            }
            is PixelFormat.RGBA -> {
                for (i in out.indices) {
                    val a = data[pixelFormat.aIndex].toInt() and 0xFF
                    val b = data[pixelFormat.bIndex].toInt().unpremultiplyAlpha(a) and 0xFF
                    val g = data[pixelFormat.gIndex].toInt().unpremultiplyAlpha(a) and 0xFF
                    val r = data[pixelFormat.rIndex].toInt().unpremultiplyAlpha(a) and 0xFF

                    out[i] = (a shl 24) or (r shl 16) or (g shl 8) or b
                }
            }
        }
        return out
    }
}

private fun Int.unpremultiplyAlpha(a: Int): Int = (this.toFloat() / a * 255).roundToInt()
