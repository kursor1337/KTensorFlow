package dev.kursor.ktensorflow.media

import android.graphics.Bitmap
import androidx.core.graphics.BitmapCompat
import androidx.core.graphics.get

actual class Image(
    val bitmap: Bitmap,
    actual val pixelFormat: PixelFormat
) {

    actual val width: Int = bitmap.width
    actual val height: Int = bitmap.height

    actual operator fun get(x: Int, y: Int): Int {
        return bitmap[x, y]
    }

    actual fun getPixels(): IntArray {
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)
        return pixels
    }
}