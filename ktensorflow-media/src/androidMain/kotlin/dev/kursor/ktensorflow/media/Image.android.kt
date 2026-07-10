package dev.kursor.ktensorflow.media

import android.graphics.Bitmap
import androidx.core.graphics.BitmapCompat
import androidx.core.graphics.get

class AndroidImage(
    override val platformImage: PlatformImage,
    override val pixelFormat: PixelFormat
) : Image {

    private val bitmap: Bitmap get() = platformImage

    override val width: Int = bitmap.width
    override val height: Int = bitmap.height

    override operator fun get(x: Int, y: Int): Int {
        return bitmap[x, y]
    }

    override fun getPixels(): IntArray {
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)
        return pixels
    }
    override fun release() {
        if (!bitmap.isRecycled) {
            bitmap.recycle()
        }
    }
}

actual fun Image(
    width: Int,
    height: Int,
    pixelFormat: PixelFormat,
    pixels: IntArray
): Image {
    val bitmap = Bitmap.createBitmap(
        pixels,
        width,
        height,
        Bitmap.Config.ARGB_8888
    )
    return AndroidImage(bitmap, pixelFormat)
}