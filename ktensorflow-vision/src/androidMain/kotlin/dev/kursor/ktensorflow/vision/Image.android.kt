package dev.kursor.ktensorflow.vision

import android.graphics.Bitmap
import androidx.core.graphics.get

class AndroidImage(
    override val platformImage: PlatformImage,
    override val pixelFormat: PixelFormat
) : Image {

    private val bitmap: Bitmap get() = platformImage

    override val width: Int = bitmap.width
    override val height: Int = bitmap.height

    override operator fun get(x: Int, y: Int): Int {
        // Вне границ возвращается прозрачный чёрный, как и на iOS: Bitmap.getPixel на
        // таких координатах бросает исключение, и одна и та же общая логика вела себя
        // на двух платформах по-разному.
        if (x !in 0 until width || y !in 0 until height) return 0
        return bitmap[x, y]
    }

    override fun getPixels(): IntArray {
        val pixels = IntArray(width * height)
        getPixels(pixels)
        return pixels
    }

    override fun getPixels(buffer: IntArray) {
        bitmap.getPixels(buffer, 0, width, 0, 0, width, height)
    }

    override fun close() {
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
    // Для Grayscale хранится один канал - младший байт пикселя, ровно как его читают
    // tensorize и iOS-реализация. Без этого приведения bitmap оставался цветным, тег
    // формата врал, а getPixels возвращал цвета там, где iOS уже отдавал серое.
    val argb = if (pixelFormat == PixelFormat.Grayscale) {
        IntArray(pixels.size) { i ->
            val v = pixels[i] and 0xFF
            (0xFF shl 24) or (v shl 16) or (v shl 8) or v
        }
    } else {
        pixels
    }

    val bitmap = Bitmap.createBitmap(
        argb,
        width,
        height,
        Bitmap.Config.ARGB_8888
    )
    return AndroidImage(bitmap, pixelFormat)
}