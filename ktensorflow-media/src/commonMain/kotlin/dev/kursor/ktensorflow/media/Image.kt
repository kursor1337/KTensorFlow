package dev.kursor.ktensorflow.media

interface Image {
    val width: Int
    val height: Int
    val pixelFormat: PixelFormat
    val platformImage: PlatformImage
    operator fun get(x: Int, y: Int): Int
    fun getPixels(): IntArray
    fun release()
}

expect fun Image(
    width: Int,
    height: Int,
    pixelFormat: PixelFormat,
    pixels: IntArray
): Image