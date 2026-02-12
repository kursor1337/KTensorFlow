package dev.kursor.ktensorflow.media

import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.ImageBitmapConfig

expect class Image {
    val width: Int
    val height: Int
    val pixelFormat: PixelFormat
    operator fun get(x: Int, y: Int): Int
    fun getPixels(): IntArray
}