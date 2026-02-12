package dev.kursor.ktensorflow.media

import kotlin.math.PI

sealed interface PixelFormat {

    val channels: Int

    data class RGB(
        val rIndex: Int,
        val gIndex: Int,
        val bIndex: Int
    ) : PixelFormat {
        override val channels = 3
    }

    data class RGBA(
        val rIndex: Int,
        val gIndex: Int,
        val bIndex: Int,
        val aIndex: Int
    ) : PixelFormat {
        override val channels = 4
    }

    data object Grayscale : PixelFormat {
        override val channels = 1
    }

    companion object {
        val RGBA = RGBA(0, 1, 2, 3)
        val ARGB = RGBA(1, 2, 3,0)
        val BGRA = RGBA(2, 1, 0, 3)
        val ABGR = RGBA(3, 2, 1, 0)

        val RGB = RGB(0, 1, 2)
        val BGR = RGB(2, 1, 0)
    }
}