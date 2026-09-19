package dev.kursor.ktensorflow.vision

/**
 * Represents the structure and channel ordering of pixel data in an image.
 *
 * This sealed interface defines how color components (Red, Green, Blue, and Alpha) are
 * mapped to indices within a pixel's data array, facilitating conversion between
 * different image representations.
 */
sealed interface PixelFormat {

    /**
     * The number of channels (color components) in this pixel format.
     */
    val channels: Int

    /**
     * Represents a 3-channel pixel format containing Red, Green, and Blue components.
     *
     * @property rIndex The index of the red channel within the pixel data.
     * @property gIndex The index of the green channel within the pixel data.
     * @property bIndex The index of the blue channel within the pixel data.
     */
    data class RGB(
        val rIndex: Int,
        val gIndex: Int,
        val bIndex: Int
    ) : PixelFormat {
        override val channels = 3
    }

    /**
     * Represents a 4-channel pixel format containing Red, Green, Blue and Alpha components.
     *
     * @property rIndex The index of the red channel within the pixel data.
     * @property gIndex The index of the green channel within the pixel data.
     * @property bIndex The index of the blue channel within the pixel data.
     * @property aIndex The index of the alpha channel within the pixel data.
     */
    data class RGBA(
        val rIndex: Int,
        val gIndex: Int,
        val bIndex: Int,
        val aIndex: Int
    ) : PixelFormat {
        override val channels = 4
    }

    /**
     * Represents a 1-channel pixel format containing a single grayscale component.
     */
    data object Grayscale : PixelFormat {
        override val channels = 1
    }

    companion object {

        /**
         * Standard 4-channel pixel format with channels in the order: Red, Green, Blue, Alpha.
         */
        val RGBA = RGBA(0, 1, 2, 3)

        /**
         * Standard 4-channel pixel format with channels in the order: Alpha, Red, Green, Blue.
         */
        val ARGB = RGBA(1, 2, 3,0)

        /**
         * Standard 4-channel pixel format with channels in the order: Blue, Green, Red, Alpha.
         */
        val BGRA = RGBA(2, 1, 0, 3)

        /**
         * Standard 4-channel pixel format with channels in the order: Alpha, Blue, Green, Red.
         */
        val ABGR = RGBA(3, 2, 1, 0)

        /**
         * Standard 3-channel pixel format with channels in the order: Red, Green, Blue.
         */
        val RGB = RGB(0, 1, 2)

        /**
         * Standard 4-channel pixel format with channels in the order: Blue, Green, Red.
         */
        val BGR = RGB(2, 1, 0)
    }
}