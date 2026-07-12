package dev.kursor.ktensorflow.media


/**
 * A cross-platform image representation.
 *
 * This interface abstracts platform-specific image types.
 * It implements [AutoCloseable] to ensure that native resources are properly released.
 * The coordinate system starts at (0,0) at the top-left corner.
 */
interface Image : AutoCloseable {
    /** The width of the image in pixels. */
    val width: Int

    /** The height of the image in pixels. */
    val height: Int

    /** The internal pixel format used by this image. */
    val pixelFormat: PixelFormat

    /**
     * The underlying platform-specific image object.
     */
    val platformImage: PlatformImage

    /**
     * Returns the pixel value at the specified coordinates as a packed ARGB Int (0xAARRGGBB).
     * For grayscale images, the R, G, and B components are identical.
     */
    operator fun get(x: Int, y: Int): Int

    /**
     * Returns all pixels as an [IntArray] of packed ARGB values in row-major order.
     * Note: This operation may involve copying and memory allocation.
     */
    fun getPixels(): IntArray

    /**
     * Copies the image pixels into the provided [buffer] as packed ARGB values.
     * @param buffer The destination array, must have a size of at least [width] * [height].
     */
    fun getPixels(buffer: IntArray)

    /**
     * Manually releases any native resources held by this image.
     * Prefer using [use] blocks over calling this directly.
     * Implementation of [AutoCloseable.close] that delegates to [release].
     */
    override fun close()
}

/**
 * Creates a new [Image] instance from the provided pixel data.
 *
 * @param width The width of the new image.
 * @param height The height of the new image.
 * @param pixelFormat The format to be used for internal storage.
 * @param pixels Initial pixel data as packed ARGB values (0xAARRGGBB).
 * @return A platform-specific implementation of [Image].
 */
expect fun Image(
    width: Int,
    height: Int,
    pixelFormat: PixelFormat,
    pixels: IntArray
): Image