package dev.kursor.ktensorflow.vision

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
     *
     * Coordinates outside the image return a transparent black pixel (0) rather than failing,
     * so sampling around edges does not need explicit bounds checks.
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
     * Manually releases the resources held by this image: the underlying `Bitmap` on Android
     * and the pixel buffer on iOS.
     *
     * Once closed, the image must not be used any more: reading pixels or transforming it
     * fails on every platform. Closing an already closed image is safe.
     *
     * Prefer using [use] blocks over calling this directly.
     */
    override fun close()
}

/**
 * Creates a new [Image] instance from the provided pixel data.
 *
 * @param width The width of the new image.
 * @param height The height of the new image.
 * @param pixelFormat The format to be used for internal storage.
 * @param pixels Initial pixel data as packed ARGB values (0xAARRGGBB). For
 * [PixelFormat.Grayscale] the single stored channel is taken from the lowest byte of each
 * value, the same byte [tensorize] reads, so passing colored pixels keeps their blue channel.
 * Use [grayscale] to convert colors to luminance properly.
 * @return A platform-specific implementation of [Image].
 */
expect fun Image(
    width: Int,
    height: Int,
    pixelFormat: PixelFormat,
    pixels: IntArray
): Image