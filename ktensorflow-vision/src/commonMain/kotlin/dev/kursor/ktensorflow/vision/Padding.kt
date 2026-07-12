package dev.kursor.ktensorflow.vision

/**
 * Contains metadata about a padding and resizing operation.
 *
 * This class stores the original dimensions, the final dimensions, the offsets applied,
 * and the scaling factor used during the [Image.resizeWithPad] operation. This information
 * is typically used to map coordinates from the padded image back to the original image space.
 *
 * @property originalWidth The width of the image before resizing.
 * @property originalHeight The height of the image before resizing.
 * @property targetWidth The width of the resulting image after resizing and padding.
 * @property targetHeight The height of the resulting image after resizing and padding.
 * @property padX The number of pixels added as padding to the left side of the image.
 * @property padY The number of pixels added as padding to the top side of the image.
 * @property scale The scaling factor used during the resize operation.
 */
data class PadInfo(
    val originalWidth: Int,
    val originalHeight: Int,
    val targetWidth: Int,
    val targetHeight: Int,
    val padX: Int,
    val padY: Int,
    val scale: Float
)

/**
 * Represents an image that has been resized and padded, preserving the metadata of the transformation.
 *
 * This class uses delegation to provide the [Image] interface while also exposing [info],
 * which contains the necessary details to map coordinates from this padded image back to
 * the original source image dimensions.
 *
 * @property delegate The underlying [Image] instance containing the resized and padded pixel data.
 * @property info The [PadInfo] containing the scaling factor, offsets, and original dimensions.
 */
class PaddedImage(
    val delegate: Image,
    val info: PadInfo
) : Image by delegate

/**
 * Resizes the image to fit within the specified [targetWidth] and [targetHeight] while maintaining
 * its original aspect ratio.
 *
 * The image is scaled down or up to fit the target dimensions, and any remaining space is filled
 * with [padColorArgb] to reach the exact target size. The resized image is centered within the
 * padded area.
 *
 * @param targetWidth The desired width of the resulting image.
 * @param targetHeight The desired height of the resulting image.
 * @param padColorArgb The color to use for padding the image.
 * @return A [PaddedImage] containing the resized and padded image.
 */
fun Image.resizeWithPad(
    targetWidth: Int,
    targetHeight: Int,
    padColorArgb: Int = 0xFF000000.toInt()
): PaddedImage {
    val scale = minOf(
        targetWidth.toFloat() / width.toFloat(),
        targetHeight.toFloat() / height.toFloat()
    )

    val scaledWidth = (width * scale).toInt()
    val scaledHeight = (height * scale).toInt()

    val padX = (targetWidth - scaledWidth) / 2
    val padY = (targetHeight - scaledHeight) / 2

    val scaledImage = this.resize(scaledWidth, scaledHeight)
    val scaledPixels = scaledImage.getPixels()

    val paddedPixels = IntArray(targetWidth * targetHeight) { padColorArgb }

    for (y in 0 until scaledHeight) {
        val srcOffset = y * scaledWidth
        val dstOffset = (y + padY) * targetWidth + padX
        
        scaledPixels.copyInto(
            destination = paddedPixels,
            destinationOffset = dstOffset,
            startIndex = srcOffset,
            endIndex = srcOffset + scaledWidth
        )
    }

    scaledImage.close()

    val finalImage = Image(targetWidth, targetHeight, pixelFormat, paddedPixels)

    return PaddedImage(
        delegate = finalImage,
        info = PadInfo(
            width,
            height,
            targetWidth,
            targetHeight,
            padX,
            padY,
            scale
        )
    )
}