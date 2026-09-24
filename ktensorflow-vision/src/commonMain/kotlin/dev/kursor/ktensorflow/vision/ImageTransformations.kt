package dev.kursor.ktensorflow.vision

/**
 * Resizes the image to the specified dimensions.
 *
 * @param newWidth The new width of the image.
 * @param newHeight The new height of the image.
 * @return A new [Image] with the specified dimensions.
 */
expect fun Image.resize(
    newWidth: Int,
    newHeight: Int,
    closeOriginal: Boolean = true
): Image

/**
 * Crops the image to the specified rectangle.
 *
 * @param rect The rectangle to crop the image to.
 * @return A new [Image] containing the cropped region.
 */
expect fun Image.crop(rect: Rect, closeOriginal: Boolean = true): Image

/**
 * Rotates the image by the specified angle.
 *
 * @param degrees The angle to rotate the image by.
 * @return A new [Image] containing the rotated image.
 */
expect fun Image.rotate(degrees: Float, closeOriginal: Boolean = true): Image

/**
 * Converts the image to grayscale.
 *
 * The luma of each pixel follows ITU-R 601 (`0.299 * R + 0.587 * G + 0.114 * B`) and is computed by
 * the platform's accelerated implementation: `ColorMatrix` on Android and vImage on iOS. They
 * round differently, so the same pixel may differ by one level between platforms.
 *
 * @param closeOriginal whether to close this image after the conversion.
 * @return A new [Image] in [PixelFormat.Grayscale].
 */
expect fun Image.grayscale(closeOriginal: Boolean = true): Image
