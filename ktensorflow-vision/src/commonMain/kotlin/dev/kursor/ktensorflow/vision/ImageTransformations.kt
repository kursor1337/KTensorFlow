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
    newHeight: Int
): Image

/**
 * Crops the image to the specified rectangle.
 *
 * @param rect The rectangle to crop the image to.
 * @return A new [Image] containing the cropped region.
 */
expect fun Image.crop(rect: Rect): Image

/**
 * Rotates the image by the specified angle.
 *
 * @param degrees The angle to rotate the image by.
 * @return A new [Image] containing the rotated image.
 */
expect fun Image.rotate(degrees: Float): Image

/**
 * Converts the image to grayscale.
 *
 * @return A new [Image] containing the grayscale image.
 */
expect fun Image.grayscale(): Image
