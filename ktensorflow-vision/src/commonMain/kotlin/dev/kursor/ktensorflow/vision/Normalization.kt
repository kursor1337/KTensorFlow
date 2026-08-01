package dev.kursor.ktensorflow.vision


/**
 * Represents normalization parameters used to scale image pixel values.
 *
 * The normalization is typically applied using the formula:
 * `output = (input - mean) / std`
 *
 * @property meanR Mean value for the Red channel.
 * @property meanG Mean value for the Green channel.
 * @property meanB Mean value for the Blue channel.
 * @property meanA Mean value for the Alpha channel.
 * @property stdR Standard deviation for the Red channel.
 * @property stdG Standard deviation for the Green channel.
 * @property stdB Standard deviation for the Blue channel.
 * @property stdA Standard deviation for the Alpha channel.
 */
data class Normalization(
    val meanR: Float = 0f,
    val meanG: Float = 0f,
    val meanB: Float = 0f,
    val meanA: Float = 0f,
    val stdR: Float = 1f,
    val stdG: Float = 1f,
    val stdB: Float = 1f,
    val stdA: Float = 1f
) {
    companion object {
        /**
         * Represents a normalization configuration that performs no transformation,
         * keeping the original pixel values unchanged.
         */
        val None = Normalization()

        /**
         * Normalizes pixel values from the range [0, 255] to the range [0.0, 1.0].
         *
         * This is achieved by setting the mean to 0 and the standard deviation to 255 for each color channel.
         */
        val ZeroToOne = Normalization(
            meanR = 0f,
            meanG = 0f,
            meanB = 0f,
            stdR = 255f,
            stdG = 255f,
            stdB = 255f
        )

        /**
         * Normalizes pixel values from the [0, 255] range to the [-1.0, 1.0] range.
         *
         * Uses a mean of 127.5 and a standard deviation of 127.5 for RGB channels.
         */
        val MinusOneToOne = Normalization(
            meanR = 127.5f,
            meanG = 127.5f,
            meanB = 127.5f,
            stdR = 127.5f,
            stdG = 127.5f,
            stdB = 127.5f
        )

        /**
         * Standard normalization constants for models pre-trained on the ImageNet dataset.
         *
         * These values represent the global mean and standard deviation of the ImageNet
         * dataset (calculated on the 0-255 range), commonly used for models like
         * ResNet, VGG, and others.
         */
        val ImageNet = Normalization(
            meanR = 123.675f,
            meanG = 116.28f,
            meanB = 103.53f,
            stdR = 58.395f,
            stdG = 57.12f,
            stdB = 57.375f
        )
    }
}