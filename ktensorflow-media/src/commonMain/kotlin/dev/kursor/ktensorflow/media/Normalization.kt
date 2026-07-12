package dev.kursor.ktensorflow.media


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
        val None = Normalization()

        val ZeroToOne = Normalization(
            meanR = 0f,
            meanG = 0f,
            meanB = 0f,
            stdR = 255f,
            stdG = 255f,
            stdB = 255f
        )

        val MinusOneToOne = Normalization(
            meanR = 127.5f,
            meanG = 127.5f,
            meanB = 127.5f,
            stdR = 127.5f,
            stdG = 127.5f,
            stdB = 127.5f
        )

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