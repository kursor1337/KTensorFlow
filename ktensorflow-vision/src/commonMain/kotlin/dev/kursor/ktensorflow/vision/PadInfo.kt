package dev.kursor.ktensorflow.vision

data class PadInfo(
    val originalWidth: Int,
    val originalHeight: Int,
    val targetWidth: Int,
    val targetHeight: Int,
    val padX: Int,
    val padY: Int,
    val scale: Float
)

class PaddedImage(
    val delegate: Image,
    val info: PadInfo
) : Image by delegate