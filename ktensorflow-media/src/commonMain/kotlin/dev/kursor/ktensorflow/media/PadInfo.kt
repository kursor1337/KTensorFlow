package dev.kursor.ktensorflow.media

import dev.kursor.ktensorflow.Delegate
import dev.kursor.ktensorflow.media.camera.Rect

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