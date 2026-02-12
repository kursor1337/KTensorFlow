package dev.kursor.ktensorflow.media

import dev.kursor.ktensorflow.media.camera.Rect

expect fun Image.resize(
    newWidth: Int,
    newHeight: Int
): Image
expect fun Image.crop(rect: Rect): Image
expect fun Image.rotate(degrees: Float): Image

expect fun Image.grayscale(): Image
