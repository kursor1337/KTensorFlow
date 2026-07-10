package dev.kursor.ktensorflow.media

import dev.kursor.ktensorflow.media.camera.Rect

/**
 * Переводит нормализованные координаты (0.0..1.0) в пиксели исходного изображения.
 */
fun Rect.Companion.fromNormalized(
    ymin: Float,
    xmin: Float,
    ymax: Float,
    xmax: Float,
    padInfo: PadInfo
): Rect {
    val absLeft = xmin * padInfo.targetWidth
    val absTop = ymin * padInfo.targetHeight
    val absRight = xmax * padInfo.targetWidth
    val absBottom = ymax * padInfo.targetHeight

    val origLeft = ((absLeft - padInfo.padX) / padInfo.scale).toInt()
    val origTop = ((absTop - padInfo.padY) / padInfo.scale).toInt()
    val origRight = ((absRight - padInfo.padX) / padInfo.scale).toInt()
    val origBottom = ((absBottom - padInfo.padY) / padInfo.scale).toInt()

    return Rect(
        left = origLeft.coerceIn(0, padInfo.originalWidth),
        top = origTop.coerceIn(0, padInfo.originalHeight),
        right = origRight.coerceIn(0, padInfo.originalWidth),
        bottom = origBottom.coerceIn(0, padInfo.originalHeight)
    )
}