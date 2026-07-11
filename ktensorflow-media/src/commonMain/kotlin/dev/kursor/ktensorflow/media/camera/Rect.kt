package dev.kursor.ktensorflow.media.camera

import dev.kursor.ktensorflow.media.PadInfo

data class Rect(
    val left: Int,
    val top: Int,
    val right: Int,
    val bottom: Int
) {
    // Площадь прямоугольника
    val area: Int
        get() = maxOf(0, right - left) * maxOf(0, bottom - top)

    /**
     * Вычисляет IoU (Intersection over Union) с другим Rect.
     * Метрика от 0.0f до 1.0f.
     */
    fun intersectionOverUnion(other: Rect): Float {
        val iLeft = maxOf(this.left, other.left)
        val iTop = maxOf(this.top, other.top)
        val iRight = minOf(this.right, other.right)
        val iBottom = minOf(this.bottom, other.bottom)

        val iArea = maxOf(0, iRight - iLeft) * maxOf(0, iBottom - iTop)
        if (iArea == 0) return 0f

        val unionArea = this.area + other.area - iArea
        return iArea.toFloat() / unionArea.toFloat()
    }

    companion object {
        /**
         * Переводит нормализованные координаты (0.0..1.0) в пиксели исходного изображения.
         */
        fun fromNormalized(
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
    }
}