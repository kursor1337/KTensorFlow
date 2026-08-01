package dev.kursor.ktensorflow.vision

import kotlin.math.roundToInt

/**
 * Represents a rectangular bounding box in 2D integer pixel coordinates.
 *
 * This class is primarily used for computer vision tasks to define regions of interest within an image.
 * It includes utility methods for geometric calculations and factory methods to transform bounding
 * box coordinates from various model output formats (Normalized, YOLO, COCO) back to the
 * original image coordinate space using [PadInfo].
 *
 * @property left The x-coordinate of the left edge.
 * @property top The y-coordinate of the top edge.
 * @property right The x-coordinate of the right edge.
 * @property bottom The y-coordinate of the bottom edge.
 */
data class Rect(
    val left: Int,
    val top: Int,
    val right: Int,
    val bottom: Int
) {

    /**
     * The area of the rectangle.
     */
    val area: Int
        get() = maxOf(0, right - left) * maxOf(0, bottom - top)

    val width: Int
        get() = maxOf(0, right - left)

    val height: Int
        get() = maxOf(0, bottom - top)

    companion object {

        /**
         * Converts normalized coordinates (0.0 to 1.0) from a padded target image back to
         * absolute pixel coordinates of the original image.
         *
         * @param ymin The normalized minimum y-coordinate.
         * @param xmin The normalized minimum x-coordinate.
         * @param ymax The normalized maximum y-coordinate.
         * @param xmax The normalized maximum x-coordinate.
         * @param padInfo Information about the padding and scaling applied to the original image.
         * @return A [Rect] representing the bounding box in the original image's pixel space.
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

            val rect = Rect(
                left = origLeft.coerceIn(0, padInfo.originalWidth),
                top = origTop.coerceIn(0, padInfo.originalHeight),
                right = origRight.coerceIn(0, padInfo.originalWidth),
                bottom = origBottom.coerceIn(0, padInfo.originalHeight)
            )

            return rect
        }

        /**
         * Creates a [Rect] from YOLO format coordinates (center x, center y, width, height).
         *
         * This method converts normalized YOLO coordinates back to the original image coordinate space
         * by accounting for any scaling or padding applied during preprocessing, as defined in [padInfo].
         *
         * @param cx The normalized x-coordinate of the center of the bounding box (0.0 to 1.0).
         * @param cy The normalized y-coordinate of the center of the bounding box (0.0 to 1.0).
         * @param w The normalized width of the bounding box (0.0 to 1.0).
         * @param h The normalized height of the bounding box (0.0 to 1.0).
         * @param padInfo The [PadInfo] containing scaling and padding metadata used to map
         * coordinates back to the original image dimensions.
         * @return A [Rect] representing the bounding box in the original image's pixel coordinates.
         */
        fun fromYolo(
            cx: Float,
            cy: Float,
            w: Float,
            h: Float,
            padInfo: PadInfo
        ): Rect {
            return fromNormalized(
                ymin = cy - (h / 2f),
                xmin = cx - (w / 2f),
                ymax = cy + (h / 2f),
                xmax = cx + (w / 2f),
                padInfo = padInfo
            )
        }

        /**
         * Creates a [Rect] from coordinates in COCO format (x-min, y-min, width, height).
         *
         * This method converts normalized COCO coordinates back to the original image coordinate
         * space by reversing the scaling and padding defined in [padInfo].
         *
         * @param x The normalized x-coordinate of the top-left corner (0.0 to 1.0).
         * @param y The normalized y-coordinate of the top-left corner (0.0 to 1.0).
         * @param w The normalized width of the bounding box (0.0 to 1.0).
         * @param h The normalized height of the bounding box (0.0 to 1.0).
         * @param padInfo The scaling and padding information used to map normalized coordinates to the original image.
         * @return A [Rect] representing the bounding box in pixels relative to the original image.
         */
        fun fromCoco(
            x: Float,
            y: Float,
            w: Float,
            h: Float,
            padInfo: PadInfo
        ): Rect {
            return fromNormalized(
                ymin = y,
                xmin = x,
                ymax = y + h,
                xmax = x + w,
                padInfo = padInfo
            )
        }
    }
}

/**
 * Calculates the Intersection over Union (IoU) between this rectangle and another.
 *
 * IoU is a measure of the overlap between two bounding boxes, calculated as the area of
 * the intersection divided by the area of the union. The result ranges from 0.0
 * (no overlap) to 1.0 (perfect overlap).
 *
 * @param other The other rectangle to calculate the overlap with.
 * @return The IoU value as a [Float] between 0.0 and 1.0.
 */
fun Rect.intersectionOverUnion(other: Rect): Float {
    val iLeft = maxOf(this.left, other.left)
    val iTop = maxOf(this.top, other.top)
    val iRight = minOf(this.right, other.right)
    val iBottom = minOf(this.bottom, other.bottom)

    val iArea = maxOf(0, iRight - iLeft) * maxOf(0, iBottom - iTop)
    if (iArea == 0) return 0f

    val unionArea = this.area + other.area - iArea
    return iArea.toFloat() / unionArea.toFloat()
}

fun Rect.scaleForContainer(
    originalContainerWidth: Int,
    originalContainerHeight: Int,
    containerWidth: Float,
    containerHeight: Float,
    isCrop: Boolean = true
): Rect {
    if (originalContainerWidth == 0 || originalContainerHeight == 0) {
        return Rect(0, 0, 0, 0)
    }
    val scaleX = containerWidth / originalContainerWidth
    val scaleY = containerHeight / originalContainerHeight

    val scale = if (isCrop) maxOf(scaleX, scaleY) else minOf(scaleX, scaleY)

    val scaledImageWidth = originalContainerWidth * scale
    val scaledImageHeight = originalContainerHeight * scale

    val offsetX = (containerWidth - scaledImageWidth) / 2f
    val offsetY = (containerHeight - scaledImageHeight) / 2f

    return Rect(
        left = (this.left * scale + offsetX).roundToInt(),
        top = (this.top * scale + offsetY).roundToInt(),
        right = (this.right * scale + offsetX).roundToInt(),
        bottom = (this.bottom * scale + offsetY).roundToInt()
    )
}