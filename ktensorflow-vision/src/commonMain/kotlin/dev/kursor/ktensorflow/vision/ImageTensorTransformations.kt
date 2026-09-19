package dev.kursor.ktensorflow.vision

import dev.kursor.ktensorflow.tensor.slice
import kotlin.jvm.JvmName

/**
 * Converts the [ImageTensor] to a grayscale representation using weighted channel summation.
 *
 * This method calculates the luminance of each pixel by applying the provided weights to the
 * red, green, and blue channels. If the source is in RGBA format, the resulting luminance
 * is additionally multiplied by the alpha channel value. If the source is already in
 * [PixelFormat.Grayscale], the original tensor is returned.
 *
 * Default weights follow the ITU-R 601 standard for luma:
 * - Red: 0.299
 * - Green: 0.587
 * - Blue: 0.114
 *
 * The batch size and layout of the source are preserved.
 *
 * @param rWeight The weight applied to the red channel.
 * @param gWeight The weight applied to the green channel.
 * @param bWeight The weight applied to the blue channel.
 * @return A new [ImageTensor] with [PixelFormat.Grayscale] containing the calculated luminance.
 */
fun ImageTensor<Float>.grayscale(
    rWeight: Float = 0.299f,
    gWeight: Float = 0.587f,
    bWeight: Float = 0.114f
): ImageTensor<Float> {

    val pf = pixelFormat
    if (pf == PixelFormat.Grayscale) return this

    // Батч и layout обязаны сохраниться: иначе для пачки изображений результат оказывался
    // тензором на одну картинку, а запись по batch-индексу уходила за его пределы.
    val result = ImageTensor(
        width = width,
        height = height,
        dataType = dataType,
        pixelFormat = PixelFormat.Grayscale,
        layout = layout,
        batchSize = batch
    )

    when (pf) {
        PixelFormat.Grayscale -> return this

        is PixelFormat.RGB -> {
            for (n in 0 until batch) {
                for (h in 0 until height) {
                    for (w in 0 until width) {
                        result[n, h, w, 0] = rWeight * this[n, h, w, pf.rIndex] +
                                gWeight * this[n, h, w, pf.gIndex] +
                                bWeight * this[n, h, w, pf.bIndex]
                    }
                }
            }
        }

        is PixelFormat.RGBA -> {
            for (n in 0 until batch) {
                for (h in 0 until height) {
                    for (w in 0 until width) {
                        result[n, h, w, 0] = (rWeight * this[n, h, w, pf.rIndex] +
                                gWeight * this[n, h, w, pf.gIndex] +
                                bWeight * this[n, h, w, pf.bIndex]) *
                                this[n, h, w, pf.aIndex]
                    }
                }
            }
        }
    }

    return result
}

/**
 * Converts the [UByte] based [ImageTensor] to a grayscale representation.
 *
 * This method uses a fixed-point integer approximation of the ITU-R 601 luma weights
 * to perform the conversion efficiently:
 * - Red: ~0.30 (77/256)
 * - Green: ~0.59 (150/256)
 * - Blue: ~0.11 (29/256)
 *
 * If the source is already in [PixelFormat.Grayscale], the original tensor is returned.
 * Note: Unlike the [Float] version, this implementation does not currently account
 * for the alpha channel in RGBA tensors.
 *
 * The batch size and layout of the source are preserved.
 *
 * @return A new [ImageTensor] with [PixelFormat.Grayscale] containing the calculated luminance.
 */
@JvmName("grayscaleUByte")
fun ImageTensor<UByte>.grayscale(): ImageTensor<UByte> {
    val pf = pixelFormat
    if (pf == PixelFormat.Grayscale) return this

    // Батч и layout обязаны сохраниться: иначе для пачки изображений результат оказывался
    // тензором на одну картинку, а запись по batch-индексу уходила за его пределы.
    val result = ImageTensor<UByte>(
        width = width,
        height = height,
        pixelFormat = PixelFormat.Grayscale,
        layout = layout,
        batchSize = batch
    )

    val rIdx = when (pf) {
        is PixelFormat.RGB -> pf.rIndex
        is PixelFormat.RGBA -> pf.rIndex
    }
    val gIdx = when (pf) {
        is PixelFormat.RGB -> pf.gIndex
        is PixelFormat.RGBA -> pf.gIndex
    }
    val bIdx = when (pf) {
        is PixelFormat.RGB -> pf.bIndex
        is PixelFormat.RGBA -> pf.bIndex
    }

    for (n in 0 until batch) {
        for (h in 0 until height) {
            for (w in 0 until width) {
                val r = this[n, h, w, rIdx].toInt() and 0xFF
                val g = this[n, h, w, gIdx].toInt() and 0xFF
                val b = this[n, h, w, bIdx].toInt() and 0xFF

                val gray = (r * 77 + g * 150 + b * 29) shr 8
                result[n, h, w, 0] = gray.toUByte()
            }
        }
    }
    return result
}


/**
 * Resizes the given Picture to the new specified dimensions using Bilinear Interpolation.
 *
 * Bilinear interpolation calculates the value of the new pixel based on a weighted average
 * of the four nearest original pixels, providing a smoother result than nearest-neighbor
 * resizing.
 *
 * Values are interpolated as they are, without clamping to any range, so tensors in 0..255
 * and normalized tensors with negative values survive the resize unchanged in scale.
 * The batch size, layout and pixel format of the source are preserved.
 *
 * @param newWidth The desired width of the resulting image.
 * @param newHeight The desired height of the resulting image.
 * @return A new [ImageTensor] with the resized dimensions and interpolated data,
 * or the original tensor if dimensions are unchanged.
 * @throws IllegalArgumentException If the requested dimensions are not positive.
 */
fun ImageTensor<Float>.resize(newWidth: Int, newHeight: Int): ImageTensor<Float> {
    require(newWidth > 0 && newHeight > 0) { "New dimensions must be positive." }

    if (newWidth == width && newHeight == height) {
        return this // No resize needed
    }

    // Батч, layout и формат пикселей переносятся как есть: ресайз меняет только размер.
    val result = ImageTensor(
        width = newWidth,
        height = newHeight,
        dataType = dataType,
        pixelFormat = pixelFormat,
        layout = layout,
        batchSize = batch
    )

    // Отношение (old - 1) / (new - 1) выравнивает границы: крайние пиксели исходника
    // попадают ровно в крайние пиксели результата. Для размера 1 делителя нет, и тогда
    // берётся единственный доступный пиксель, иначе получилась бы Infinity и NaN.
    val xRatio = if (newWidth > 1) (width - 1).toFloat() / (newWidth - 1) else 0f
    val yRatio = if (newHeight > 1) (height - 1).toFloat() / (newHeight - 1) else 0f

    for (n in 0 until batch) {
        for (ny in 0 until newHeight) {
            // Координаты по вертикали не зависят от столбца, поэтому считаются один раз на строку
            val oy = ny * yRatio
            val y1 = oy.toInt()
            val y2 = (y1 + 1).coerceAtMost(height - 1)
            val dy = oy - y1

            for (nx in 0 until newWidth) {
                val ox = nx * xRatio
                val x1 = ox.toInt()
                val x2 = (x1 + 1).coerceAtMost(width - 1)
                val dx = ox - x1

                for (c in 0 until channels) {
                    // Четыре соседних пикселя исходника вокруг искомой точки
                    val topLeft = this[n, y1, x1, c]
                    val topRight = this[n, y1, x2, c]
                    val bottomLeft = this[n, y2, x1, c]
                    val bottomRight = this[n, y2, x2, c]

                    val top = topLeft * (1f - dx) + topRight * dx
                    val bottom = bottomLeft * (1f - dx) + bottomRight * dx

                    // Без обрезки диапазона: тензор может быть и в 0..255, и нормализованным
                    // в отрицательные значения, а интерполяция не вправе менять его шкалу.
                    result[n, ny, nx, c] = top * (1f - dy) + bottom * dy
                }
            }
        }
    }

    return result
}

/**
 * Resizes the image tensor to the specified dimensions using Nearest Neighbor interpolation.
 *
 * This implementation uses nearest-neighbor scaling for [UByte] tensors to maintain efficiency
 * and avoid the floating-point conversions required for bilinear interpolation.
 *
 * The batch size, layout and pixel format of the source are preserved.
 *
 * @param newWidth The desired width of the resulting image.
 * @param newHeight The desired height of the resulting image.
 * @return A new [ImageTensor] with the specified dimensions, or the original tensor if dimensions are unchanged.
 * @throws IllegalArgumentException If the requested dimensions are not positive.
 */
@JvmName("resizeUByte")
fun ImageTensor<UByte>.resize(newWidth: Int, newHeight: Int): ImageTensor<UByte> {
    require(newWidth > 0 && newHeight > 0) { "New dimensions must be positive." }

    if (newWidth == width && newHeight == height) return this

    // Батч, layout и формат пикселей переносятся как есть: ресайз меняет только размер.
    val result = ImageTensor<UByte>(
        width = newWidth,
        height = newHeight,
        pixelFormat = pixelFormat,
        layout = layout,
        batchSize = batch
    )
    val xRatio = width.toFloat() / newWidth
    val yRatio = height.toFloat() / newHeight

    // For UInt8 use Nearest Neighbor,
    // since bilinear interpolation requires converting to float
    for (n in 0 until batch) {
        for (h in 0 until newHeight) {
            val srcH = (h * yRatio).toInt()
            for (w in 0 until newWidth) {
                val srcW = (w * xRatio).toInt()
                for (c in 0 until channels) {
                    result[n, h, w, c] = this[n, srcH, srcW, c]
                }
            }
        }
    }
    return result
}

/**
 * Crops the [ImageTensor] to the specified rectangle.
 *
 * @param rect The rectangle to crop the image to.
 * @return A new [ImageTensor] containing the cropped region.
 */
fun <T : Any> ImageTensor<T>.crop(rect: Rect): ImageTensor<T> {
    val ranges = Array(4) { index ->
        when (index) {
            layout.nIndex -> 0..<batch
            layout.hIndex -> rect.top..<rect.bottom
            layout.wIndex -> rect.left..<rect.right
            layout.cIndex -> 0..<channels
            else -> error("Unreachable: ImageTensorLayout must map to exactly 4 distinct indices")
        }
    }

    return this
        .slice(ranges)
        .toImageTensor(this.pixelFormat, layout)
}
