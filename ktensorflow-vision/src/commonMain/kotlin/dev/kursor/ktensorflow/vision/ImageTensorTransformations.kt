package dev.kursor.ktensorflow.vision

import dev.kursor.ktensorflow.tensor.Tensor
import dev.kursor.ktensorflow.tensor.TensorDataType
import dev.kursor.ktensorflow.tensor.readFloat
import dev.kursor.ktensorflow.tensor.readUByte
import dev.kursor.ktensorflow.tensor.slice
import dev.kursor.ktensorflow.tensor.writeFloat
import kotlin.jvm.JvmName

/**
 * Converts the [ImageTensor] to a grayscale representation using weighted channel summation.
 *
 * This method calculates the luminance of each pixel by applying the provided weights to the
 * red, green, and blue channels. The alpha channel of an RGBA source is dropped: a grayscale
 * tensor has a single channel and no transparency. If the source is already in
 * [PixelFormat.Grayscale], the original tensor is returned.
 *
 * Default weights follow the ITU-R 601 standard for luma, the same as [Image.grayscale] and the
 * [UByte] overload:
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
    // Альфа в яркость не входит: раньше RGBA-версия умножала яркость на неё, и для обычного
    // тензора без нормализации (альфа = 255) яркость получалась в 255 раз больше
    val channels = pixelFormat.rgbIndices() ?: return this

    val output = Tensor(
        dataType,
        TensorShape(n = batch, h = height, w = width, c = PixelFormat.Grayscale.channels, layout = layout)
    )
    val result = ImageTensor(output, PixelFormat.Grayscale, layout)

    val src = ImageOffsets(this)
    val r = channels.r * src.channel
    val g = channels.g * src.channel
    val b = channels.b * src.channel

    // Напрямую в байтах: getFlat и setFlat упаковывают каждое значение
    val source = toPhysical().data
    val target = output.data
    forEachPixel(src, ImageOffsets(result), batch) { from, to ->
        target.writeFloat(
            to,
            rWeight * source.readFloat(from + r) + gWeight * source.readFloat(from + g) + bWeight * source.readFloat(from + b)
        )
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
 * The alpha channel of an RGBA source is dropped, as in the [Float] overload.
 *
 * The batch size and layout of the source are preserved.
 *
 * @return A new [ImageTensor] with [PixelFormat.Grayscale] containing the calculated luminance.
 */
@JvmName("grayscaleUByte")
fun ImageTensor<UByte>.grayscale(): ImageTensor<UByte> {
    val channels = pixelFormat.rgbIndices() ?: return this

    val output = Tensor(
        TensorDataType.UInt8,
        TensorShape(n = batch, h = height, w = width, c = PixelFormat.Grayscale.channels, layout = layout)
    )
    val result = ImageTensor(output, PixelFormat.Grayscale, layout)

    val src = ImageOffsets(this)
    val r = channels.r * src.channel
    val g = channels.g * src.channel
    val b = channels.b * src.channel

    val source = toPhysical().data
    val target = output.data
    forEachPixel(src, ImageOffsets(result), batch) { from, to ->
        val red = source.readUByte(from + r).toInt()
        val green = source.readUByte(from + g).toInt()
        val blue = source.readUByte(from + b).toInt()
        target[to] = ((red * 77 + green * 150 + blue * 29) shr 8).toByte()
    }

    return result
}

/**
 * Resizes the given image tensor to the new specified dimensions using Bilinear Interpolation.
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

    val output = Tensor(
        dataType,
        TensorShape(n = batch, h = newHeight, w = newWidth, c = pixelFormat.channels, layout = layout)
    )
    val result = ImageTensor(output, pixelFormat, layout)

    val src = ImageOffsets(this)
    val dst = ImageOffsets(result)

    // Отношение (old - 1) / (new - 1) выравнивает границы: крайние пиксели исходника
    // попадают ровно в крайние пиксели результата. Для размера 1 делителя нет, и тогда
    // берётся единственный доступный пиксель, иначе получилась бы Infinity и NaN.
    val xRatio = if (newWidth > 1) (width - 1).toFloat() / (newWidth - 1) else 0f
    val yRatio = if (newHeight > 1) (height - 1).toFloat() / (newHeight - 1) else 0f

    // Столбцы исходника и веса интерполяции от строки не зависят - считаются один раз
    val left = IntArray(newWidth)
    val right = IntArray(newWidth)
    val dx = FloatArray(newWidth)
    for (nx in 0 until newWidth) {
        val ox = nx * xRatio
        val x1 = ox.toInt()
        left[nx] = x1 * src.column
        right[nx] = (x1 + 1).coerceAtMost(width - 1) * src.column
        dx[nx] = ox - x1
    }
    val srcChannel = IntArray(channels) { it * src.channel }
    val dstChannel = IntArray(channels) { it * dst.channel }
    // Напрямую в байтах: getFlat и setFlat упаковывают каждое значение
    val source = toPhysical().data
    val target = output.data

    for (n in 0 until batch) {
        var dstRow = dst.batch * n
        for (ny in 0 until newHeight) {
            val oy = ny * yRatio
            val y1 = oy.toInt()
            val dy = oy - y1
            val top = src.batch * n + y1 * src.row
            val bottom = src.batch * n + (y1 + 1).coerceAtMost(height - 1) * src.row

            var to = dstRow
            for (nx in 0 until newWidth) {
                val fx = dx[nx]
                for (c in 0 until channels) {
                    val co = srcChannel[c]
                    val upper = source.readFloat(top + left[nx] + co) * (1f - fx) +
                        source.readFloat(top + right[nx] + co) * fx
                    val lower = source.readFloat(bottom + left[nx] + co) * (1f - fx) +
                        source.readFloat(bottom + right[nx] + co) * fx
                    // Без обрезки диапазона: интерполяция не вправе менять шкалу тензора
                    target.writeFloat(to + dstChannel[c], upper * (1f - dy) + lower * dy)
                }
                to += dst.column
            }
            dstRow += dst.row
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

    val output = Tensor(
        TensorDataType.UInt8,
        TensorShape(n = batch, h = newHeight, w = newWidth, c = pixelFormat.channels, layout = layout)
    )
    val result = ImageTensor(output, pixelFormat, layout)

    val src = ImageOffsets(this)
    val dst = ImageOffsets(result)
    val xRatio = width.toFloat() / newWidth
    val yRatio = height.toFloat() / newHeight

    // Для UInt8 - ближайший сосед: билинейная интерполяция потребовала бы перевода во float.
    // Столбец исходника от строки не зависит и считается один раз.
    val column = IntArray(newWidth) { (it * xRatio).toInt() * src.column }
    val srcChannel = IntArray(channels) { it * src.channel }
    val dstChannel = IntArray(channels) { it * dst.channel }
    // Байт UInt8 копируется как есть, без упаковки в getFlat и setFlat
    val source = toPhysical().data
    val target = output.data

    for (n in 0 until batch) {
        var dstRow = dst.batch * n
        for (h in 0 until newHeight) {
            val srcRow = src.batch * n + (h * yRatio).toInt() * src.row
            var to = dstRow
            for (w in 0 until newWidth) {
                val from = srcRow + column[w]
                for (c in 0 until channels) {
                    target[to + dstChannel[c]] = source[from + srcChannel[c]]
                }
                to += dst.column
            }
            dstRow += dst.row
        }
    }
    return result
}

/** Индексы цветовых каналов формата; null для формата без цвета. */
private class RgbIndices(val r: Int, val g: Int, val b: Int)

private fun PixelFormat.rgbIndices(): RgbIndices? = when (this) {
    PixelFormat.Grayscale -> null
    is PixelFormat.RGB -> RgbIndices(rIndex, gIndex, bIndex)
    is PixelFormat.RGBA -> RgbIndices(rIndex, gIndex, bIndex)
}

/**
 * Обходит все пиксели батча в тензорах одинакового размера, передавая плоские смещения
 * пикселя в [src] и [dst]. Смещения только наращиваются - тот же обход, что у тензоризации.
 */
private inline fun forEachPixel(
    src: ImageOffsets,
    dst: ImageOffsets,
    batch: Int,
    action: (from: Int, to: Int) -> Unit
) {
    for (n in 0 until batch) {
        var srcRow = src.batch * n
        var dstRow = dst.batch * n
        for (h in 0 until src.height) {
            var from = srcRow
            var to = dstRow
            for (w in 0 until src.width) {
                action(from, to)
                from += src.column
                to += dst.column
            }
            srcRow += src.row
            dstRow += dst.row
        }
    }
}

/**
 * Crops the [ImageTensor] to the specified rectangle.
 *
 * @param rect The rectangle to crop the image to.
 * @return A new [ImageTensor] containing the cropped region.
 * @throws IllegalArgumentException if [rect] is empty or not inside the image, as for [Image.crop].
 */
fun <T : Any> ImageTensor<T>.crop(rect: Rect): ImageTensor<T> {
    // Та же проверка, что у Image.crop: пустой прямоугольник раньше давал тензор нулевой
    // ширины, и ошибка всплывала позже - в resize про индекс, в toImage про размер
    require(rect.right > rect.left && rect.bottom > rect.top) { "Crop rect $rect is empty" }
    require(rect.left >= 0 && rect.top >= 0 && rect.right <= width && rect.bottom <= height) {
        "Crop rect $rect is outside the ${width}x$height image tensor"
    }

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
