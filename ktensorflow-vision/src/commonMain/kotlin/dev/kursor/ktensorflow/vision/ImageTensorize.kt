package dev.kursor.ktensorflow.vision

import dev.kursor.ktensorflow.tensor.TensorDataType
import dev.kursor.ktensorflow.tensor.strides

/**
 * Converts this [Image] into an [ImageTensor] with the specified type [T].
 *
 * The data type is automatically inferred from the type parameter [T].
 * Supported types include [Float], [Int], [Long], and [UByte].
 *
 * @param T The numeric type of the tensor elements.
 * @param layout The memory layout of the resulting tensor (defaults to [ImageTensorLayout.NHWC]).
 * @param pixelFormat The pixel format of the resulting image tensor (defaults to [Image.pixelFormat]).
 * @return An [ImageTensor] containing the pixel data of this image.
 */
inline fun <reified T : Any> Image.tensorize(
    layout: ImageTensorLayout = ImageTensorLayout.NHWC,
    pixelFormat: PixelFormat = this.pixelFormat
) = tensorize(
    dataType = TensorDataType.of<T>(),
    layout = layout,
    pixelFormat = pixelFormat
)

/**
 * Converts this [Image] into an [ImageTensor] with the specified data type and layout.
 *
 * This function iterates through the image pixels and maps each color channel value
 * to the target tensor type [T]. The resulting tensor will have a shape based on the
 * image dimensions and the number of channels defined by the [PixelFormat].
 *
 * @param T The desired primitive type for the tensor elements (e.g., [Float], [Int], [UByte]).
 * @param layout The memory layout of the resulting tensor (defaults to [ImageTensorLayout.NHWC]).
 * @param pixelFormat The pixel format of the resulting image tensor (defaults to [Image.pixelFormat]).
 * @return An [ImageTensor] containing the pixel data of this image.
 * @throws IllegalArgumentException If the provided [dataType] is not supported.
 */
fun <T : Any> Image.tensorize(
    dataType: TensorDataType<T>,
    layout: ImageTensorLayout = ImageTensorLayout.NHWC,
    pixelFormat: PixelFormat = this.pixelFormat
): ImageTensor<T> {
    val tensor = ImageTensor(
        width = width,
        height = height,
        dataType = dataType,
        pixelFormat = pixelFormat,
        layout = layout
    )

    writeImage(
        tensor = tensor,
        pixels = getPixels(),
        offsets = ImageOffsets(tensor),
        batchIndex = 0,
        pixelFormat = pixelFormat,
        convert = dataType.converter
    )

    return tensor
}

/**
 * Converts a list of [Image] objects into a single batched [ImageTensor] with the specified type [T].
 *
 * The data type is automatically inferred from the type parameter [T].
 * All images in the list must have the same dimensions (width and height).
 *
 * @param T The numeric type of the tensor elements (e.g., [Float], [Int], [UByte]).
 * @param layout The memory layout of the resulting tensor (defaults to [ImageTensorLayout.NHWC]).
 * @param pixelFormat The pixel format of the resulting image tensor (defaults to [Image.pixelFormat]).
 * @return An [ImageTensor] containing the pixel data of all images in the batch.
 * @throws IllegalArgumentException If the list is empty or images have mismatched dimensions.
 */
inline fun <reified T : Any> List<Image>.tensorizeBatch(
    layout: ImageTensorLayout = ImageTensorLayout.NHWC,
    pixelFormat: PixelFormat = this.firstOrNull()?.pixelFormat ?: PixelFormat.ARGB
): ImageTensor<T> = tensorizeBatch(
    dataType = TensorDataType.of<T>(),
    layout = layout,
    pixelFormat = pixelFormat
)

/**
 * Converts a list of [Image] objects into a single batched [ImageTensor] with the specified data type and layout.
 *
 * All images in the list must have identical dimensions (width and height). The resulting tensor
 * will have a shape of `[batchSize, height, width, channels]` for [ImageTensorLayout.NHWC]
 * or `[batchSize, channels, height, width]` for [ImageTensorLayout.NCHW].
 *
 * @param T The desired primitive type for the tensor data.
 * @param dataType The [TensorDataType] representing the type [T].
 * @param layout The memory layout of the resulting tensor (defaults to [ImageTensorLayout.NHWC]).
 * @param pixelFormat The pixel format of the resulting image tensor (defaults to [Image.pixelFormat]).
 * @return An [ImageTensor] containing the batched pixel data from all images in the list.
 * @throws IllegalArgumentException If the list is empty or if images have inconsistent dimensions.
 */
fun <T : Any> List<Image>.tensorizeBatch(
    dataType: TensorDataType<T>,
    layout: ImageTensorLayout = ImageTensorLayout.NHWC,
    pixelFormat: PixelFormat = this.firstOrNull()?.pixelFormat ?: PixelFormat.ARGB
): ImageTensor<T> {
    require(isNotEmpty()) { "Empty image batch" }

    val first = first()

    val tensor = ImageTensor(
        width = first.width,
        height = first.height,
        dataType = dataType,
        pixelFormat = pixelFormat,
        layout = layout,
        batchSize = size
    )

    val offsets = ImageOffsets(tensor)
    val convert = dataType.converter

    forEachIndexed { batchIndex, image ->
        require(image.width == first.width && image.height == first.height) {
            "All images must have the same size"
        }

        writeImage(
            tensor = tensor,
            pixels = image.getPixels(),
            offsets = offsets,
            batchIndex = batchIndex,
            pixelFormat = pixelFormat,
            convert = convert
        )
    }

    return tensor
}

/**
 * Converts this [Image] into an [ImageTensor] of [Float] values with optional normalization.
 *
 * This function is specifically designed for machine learning workflows where pixel values
 * need to be scaled or normalized (e.g., to a range of [0, 1] or [-1, 1]). It applies the
 * formula `(value - mean) / std` to each channel during the conversion process.
 *
 * @param layout The memory layout of the resulting tensor (defaults to [ImageTensorLayout.NHWC]).
 * @param pixelFormat The pixel format of the resulting image tensor (defaults to [Image.pixelFormat]).
 * @param normalization The [Normalization] parameters (mean and standard deviation) to apply
 * to the pixel values (defaults to [Normalization.None], which performs no scaling).
 * @return An [ImageTensor] containing the normalized floating-point pixel data.
 */
fun Image.tensorizeFloat(
    layout: ImageTensorLayout = ImageTensorLayout.NHWC,
    pixelFormat: PixelFormat = this.pixelFormat,
    normalization: Normalization = Normalization.None
): ImageTensor<Float> {
    val tensor = ImageTensor(
        width = width,
        height = height,
        dataType = TensorDataType.Float32,
        pixelFormat = pixelFormat,
        layout = layout
    )

    writeImageFloat(
        tensor = tensor,
        pixels = getPixels(),
        offsets = ImageOffsets(tensor),
        batchIndex = 0,
        pixelFormat = pixelFormat,
        normalization = normalization
    )

    return tensor
}

/**
 * Converts a list of [Image] objects into a single batched [ImageTensor] of type [Float]
 * while applying the specified [Normalization].
 *
 * All images in the list must have identical dimensions (width and height).
 *
 * @param layout The memory layout of the resulting tensor (defaults to [ImageTensorLayout.NHWC]).
 * @param pixelFormat The pixel format of the resulting image tensor (defaults to [Image.pixelFormat]).
 * @param normalization The normalization parameters to apply to each pixel.
 * @return An [ImageTensor] of type [Float] containing the normalized batched pixel data.
 * @throws IllegalArgumentException If the list is empty or if images have inconsistent dimensions.
 */
fun List<Image>.tensorizeBatchFloat(
    layout: ImageTensorLayout = ImageTensorLayout.NHWC,
    pixelFormat: PixelFormat = this.firstOrNull()?.pixelFormat ?: PixelFormat.ARGB,
    normalization: Normalization = Normalization.None
): ImageTensor<Float> {
    require(isNotEmpty()) { "Empty image batch" }

    val first = first()

    val tensor = ImageTensor(
        width = first.width,
        height = first.height,
        dataType = TensorDataType.Float32,
        pixelFormat = pixelFormat,
        layout = layout,
        batchSize = size
    )

    val offsets = ImageOffsets(tensor)

    forEachIndexed { batchIndex, image ->
        require(image.width == first.width && image.height == first.height) {
            "All images must have the same size"
        }

        writeImageFloat(
            tensor = tensor,
            pixels = image.getPixels(),
            offsets = offsets,
            batchIndex = batchIndex,
            pixelFormat = pixelFormat,
            normalization = normalization
        )
    }

    return tensor
}

/**
 * Converts this [ImageTensor] of type [Float] back into an [Image].
 *
 * This function reverses the tensorization process by applying "denormalization"
 * (multiplying by the standard deviation and adding the mean) and clipping the
 * resulting values to the valid color range (0-255).
 *
 * If the tensor is batched, you can specify which image to extract using [batchIndex].
 *
 * @param normalization The [Normalization] parameters used to reverse scaling/shifting
 * applied during the initial tensorization. Defaults to [Normalization.None].
 */
fun ImageTensor<Float>.toImage(
    normalization: Normalization = Normalization.None,
    batchIndex: Int = 0
): Image {
    val pixels = IntArray(width * height)
    val offsets = ImageOffsets(this)

    when (val format = pixelFormat) {
        PixelFormat.Grayscale -> readGrayscale(this, pixels, offsets, batchIndex, normalization)
        is PixelFormat.RGB -> readRgb(this, pixels, offsets, batchIndex, format, normalization)
        is PixelFormat.RGBA -> readRgba(this, pixels, offsets, batchIndex, format, normalization)
    }

    return Image(width, height, pixelFormat, pixels)
}

/**
 * Смещения элементов тензора изображения, посчитанные один раз на всю тензоризацию.
 *
 * Обход идёт построчно и по столбцам с наращиванием смещения, поэтому позиция пикселя
 * не пересчитывается по страйдам на каждое обращение к каналу, а деление и остаток от
 * деления для восстановления координат из плоского индекса вообще не нужны. Так один
 * и тот же код одинаково быстро работает для любого [ImageTensorLayout] - и для NHWC,
 * и для NCHW, и для пользовательского, - а также для любого элемента батча.
 */
private class ImageOffsets(tensor: ImageTensor<*>) {

    /** Расстояние между соседними изображениями батча. */
    val batch: Int

    /** Расстояние между соседними строками изображения. */
    val row: Int

    /** Расстояние между соседними пикселями строки. */
    val column: Int

    /** Расстояние между соседними каналами одного пикселя. */
    val channel: Int

    val width: Int = tensor.width

    val height: Int = tensor.height

    init {
        val strides = tensor.shape.strides()
        val layout = tensor.layout

        batch = strides[layout.nIndex]
        row = strides[layout.hIndex]
        column = strides[layout.wIndex]
        channel = strides[layout.cIndex]
    }
}

// Каждый формат пикселей вынесен в отдельную небольшую функцию намеренно: ART не оптимизирует
// крупные методы, а один when со всеми вариантами внутри делал тензоризацию именно таким методом.

private fun <T : Any> writeImage(
    tensor: ImageTensor<T>,
    pixels: IntArray,
    offsets: ImageOffsets,
    batchIndex: Int,
    pixelFormat: PixelFormat,
    convert: (Int) -> T
) {
    when (pixelFormat) {
        PixelFormat.Grayscale -> writeGrayscale(tensor, pixels, offsets, batchIndex, convert)
        is PixelFormat.RGB -> writeRgb(tensor, pixels, offsets, batchIndex, pixelFormat, convert)
        is PixelFormat.RGBA -> writeRgba(tensor, pixels, offsets, batchIndex, pixelFormat, convert)
    }
}

private fun <T : Any> writeGrayscale(
    tensor: ImageTensor<T>,
    pixels: IntArray,
    offsets: ImageOffsets,
    batchIndex: Int,
    convert: (Int) -> T
) {
    val width = offsets.width
    val height = offsets.height
    val rowStride = offsets.row
    val columnStride = offsets.column

    var idx = 0
    var rowBase = offsets.batch * batchIndex
    for (h in 0 until height) {
        var base = rowBase
        for (w in 0 until width) {
            tensor.setFlat(base, convert(pixels[idx++] and 0xFF))
            base += columnStride
        }
        rowBase += rowStride
    }
}

private fun <T : Any> writeRgb(
    tensor: ImageTensor<T>,
    pixels: IntArray,
    offsets: ImageOffsets,
    batchIndex: Int,
    pixelFormat: PixelFormat.RGB,
    convert: (Int) -> T
) {
    val width = offsets.width
    val height = offsets.height
    val rowStride = offsets.row
    val columnStride = offsets.column
    val r = pixelFormat.rIndex * offsets.channel
    val g = pixelFormat.gIndex * offsets.channel
    val b = pixelFormat.bIndex * offsets.channel

    var idx = 0
    var rowBase = offsets.batch * batchIndex
    for (h in 0 until height) {
        var base = rowBase
        for (w in 0 until width) {
            val p = pixels[idx++]
            tensor.setFlat(base + r, convert((p shr 16) and 0xFF))
            tensor.setFlat(base + g, convert((p shr 8) and 0xFF))
            tensor.setFlat(base + b, convert(p and 0xFF))
            base += columnStride
        }
        rowBase += rowStride
    }
}

private fun <T : Any> writeRgba(
    tensor: ImageTensor<T>,
    pixels: IntArray,
    offsets: ImageOffsets,
    batchIndex: Int,
    pixelFormat: PixelFormat.RGBA,
    convert: (Int) -> T
) {
    val width = offsets.width
    val height = offsets.height
    val rowStride = offsets.row
    val columnStride = offsets.column
    val r = pixelFormat.rIndex * offsets.channel
    val g = pixelFormat.gIndex * offsets.channel
    val b = pixelFormat.bIndex * offsets.channel
    val a = pixelFormat.aIndex * offsets.channel

    var idx = 0
    var rowBase = offsets.batch * batchIndex
    for (h in 0 until height) {
        var base = rowBase
        for (w in 0 until width) {
            val p = pixels[idx++]
            tensor.setFlat(base + r, convert((p shr 16) and 0xFF))
            tensor.setFlat(base + g, convert((p shr 8) and 0xFF))
            tensor.setFlat(base + b, convert(p and 0xFF))
            tensor.setFlat(base + a, convert((p shr 24) and 0xFF))
            base += columnStride
        }
        rowBase += rowStride
    }
}

// Float-вариант повторяет структуру обобщённого, но применяет нормализацию арифметикой
// на месте: это самый горячий путь библиотеки, и вызов конвертера на каждый канал здесь
// стоил бы дороже самой нормализации.

private fun writeImageFloat(
    tensor: ImageTensor<Float>,
    pixels: IntArray,
    offsets: ImageOffsets,
    batchIndex: Int,
    pixelFormat: PixelFormat,
    normalization: Normalization
) {
    when (pixelFormat) {
        PixelFormat.Grayscale ->
            writeGrayscaleFloat(tensor, pixels, offsets, batchIndex, normalization)

        is PixelFormat.RGB ->
            writeRgbFloat(tensor, pixels, offsets, batchIndex, pixelFormat, normalization)

        is PixelFormat.RGBA ->
            writeRgbaFloat(tensor, pixels, offsets, batchIndex, pixelFormat, normalization)
    }
}

private fun writeGrayscaleFloat(
    tensor: ImageTensor<Float>,
    pixels: IntArray,
    offsets: ImageOffsets,
    batchIndex: Int,
    normalization: Normalization
) {
    val width = offsets.width
    val height = offsets.height
    val rowStride = offsets.row
    val columnStride = offsets.column
    val mean = normalization.meanR
    val std = normalization.stdR

    var idx = 0
    var rowBase = offsets.batch * batchIndex
    for (h in 0 until height) {
        var base = rowBase
        for (w in 0 until width) {
            tensor.setFlat(base, ((pixels[idx++] and 0xFF).toFloat() - mean) / std)
            base += columnStride
        }
        rowBase += rowStride
    }
}

private fun writeRgbFloat(
    tensor: ImageTensor<Float>,
    pixels: IntArray,
    offsets: ImageOffsets,
    batchIndex: Int,
    pixelFormat: PixelFormat.RGB,
    normalization: Normalization
) {
    val width = offsets.width
    val height = offsets.height
    val rowStride = offsets.row
    val columnStride = offsets.column
    val r = pixelFormat.rIndex * offsets.channel
    val g = pixelFormat.gIndex * offsets.channel
    val b = pixelFormat.bIndex * offsets.channel
    val meanR = normalization.meanR
    val meanG = normalization.meanG
    val meanB = normalization.meanB
    val stdR = normalization.stdR
    val stdG = normalization.stdG
    val stdB = normalization.stdB

    var idx = 0
    var rowBase = offsets.batch * batchIndex
    for (h in 0 until height) {
        var base = rowBase
        for (w in 0 until width) {
            val p = pixels[idx++]
            tensor.setFlat(base + r, (((p shr 16) and 0xFF).toFloat() - meanR) / stdR)
            tensor.setFlat(base + g, (((p shr 8) and 0xFF).toFloat() - meanG) / stdG)
            tensor.setFlat(base + b, ((p and 0xFF).toFloat() - meanB) / stdB)
            base += columnStride
        }
        rowBase += rowStride
    }
}

private fun writeRgbaFloat(
    tensor: ImageTensor<Float>,
    pixels: IntArray,
    offsets: ImageOffsets,
    batchIndex: Int,
    pixelFormat: PixelFormat.RGBA,
    normalization: Normalization
) {
    val width = offsets.width
    val height = offsets.height
    val rowStride = offsets.row
    val columnStride = offsets.column
    val r = pixelFormat.rIndex * offsets.channel
    val g = pixelFormat.gIndex * offsets.channel
    val b = pixelFormat.bIndex * offsets.channel
    val a = pixelFormat.aIndex * offsets.channel
    val meanR = normalization.meanR
    val meanG = normalization.meanG
    val meanB = normalization.meanB
    val meanA = normalization.meanA
    val stdR = normalization.stdR
    val stdG = normalization.stdG
    val stdB = normalization.stdB
    val stdA = normalization.stdA

    var idx = 0
    var rowBase = offsets.batch * batchIndex
    for (h in 0 until height) {
        var base = rowBase
        for (w in 0 until width) {
            val p = pixels[idx++]
            tensor.setFlat(base + r, (((p shr 16) and 0xFF).toFloat() - meanR) / stdR)
            tensor.setFlat(base + g, (((p shr 8) and 0xFF).toFloat() - meanG) / stdG)
            tensor.setFlat(base + b, ((p and 0xFF).toFloat() - meanB) / stdB)
            tensor.setFlat(base + a, (((p shr 24) and 0xFF).toFloat() - meanA) / stdA)
            base += columnStride
        }
        rowBase += rowStride
    }
}

// Обратное преобразование тензора в изображение устроено симметрично записи.

private fun readGrayscale(
    tensor: ImageTensor<Float>,
    pixels: IntArray,
    offsets: ImageOffsets,
    batchIndex: Int,
    normalization: Normalization
) {
    val width = offsets.width
    val height = offsets.height
    val rowStride = offsets.row
    val columnStride = offsets.column
    val mean = normalization.meanR
    val std = normalization.stdR

    var idx = 0
    var rowBase = offsets.batch * batchIndex
    for (h in 0 until height) {
        var base = rowBase
        for (w in 0 until width) {
            val v = (tensor.getFlat(base) * std + mean).toInt().coerceIn(0, 255)
            pixels[idx++] = (0xFF shl 24) or (v shl 16) or (v shl 8) or v
            base += columnStride
        }
        rowBase += rowStride
    }
}

private fun readRgb(
    tensor: ImageTensor<Float>,
    pixels: IntArray,
    offsets: ImageOffsets,
    batchIndex: Int,
    pixelFormat: PixelFormat.RGB,
    normalization: Normalization
) {
    val width = offsets.width
    val height = offsets.height
    val rowStride = offsets.row
    val columnStride = offsets.column
    val rOffset = pixelFormat.rIndex * offsets.channel
    val gOffset = pixelFormat.gIndex * offsets.channel
    val bOffset = pixelFormat.bIndex * offsets.channel
    val meanR = normalization.meanR
    val meanG = normalization.meanG
    val meanB = normalization.meanB
    val stdR = normalization.stdR
    val stdG = normalization.stdG
    val stdB = normalization.stdB

    var idx = 0
    var rowBase = offsets.batch * batchIndex
    for (h in 0 until height) {
        var base = rowBase
        for (w in 0 until width) {
            val r = (tensor.getFlat(base + rOffset) * stdR + meanR).toInt().coerceIn(0, 255)
            val g = (tensor.getFlat(base + gOffset) * stdG + meanG).toInt().coerceIn(0, 255)
            val b = (tensor.getFlat(base + bOffset) * stdB + meanB).toInt().coerceIn(0, 255)
            pixels[idx++] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            base += columnStride
        }
        rowBase += rowStride
    }
}

private fun readRgba(
    tensor: ImageTensor<Float>,
    pixels: IntArray,
    offsets: ImageOffsets,
    batchIndex: Int,
    pixelFormat: PixelFormat.RGBA,
    normalization: Normalization
) {
    val width = offsets.width
    val height = offsets.height
    val rowStride = offsets.row
    val columnStride = offsets.column
    val rOffset = pixelFormat.rIndex * offsets.channel
    val gOffset = pixelFormat.gIndex * offsets.channel
    val bOffset = pixelFormat.bIndex * offsets.channel
    val aOffset = pixelFormat.aIndex * offsets.channel
    val meanR = normalization.meanR
    val meanG = normalization.meanG
    val meanB = normalization.meanB
    val meanA = normalization.meanA
    val stdR = normalization.stdR
    val stdG = normalization.stdG
    val stdB = normalization.stdB
    val stdA = normalization.stdA

    var idx = 0
    var rowBase = offsets.batch * batchIndex
    for (h in 0 until height) {
        var base = rowBase
        for (w in 0 until width) {
            val r = (tensor.getFlat(base + rOffset) * stdR + meanR).toInt().coerceIn(0, 255)
            val g = (tensor.getFlat(base + gOffset) * stdG + meanG).toInt().coerceIn(0, 255)
            val b = (tensor.getFlat(base + bOffset) * stdB + meanB).toInt().coerceIn(0, 255)
            val a = (tensor.getFlat(base + aOffset) * stdA + meanA).toInt().coerceIn(0, 255)
            pixels[idx++] = (a shl 24) or (r shl 16) or (g shl 8) or b
            base += columnStride
        }
        rowBase += rowStride
    }
}

@Suppress("UNCHECKED_CAST")
private val <T : Any> TensorDataType<T>.converter: (Int) -> T
    get() = when (this) {
        TensorDataType.Float32 -> { v: Int -> v.toFloat() as T }
        TensorDataType.Int32 -> { v: Int -> v as T }
        TensorDataType.Int64 -> { v: Int -> v.toLong() as T }
        TensorDataType.UInt8 -> { v: Int -> v.toUByte() as T }
    }
