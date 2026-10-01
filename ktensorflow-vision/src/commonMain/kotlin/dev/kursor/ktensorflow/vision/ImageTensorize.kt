package dev.kursor.ktensorflow.vision

import dev.kursor.ktensorflow.tensor.Tensor
import dev.kursor.ktensorflow.tensor.TensorDataType
import dev.kursor.ktensorflow.tensor.readFloat
import dev.kursor.ktensorflow.tensor.writeFloat
import dev.kursor.ktensorflow.tensor.writeInt
import dev.kursor.ktensorflow.tensor.writeLong

/**
 * Converts this [Image] into an [ImageTensor] with the specified type [T].
 *
 * The data type is automatically inferred from the type parameter [T].
 * Supported types include [Float], [Int], [Long], and [UByte].
 *
 * @param T The numeric type of the tensor elements.
 * @param layout The memory layout of the resulting tensor (defaults to [ImageTensorLayout.NHWC]).
 * @param pixelFormat The pixel format of the resulting image tensor (defaults to [Image.pixelFormat]).
 * [PixelFormat.Grayscale] stores the ITU-R 601 luma of each pixel, so a color image can be
 * tensorized straight into a single-channel tensor.
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
 * [PixelFormat.Grayscale] stores the ITU-R 601 luma of each pixel, so a color image can be
 * tensorized straight into a single-channel tensor.
 * @return An [ImageTensor] containing the pixel data of this image.
 * @throws IllegalArgumentException If the provided [dataType] is not supported.
 */
fun <T : Any> Image.tensorize(
    dataType: TensorDataType<T>,
    layout: ImageTensorLayout = ImageTensorLayout.NHWC,
    pixelFormat: PixelFormat = this.pixelFormat
): ImageTensor<T> {
    val physical = Tensor(
        dataType,
        TensorShape(n = 1, h = height, w = width, c = pixelFormat.channels, layout = layout)
    )
    val tensor = ImageTensor(physical, pixelFormat, layout)

    writeImage(
        data = physical.data,
        dataType = dataType,
        pixels = getPixels(),
        offsets = ImageOffsets(tensor),
        batchIndex = 0,
        pixelFormat = pixelFormat
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
 * [PixelFormat.Grayscale] stores the ITU-R 601 luma of each pixel, so a color image can be
 * tensorized straight into a single-channel tensor.
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
 * [PixelFormat.Grayscale] stores the ITU-R 601 luma of each pixel, so a color image can be
 * tensorized straight into a single-channel tensor.
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

    val physical = Tensor(
        dataType,
        TensorShape(n = size, h = first.height, w = first.width, c = pixelFormat.channels, layout = layout)
    )
    val tensor = ImageTensor(physical, pixelFormat, layout)

    val offsets = ImageOffsets(tensor)
    val data = physical.data
    // Один буфер пикселей на весь батч: размеры у изображений одинаковые
    val pixels = IntArray(first.width * first.height)

    forEachIndexed { batchIndex, image ->
        require(image.width == first.width && image.height == first.height) {
            "All images must have the same size"
        }
        image.getPixels(pixels)

        writeImage(
            data = data,
            dataType = dataType,
            pixels = pixels,
            offsets = offsets,
            batchIndex = batchIndex,
            pixelFormat = pixelFormat
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
 * [PixelFormat.Grayscale] stores the ITU-R 601 luma of each pixel, so a color image can be
 * tensorized straight into a single-channel tensor.
 * @param normalization The [Normalization] parameters (mean and standard deviation) to apply
 * to the pixel values (defaults to [Normalization.None], which performs no scaling).
 * @return An [ImageTensor] containing the normalized floating-point pixel data.
 */
fun Image.tensorizeFloat(
    layout: ImageTensorLayout = ImageTensorLayout.NHWC,
    pixelFormat: PixelFormat = this.pixelFormat,
    normalization: Normalization = Normalization.None
): ImageTensor<Float> {
    val physical = Tensor(
        TensorDataType.Float32,
        TensorShape(n = 1, h = height, w = width, c = pixelFormat.channels, layout = layout)
    )
    val tensor = ImageTensor(physical, pixelFormat, layout)

    writeImageFloat(
        data = physical.data,
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
 * [PixelFormat.Grayscale] stores the ITU-R 601 luma of each pixel, so a color image can be
 * tensorized straight into a single-channel tensor.
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

    val physical = Tensor(
        TensorDataType.Float32,
        TensorShape(n = size, h = first.height, w = first.width, c = pixelFormat.channels, layout = layout)
    )
    val tensor = ImageTensor(physical, pixelFormat, layout)

    val offsets = ImageOffsets(tensor)
    val data = physical.data
    // Один буфер пикселей на весь батч: размеры у изображений одинаковые
    val pixels = IntArray(first.width * first.height)

    forEachIndexed { batchIndex, image ->
        require(image.width == first.width && image.height == first.height) {
            "All images must have the same size"
        }
        image.getPixels(pixels)

        writeImageFloat(
            data = data,
            pixels = pixels,
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
 * @param batchIndex The index of the image to extract from a batched tensor.
 * @throws IllegalArgumentException If [batchIndex] is outside the tensor's batch.
 */
fun ImageTensor<Float>.toImage(
    normalization: Normalization = Normalization.None,
    batchIndex: Int = 0
): Image {
    require(batchIndex in 0 until batch) {
        "batchIndex $batchIndex is out of bounds for a tensor with batch size $batch"
    }

    val pixels = IntArray(width * height)
    val offsets = ImageOffsets(this)
    val data = toPhysical().data

    when (val format = pixelFormat) {
        PixelFormat.Grayscale -> readGrayscale(data, pixels, offsets, batchIndex, normalization)
        is PixelFormat.RGB -> readRgb(data, pixels, offsets, batchIndex, format, normalization)
        is PixelFormat.RGBA -> readRgba(data, pixels, offsets, batchIndex, format, normalization)
    }

    return Image(width, height, pixelFormat, pixels)
}

// Циклы пишут и читают байты тензора напрямую, а не через обобщённые setFlat и getFlat: те
// упаковывают каждое значение в объект и вызываются через интерфейс, и на кадре 640x640 это
// давало 7-17 мс вместо 2 мс (замер release-сборок на обеих платформах).
//
// Правило для всех таких циклов: результат создаётся физическим тензором явно, и запись идёт в
// его data; источник читается через toPhysical().data - у физического тензора это его же байты,
// а view копируется один раз. data берётся до цикла: setFlatUnboxed на каждом элементе на iOS
// в 1,6 раза медленнее, потому что геттер data вызывается через интерфейс.
//
// Каждое сочетание формата и типа данных вынесено в отдельную небольшую функцию намеренно: ART
// не оптимизирует крупные методы, а один when со всеми вариантами внутри делал тензоризацию
// именно таким методом. Сам обход пикселей встраивается в каждую из них.

/**
 * Обходит пиксели одного изображения батча: передаёт номер пикселя в массиве ARGB и плоское
 * смещение его первого канала в тензоре. Смещения только наращиваются.
 */
private inline fun forEachImagePixel(
    offsets: ImageOffsets,
    batchIndex: Int,
    action: (pixel: Int, base: Int) -> Unit
) {
    val width = offsets.width
    val height = offsets.height
    val rowStride = offsets.row
    val columnStride = offsets.column

    var pixel = 0
    var rowBase = offsets.batch * batchIndex
    for (h in 0 until height) {
        var base = rowBase
        for (w in 0 until width) {
            action(pixel++, base)
            base += columnStride
        }
        rowBase += rowStride
    }
}

private fun writeImage(
    data: ByteArray,
    dataType: TensorDataType<*>,
    pixels: IntArray,
    offsets: ImageOffsets,
    batchIndex: Int,
    pixelFormat: PixelFormat
) {
    when (dataType) {
        // Без нормализации (v - 0) / 1 даёт ровно v, поэтому Float идёт путём tensorizeFloat
        TensorDataType.Float32 ->
            writeImageFloat(data, pixels, offsets, batchIndex, pixelFormat, Normalization.None)

        TensorDataType.Int32 -> when (pixelFormat) {
            PixelFormat.Grayscale -> writeGrayscaleInt32(data, pixels, offsets, batchIndex)
            is PixelFormat.RGB -> writeRgbInt32(data, pixels, offsets, batchIndex, pixelFormat)
            is PixelFormat.RGBA -> writeRgbaInt32(data, pixels, offsets, batchIndex, pixelFormat)
        }

        TensorDataType.Int64 -> when (pixelFormat) {
            PixelFormat.Grayscale -> writeGrayscaleInt64(data, pixels, offsets, batchIndex)
            is PixelFormat.RGB -> writeRgbInt64(data, pixels, offsets, batchIndex, pixelFormat)
            is PixelFormat.RGBA -> writeRgbaInt64(data, pixels, offsets, batchIndex, pixelFormat)
        }

        TensorDataType.UInt8 -> when (pixelFormat) {
            PixelFormat.Grayscale -> writeGrayscaleUInt8(data, pixels, offsets, batchIndex)
            is PixelFormat.RGB -> writeRgbUInt8(data, pixels, offsets, batchIndex, pixelFormat)
            is PixelFormat.RGBA -> writeRgbaUInt8(data, pixels, offsets, batchIndex, pixelFormat)
        }
    }
}

/** Пишет яркость каждого пикселя через [put]: тип данных задаёт вызывающая функция. */
private inline fun writeGrayscale(
    pixels: IntArray,
    offsets: ImageOffsets,
    batchIndex: Int,
    put: (index: Int, value: Int) -> Unit
) = forEachImagePixel(offsets, batchIndex) { pixel, base ->
    put(base, luma(pixels[pixel]))
}

/** Пишет каналы RGB каждого пикселя через [put]: тип данных задаёт вызывающая функция. */
private inline fun writeRgb(
    pixels: IntArray,
    offsets: ImageOffsets,
    batchIndex: Int,
    pixelFormat: PixelFormat.RGB,
    put: (index: Int, value: Int) -> Unit
) {
    val r = pixelFormat.rIndex * offsets.channel
    val g = pixelFormat.gIndex * offsets.channel
    val b = pixelFormat.bIndex * offsets.channel

    forEachImagePixel(offsets, batchIndex) { pixel, base ->
        val p = pixels[pixel]
        put(base + r, (p shr 16) and 0xFF)
        put(base + g, (p shr 8) and 0xFF)
        put(base + b, p and 0xFF)
    }
}

/** Пишет каналы RGBA каждого пикселя через [put]: тип данных задаёт вызывающая функция. */
private inline fun writeRgba(
    pixels: IntArray,
    offsets: ImageOffsets,
    batchIndex: Int,
    pixelFormat: PixelFormat.RGBA,
    put: (index: Int, value: Int) -> Unit
) {
    val r = pixelFormat.rIndex * offsets.channel
    val g = pixelFormat.gIndex * offsets.channel
    val b = pixelFormat.bIndex * offsets.channel
    val a = pixelFormat.aIndex * offsets.channel

    forEachImagePixel(offsets, batchIndex) { pixel, base ->
        val p = pixels[pixel]
        put(base + r, (p shr 16) and 0xFF)
        put(base + g, (p shr 8) and 0xFF)
        put(base + b, p and 0xFF)
        put(base + a, (p shr 24) and 0xFF)
    }
}

private fun writeGrayscaleInt32(data: ByteArray, pixels: IntArray, offsets: ImageOffsets, batchIndex: Int) =
    writeGrayscale(pixels, offsets, batchIndex) { index, value -> data.writeInt(index, value) }

private fun writeRgbInt32(
    data: ByteArray,
    pixels: IntArray,
    offsets: ImageOffsets,
    batchIndex: Int,
    pixelFormat: PixelFormat.RGB
) = writeRgb(pixels, offsets, batchIndex, pixelFormat) { index, value -> data.writeInt(index, value) }

private fun writeRgbaInt32(
    data: ByteArray,
    pixels: IntArray,
    offsets: ImageOffsets,
    batchIndex: Int,
    pixelFormat: PixelFormat.RGBA
) = writeRgba(pixels, offsets, batchIndex, pixelFormat) { index, value -> data.writeInt(index, value) }

private fun writeGrayscaleInt64(data: ByteArray, pixels: IntArray, offsets: ImageOffsets, batchIndex: Int) =
    writeGrayscale(pixels, offsets, batchIndex) { index, value -> data.writeLong(index, value.toLong()) }

private fun writeRgbInt64(
    data: ByteArray,
    pixels: IntArray,
    offsets: ImageOffsets,
    batchIndex: Int,
    pixelFormat: PixelFormat.RGB
) = writeRgb(pixels, offsets, batchIndex, pixelFormat) { index, value -> data.writeLong(index, value.toLong()) }

private fun writeRgbaInt64(
    data: ByteArray,
    pixels: IntArray,
    offsets: ImageOffsets,
    batchIndex: Int,
    pixelFormat: PixelFormat.RGBA
) = writeRgba(pixels, offsets, batchIndex, pixelFormat) { index, value -> data.writeLong(index, value.toLong()) }

private fun writeGrayscaleUInt8(data: ByteArray, pixels: IntArray, offsets: ImageOffsets, batchIndex: Int) =
    writeGrayscale(pixels, offsets, batchIndex) { index, value -> data[index] = value.toByte() }

private fun writeRgbUInt8(
    data: ByteArray,
    pixels: IntArray,
    offsets: ImageOffsets,
    batchIndex: Int,
    pixelFormat: PixelFormat.RGB
) = writeRgb(pixels, offsets, batchIndex, pixelFormat) { index, value -> data[index] = value.toByte() }

private fun writeRgbaUInt8(
    data: ByteArray,
    pixels: IntArray,
    offsets: ImageOffsets,
    batchIndex: Int,
    pixelFormat: PixelFormat.RGBA
) = writeRgba(pixels, offsets, batchIndex, pixelFormat) { index, value -> data[index] = value.toByte() }

// Float-вариант применяет нормализацию арифметикой на месте: это самый горячий путь библиотеки.

private fun writeImageFloat(
    data: ByteArray,
    pixels: IntArray,
    offsets: ImageOffsets,
    batchIndex: Int,
    pixelFormat: PixelFormat,
    normalization: Normalization
) {
    when (pixelFormat) {
        PixelFormat.Grayscale ->
            writeGrayscaleFloat(data, pixels, offsets, batchIndex, normalization)

        is PixelFormat.RGB ->
            writeRgbFloat(data, pixels, offsets, batchIndex, pixelFormat, normalization)

        is PixelFormat.RGBA ->
            writeRgbaFloat(data, pixels, offsets, batchIndex, pixelFormat, normalization)
    }
}

private fun writeGrayscaleFloat(
    data: ByteArray,
    pixels: IntArray,
    offsets: ImageOffsets,
    batchIndex: Int,
    normalization: Normalization
) {
    val mean = normalization.meanR
    val std = normalization.stdR

    forEachImagePixel(offsets, batchIndex) { pixel, base ->
        data.writeFloat(base, (luma(pixels[pixel]).toFloat() - mean) / std)
    }
}

private fun writeRgbFloat(
    data: ByteArray,
    pixels: IntArray,
    offsets: ImageOffsets,
    batchIndex: Int,
    pixelFormat: PixelFormat.RGB,
    normalization: Normalization
) {
    val r = pixelFormat.rIndex * offsets.channel
    val g = pixelFormat.gIndex * offsets.channel
    val b = pixelFormat.bIndex * offsets.channel
    val meanR = normalization.meanR
    val meanG = normalization.meanG
    val meanB = normalization.meanB
    val stdR = normalization.stdR
    val stdG = normalization.stdG
    val stdB = normalization.stdB

    forEachImagePixel(offsets, batchIndex) { pixel, base ->
        val p = pixels[pixel]
        data.writeFloat(base + r, (((p shr 16) and 0xFF).toFloat() - meanR) / stdR)
        data.writeFloat(base + g, (((p shr 8) and 0xFF).toFloat() - meanG) / stdG)
        data.writeFloat(base + b, ((p and 0xFF).toFloat() - meanB) / stdB)
    }
}

private fun writeRgbaFloat(
    data: ByteArray,
    pixels: IntArray,
    offsets: ImageOffsets,
    batchIndex: Int,
    pixelFormat: PixelFormat.RGBA,
    normalization: Normalization
) {
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

    forEachImagePixel(offsets, batchIndex) { pixel, base ->
        val p = pixels[pixel]
        data.writeFloat(base + r, (((p shr 16) and 0xFF).toFloat() - meanR) / stdR)
        data.writeFloat(base + g, (((p shr 8) and 0xFF).toFloat() - meanG) / stdG)
        data.writeFloat(base + b, ((p and 0xFF).toFloat() - meanB) / stdB)
        data.writeFloat(base + a, (((p shr 24) and 0xFF).toFloat() - meanA) / stdA)
    }
}

// Обратное преобразование тензора в изображение устроено симметрично записи.

/**
 * Денормализованное значение канала в байт с округлением к ближайшему. Отбрасывание дроби
 * сдвигало на единицу 6 уровней из 256 при tensorize/toImage с Normalization.ImageNet и
 * занижало результат билинейного resize. В отличие от roundToInt не падает на NaN (даёт 0).
 */
@Suppress("NOTHING_TO_INLINE")
private inline fun toChannel(value: Float): Int = (value + 0.5f).toInt().coerceIn(0, 255)

private fun readGrayscale(
    data: ByteArray,
    pixels: IntArray,
    offsets: ImageOffsets,
    batchIndex: Int,
    normalization: Normalization
) {
    val mean = normalization.meanR
    val std = normalization.stdR

    forEachImagePixel(offsets, batchIndex) { pixel, base ->
        val v = toChannel(data.readFloat(base) * std + mean)
        pixels[pixel] = (0xFF shl 24) or (v shl 16) or (v shl 8) or v
    }
}

private fun readRgb(
    data: ByteArray,
    pixels: IntArray,
    offsets: ImageOffsets,
    batchIndex: Int,
    pixelFormat: PixelFormat.RGB,
    normalization: Normalization
) {
    val rOffset = pixelFormat.rIndex * offsets.channel
    val gOffset = pixelFormat.gIndex * offsets.channel
    val bOffset = pixelFormat.bIndex * offsets.channel
    val meanR = normalization.meanR
    val meanG = normalization.meanG
    val meanB = normalization.meanB
    val stdR = normalization.stdR
    val stdG = normalization.stdG
    val stdB = normalization.stdB

    forEachImagePixel(offsets, batchIndex) { pixel, base ->
        val r = toChannel(data.readFloat(base + rOffset) * stdR + meanR)
        val g = toChannel(data.readFloat(base + gOffset) * stdG + meanG)
        val b = toChannel(data.readFloat(base + bOffset) * stdB + meanB)
        pixels[pixel] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
    }
}

private fun readRgba(
    data: ByteArray,
    pixels: IntArray,
    offsets: ImageOffsets,
    batchIndex: Int,
    pixelFormat: PixelFormat.RGBA,
    normalization: Normalization
) {
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

    forEachImagePixel(offsets, batchIndex) { pixel, base ->
        val r = toChannel(data.readFloat(base + rOffset) * stdR + meanR)
        val g = toChannel(data.readFloat(base + gOffset) * stdG + meanG)
        val b = toChannel(data.readFloat(base + bOffset) * stdB + meanB)
        val a = toChannel(data.readFloat(base + aOffset) * stdA + meanA)
        pixels[pixel] = (a shl 24) or (r shl 16) or (g shl 8) or b
    }
}

/**
 * Яркость упакованного ARGB-пикселя по ITU-R 601 в целых числах - та же формула, что у
 * ImageTensor<UByte>.grayscale. Раньше в серый тензор писался младший байт, то есть синий канал:
 * чистый красный давал 0 вместо 76. У серого пикселя (R = G = B) результат прежний, бит в бит,
 * потому что 77 + 150 + 29 = 256.
 */
@Suppress("NOTHING_TO_INLINE")
private inline fun luma(p: Int): Int =
    (((p shr 16) and 0xFF) * 77 + ((p shr 8) and 0xFF) * 150 + (p and 0xFF) * 29) shr 8
