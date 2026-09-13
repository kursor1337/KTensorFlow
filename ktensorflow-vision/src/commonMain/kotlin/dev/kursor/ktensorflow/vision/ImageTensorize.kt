package dev.kursor.ktensorflow.vision

import dev.kursor.ktensorflow.tensor.Tensor
import dev.kursor.ktensorflow.tensor.TensorDataType

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

    val pixels = getPixels()
    val total = width * height

    when (pixelFormat) {
        PixelFormat.Grayscale -> {
            for (idx in 0 until total) {
                val w = idx % width
                val h = idx / width
                val v = pixels[idx] and 0xFF

                tensor[0, h, w, 0] = dataType.converter(v)
            }
        }

        is PixelFormat.RGB -> {

            for (idx in 0 until total) {
                val p = pixels[idx]
                val w = idx % width
                val h = idx / width

                tensor[0, h, w, pixelFormat.rIndex] = dataType.converter((p shr 16) and 0xFF)
                tensor[0, h, w, pixelFormat.gIndex] = dataType.converter((p shr 8) and 0xFF)
                tensor[0, h, w, pixelFormat.bIndex] = dataType.converter((p shr 0) and 0xFF)
            }
        }

        is PixelFormat.RGBA -> {

            for (idx in 0 until total) {
                val p = pixels[idx]
                val w = idx % width
                val h = idx / width

                tensor[0, h, w, pixelFormat.rIndex] = dataType.converter((p shr 16) and 0xFF)
                tensor[0, h, w, pixelFormat.gIndex] = dataType.converter((p shr 8) and 0xFF)
                tensor[0, h, w, pixelFormat.bIndex] = dataType.converter((p shr 0) and 0xFF)
                tensor[0, h, w, pixelFormat.aIndex] = dataType.converter((p shr 24) and 0xFF)
            }
        }
    }

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
    val w = first.width
    val h = first.height
    val pf = first.pixelFormat
    val c = pf.channels
    val n = size

    val tensor = ImageTensor(
        tensor = Tensor(
            dataType = dataType,
            shape = TensorShape(n, h, w, c, layout)
        ),
        pixelFormat = pf,
        layout = layout
    )

    forEachIndexed { batchIndex, image ->
        require(image.width == w && image.height == h) {
            "All images must have same size"
        }

        writeImageInto(
            image = image,
            tensor = tensor,
            batchIndex = batchIndex,
            convert = dataType.converter,
            pixelFormat = pixelFormat
        )
    }

    return tensor
}

private inline fun <T : Any> writeImageInto(
    image: Image,
    tensor: ImageTensor<T>,
    batchIndex: Int,
    pixelFormat: PixelFormat,
    convert: (Int) -> T
) {
    val width = image.width
    val height = image.height
    val pixels = image.getPixels()
    val total = width * height

    when (pixelFormat) {

        PixelFormat.Grayscale -> {
            for (idx in 0 until total) {
                val w = idx % width
                val h = idx / width
                val v = pixels[idx] and 0xFF

                tensor[batchIndex, 0, h, w] = convert(v)
            }
        }

        is PixelFormat.RGB -> {

            for (idx in 0 until total) {
                val p = pixels[idx]
                val w = idx % width
                val h = idx / width

                tensor[batchIndex, pixelFormat.rIndex, h, w] = convert((p shr 16) and 0xFF)
                tensor[batchIndex, pixelFormat.gIndex, h, w] = convert((p shr 8) and 0xFF)
                tensor[batchIndex, pixelFormat.bIndex, h, w] = convert((p shr 0) and 0xFF)
            }
        }

        is PixelFormat.RGBA -> {

            for (idx in 0 until total) {
                val p = pixels[idx]
                val w = idx % width
                val h = idx / width

                tensor[batchIndex, pixelFormat.rIndex, h, w] = convert((p shr 16) and 0xFF)
                tensor[batchIndex, pixelFormat.gIndex, h, w] = convert((p shr 8) and 0xFF)
                tensor[batchIndex, pixelFormat.bIndex, h, w] = convert((p shr 0) and 0xFF)
                tensor[batchIndex, pixelFormat.aIndex, h, w] = convert((p shr 24) and 0xFF)
            }
        }
    }
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

    val pixels = getPixels()

    // При NHWC плоский индекс элемента [0, h, w, c] равен (h * width + w) * channels + c,
    // то есть его можно наращивать по ходу обхода и писать через setFlat, полностью минуя
    // пересчёт offset'а по страйдам на каждый канал.
    val sequential = layout == ImageTensorLayout.NHWC

    // Циклы вынесены в отдельные небольшие функции намеренно: ART не оптимизирует крупные
    // методы, а один when со всеми вариантами делал тензоризацию именно таким методом.
    when (pixelFormat) {
        PixelFormat.Grayscale -> {
            if (sequential) {
                fillGrayscaleSequential(tensor, pixels, normalization)
            } else {
                fillGrayscale(tensor, pixels, width, normalization)
            }
        }

        is PixelFormat.RGB -> {
            if (sequential) {
                fillRgbSequential(tensor, pixels, pixelFormat, normalization)
            } else {
                fillRgb(tensor, pixels, width, pixelFormat, normalization)
            }
        }

        is PixelFormat.RGBA -> {
            if (sequential) {
                fillRgbaSequential(tensor, pixels, pixelFormat, normalization)
            } else {
                fillRgba(tensor, pixels, width, pixelFormat, normalization)
            }
        }
    }
    return tensor
}

private fun fillGrayscaleSequential(
    tensor: ImageTensor<Float>,
    pixels: IntArray,
    normalization: Normalization
) {
    val mean = normalization.meanR
    val std = normalization.stdR
    for (idx in pixels.indices) {
        tensor.setFlat(idx, ((pixels[idx] and 0xFF).toFloat() - mean) / std)
    }
}

private fun fillGrayscale(
    tensor: ImageTensor<Float>,
    pixels: IntArray,
    width: Int,
    normalization: Normalization
) {
    val mean = normalization.meanR
    val std = normalization.stdR
    for (idx in pixels.indices) {
        tensor[0, idx / width, idx % width, 0] = ((pixels[idx] and 0xFF).toFloat() - mean) / std
    }
}

private fun fillRgbSequential(
    tensor: ImageTensor<Float>,
    pixels: IntArray,
    pixelFormat: PixelFormat.RGB,
    normalization: Normalization
) {
    val rIndex = pixelFormat.rIndex
    val gIndex = pixelFormat.gIndex
    val bIndex = pixelFormat.bIndex
    val meanR = normalization.meanR
    val meanG = normalization.meanG
    val meanB = normalization.meanB
    val stdR = normalization.stdR
    val stdG = normalization.stdG
    val stdB = normalization.stdB

    var base = 0
    for (idx in pixels.indices) {
        val p = pixels[idx]
        tensor.setFlat(base + rIndex, (((p shr 16) and 0xFF).toFloat() - meanR) / stdR)
        tensor.setFlat(base + gIndex, (((p shr 8) and 0xFF).toFloat() - meanG) / stdG)
        tensor.setFlat(base + bIndex, ((p and 0xFF).toFloat() - meanB) / stdB)
        base += 3
    }
}

private fun fillRgb(
    tensor: ImageTensor<Float>,
    pixels: IntArray,
    width: Int,
    pixelFormat: PixelFormat.RGB,
    normalization: Normalization
) {
    val rIndex = pixelFormat.rIndex
    val gIndex = pixelFormat.gIndex
    val bIndex = pixelFormat.bIndex

    for (idx in pixels.indices) {
        val p = pixels[idx]
        val w = idx % width
        val h = idx / width
        tensor[0, h, w, rIndex] =
            (((p shr 16) and 0xFF).toFloat() - normalization.meanR) / normalization.stdR
        tensor[0, h, w, gIndex] =
            (((p shr 8) and 0xFF).toFloat() - normalization.meanG) / normalization.stdG
        tensor[0, h, w, bIndex] =
            ((p and 0xFF).toFloat() - normalization.meanB) / normalization.stdB
    }
}

private fun fillRgbaSequential(
    tensor: ImageTensor<Float>,
    pixels: IntArray,
    pixelFormat: PixelFormat.RGBA,
    normalization: Normalization
) {
    val rIndex = pixelFormat.rIndex
    val gIndex = pixelFormat.gIndex
    val bIndex = pixelFormat.bIndex
    val aIndex = pixelFormat.aIndex
    val meanR = normalization.meanR
    val meanG = normalization.meanG
    val meanB = normalization.meanB
    val meanA = normalization.meanA
    val stdR = normalization.stdR
    val stdG = normalization.stdG
    val stdB = normalization.stdB
    val stdA = normalization.stdA

    var base = 0
    for (idx in pixels.indices) {
        val p = pixels[idx]
        tensor.setFlat(base + rIndex, (((p shr 16) and 0xFF).toFloat() - meanR) / stdR)
        tensor.setFlat(base + gIndex, (((p shr 8) and 0xFF).toFloat() - meanG) / stdG)
        tensor.setFlat(base + bIndex, ((p and 0xFF).toFloat() - meanB) / stdB)
        tensor.setFlat(base + aIndex, (((p shr 24) and 0xFF).toFloat() - meanA) / stdA)
        base += 4
    }
}

private fun fillRgba(
    tensor: ImageTensor<Float>,
    pixels: IntArray,
    width: Int,
    pixelFormat: PixelFormat.RGBA,
    normalization: Normalization
) {
    val rIndex = pixelFormat.rIndex
    val gIndex = pixelFormat.gIndex
    val bIndex = pixelFormat.bIndex
    val aIndex = pixelFormat.aIndex

    for (idx in pixels.indices) {
        val p = pixels[idx]
        val w = idx % width
        val h = idx / width
        tensor[0, h, w, rIndex] =
            (((p shr 16) and 0xFF).toFloat() - normalization.meanR) / normalization.stdR
        tensor[0, h, w, gIndex] =
            (((p shr 8) and 0xFF).toFloat() - normalization.meanG) / normalization.stdG
        tensor[0, h, w, bIndex] =
            ((p and 0xFF).toFloat() - normalization.meanB) / normalization.stdB
        tensor[0, h, w, aIndex] =
            (((p shr 24) and 0xFF).toFloat() - normalization.meanA) / normalization.stdA
    }
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
    val w = first.width
    val h = first.height
    val pf = first.pixelFormat
    val n = size

    val tensor = ImageTensor(
        width = w,
        height = h,
        dataType = TensorDataType.Float32,
        pixelFormat = pf,
        layout = layout,
        batchSize = n
    )

    forEachIndexed { batchIndex, image ->
        require(image.width == w && image.height == h) {
            "All images must have the same size"
        }

        val pixels = image.getPixels()
        val total = w * h

        when (pixelFormat) {
            PixelFormat.Grayscale -> {
                for (idx in 0 until total) {
                    val pxW = idx % w
                    val pxH = idx / w
                    val p = pixels[idx]
                    val v = (p and 0xFF).toFloat()
                    tensor[batchIndex, pxH, pxW, 0] = (v - normalization.meanR) / normalization.stdR
                }
            }

            is PixelFormat.RGB -> {
                for (idx in 0 until total) {
                    val p = pixels[idx]
                    val pxW = idx % w
                    val pxH = idx / w

                    val r = ((p shr 16) and 0xFF).toFloat()
                    val g = ((p shr 8) and 0xFF).toFloat()
                    val b = (p and 0xFF).toFloat()

                    tensor[batchIndex, pxH, pxW, pixelFormat.rIndex] = (r - normalization.meanR) / normalization.stdR
                    tensor[batchIndex, pxH, pxW, pixelFormat.gIndex] = (g - normalization.meanG) / normalization.stdG
                    tensor[batchIndex, pxH, pxW, pixelFormat.bIndex] = (b - normalization.meanB) / normalization.stdB
                }
            }

            is PixelFormat.RGBA -> {
                for (idx in 0 until total) {
                    val p = pixels[idx]
                    val pxW = idx % w
                    val pxH = idx / w

                    val a = ((p shr 24) and 0xFF).toFloat()
                    val r = ((p shr 16) and 0xFF).toFloat()
                    val g = ((p shr 8) and 0xFF).toFloat()
                    val b = (p and 0xFF).toFloat()

                    tensor[batchIndex, pxH, pxW, pixelFormat.rIndex] = (r - normalization.meanR) / normalization.stdR
                    tensor[batchIndex, pxH, pxW, pixelFormat.gIndex] = (g - normalization.meanG) / normalization.stdG
                    tensor[batchIndex, pxH, pxW, pixelFormat.bIndex] = (b - normalization.meanB) / normalization.stdB
                    tensor[batchIndex, pxH, pxW, pixelFormat.aIndex] = (a - normalization.meanA) / normalization.stdA
                }
            }
        }
    }

    return tensor
}

@Suppress("UNCHECKED_CAST")
private val <T : Any> TensorDataType<T>.converter: (Int) -> T
    get() = when (this) {
        TensorDataType.Float32 -> { v: Int -> v.toFloat() as T }
        TensorDataType.Int32 -> { v: Int -> v as T }
        TensorDataType.Int64 -> { v: Int -> v.toLong() as T }
        TensorDataType.UInt8 -> { v: Int -> v.toUByte() as T }
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

    when (val pf = pixelFormat) {
        PixelFormat.Grayscale -> {
            var idx = 0
            for (h in 0 until height) {
                for (w in 0 until width) {
                    val v =
                        (this[batchIndex, h, w, 0] * normalization.stdR + normalization.meanR).toInt()
                            .coerceIn(0, 255)
                    pixels[idx++] = (0xFF shl 24) or (v shl 16) or (v shl 8) or v
                }
            }
        }

        is PixelFormat.RGB -> {
            var idx = 0
            for (h in 0 until height) {
                for (w in 0 until width) {
                    val r =
                        (this[batchIndex, h, w, pf.rIndex] * normalization.stdR + normalization.meanR).toInt()
                            .coerceIn(0, 255)
                    val g =
                        (this[batchIndex, h, w, pf.gIndex] * normalization.stdG + normalization.meanG).toInt()
                            .coerceIn(0, 255)
                    val b =
                        (this[batchIndex, h, w, pf.bIndex] * normalization.stdB + normalization.meanB).toInt()
                            .coerceIn(0, 255)

                    pixels[idx++] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                }
            }
        }

        is PixelFormat.RGBA -> {
            var idx = 0
            for (h in 0 until height) {
                for (w in 0 until width) {
                    val r =
                        (this[batchIndex, h, w, pf.rIndex] * normalization.stdR + normalization.meanR).toInt()
                            .coerceIn(0, 255)
                    val g =
                        (this[batchIndex, h, w, pf.gIndex] * normalization.stdG + normalization.meanG).toInt()
                            .coerceIn(0, 255)
                    val b =
                        (this[batchIndex, h, w, pf.bIndex] * normalization.stdB + normalization.meanB).toInt()
                            .coerceIn(0, 255)
                    val a =
                        (this[batchIndex, h, w, pf.aIndex] * normalization.stdA + normalization.meanA).toInt()
                            .coerceIn(0, 255)

                    pixels[idx++] = (a shl 24) or (r shl 16) or (g shl 8) or b
                }
            }
        }
    }

    return Image(width, height, pixelFormat, pixels)
}