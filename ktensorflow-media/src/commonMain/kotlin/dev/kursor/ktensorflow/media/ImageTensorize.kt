package dev.kursor.ktensorflow.media

import dev.kursor.ktensorflow.tensor.Tensor
import dev.kursor.ktensorflow.tensor.TensorDataType
import dev.kursor.ktensorflow.tensor.set

inline fun <reified T : Any> Image.tensorize(
    layout: ImageTensorLayout = ImageTensorLayout.NHWC
) = tensorize(
    dataType = TensorDataType.of<T>(),
    layout = layout
)

fun <T : Any> Image.tensorize(
    dataType: TensorDataType<T>,
    layout: ImageTensorLayout = ImageTensorLayout.NHWC
): ImageTensor<T> = when (dataType) {
    TensorDataType.Float32 -> tensorize(
        dataType = TensorDataType.Float32,
        layout = layout,
        convert = Int::toFloat
    )
    TensorDataType.Int32 -> tensorize(
        dataType = TensorDataType.Int32,
        layout = layout,
        convert = Int::toInt
    )
    TensorDataType.Int64 -> tensorize(
        dataType = TensorDataType.Int64,
        layout = layout,
        convert = Int::toLong
    )
    TensorDataType.UInt8 -> tensorize(
        dataType = TensorDataType.UInt8,
        layout = layout,
        convert = Int::toUByte
    )
} as ImageTensor<T>

private fun <T : Any> Image.tensorize(
    dataType: TensorDataType<T>,
    layout: ImageTensorLayout = ImageTensorLayout.NHWC,
    convert: (Int) -> T
): ImageTensor<T> {

    val tensor = ImageTensor(
        width = width,
        height = height,
        dataType = dataType,
        pixelFormat = pixelFormat,
        layout = layout
    )

    val pixelFormat = pixelFormat
    val pixels = getPixels()
    val total = width * height

    when (pixelFormat) {
        PixelFormat.Grayscale -> {
            for (idx in 0 until total) {
                val w = idx % width
                val h = idx / width
                val v = pixels[idx] and 0xFF

                tensor[0, h, w, 0] = convert(v)
            }
        }

        is PixelFormat.RGB -> {
            val rShift = pixelFormat.rIndex * 8
            val gShift = pixelFormat.gIndex * 8
            val bShift = pixelFormat.bIndex * 8

            for (idx in 0 until total) {
                val p = pixels[idx]
                val w = idx % width
                val h = idx / width

                tensor[0, h, w, pixelFormat.rIndex] = convert((p shr rShift) and 0xFF)
                tensor[0,  h, w, pixelFormat.gIndex] = convert((p shr gShift) and 0xFF)
                tensor[0, h, w, pixelFormat.bIndex] = convert((p shr bShift) and 0xFF)
            }
        }

        is PixelFormat.RGBA -> {
            val rShift = pixelFormat.rIndex * 8
            val gShift = pixelFormat.gIndex * 8
            val bShift = pixelFormat.bIndex * 8
            val aShift = pixelFormat.aIndex * 8

            for (idx in 0 until total) {
                val p = pixels[idx]
                val w = idx % width
                val h = idx / width

                tensor[0, h, w, pixelFormat.rIndex] = convert((p shr rShift) and 0xFF)
                tensor[0, h, w, pixelFormat.gIndex] = convert((p shr gShift) and 0xFF)
                tensor[0, h, w, pixelFormat.bIndex] = convert((p shr bShift) and 0xFF)
                tensor[0, h, w, pixelFormat.aIndex] = convert((p shr aShift) and 0xFF)
            }
        }
    }

    return tensor
}

inline fun <reified T : Any> List<Image>.tensorizeBatch(
    layout: ImageTensorLayout = ImageTensorLayout.NHWC
): ImageTensor<T> = tensorizeBatch(
    dataType = TensorDataType.of<T>(),
    layout = layout
)

fun <T : Any> List<Image>.tensorizeBatch(
    dataType: TensorDataType<T>,
    layout: ImageTensorLayout = ImageTensorLayout.NHWC
): ImageTensor<T> {

    require(isNotEmpty()) { "Empty image batch" }

    val first = first()
    val w = first.width
    val h = first.height
    val pf = first.pixelFormat
    val c = pf.channels
    val n = size

    // 🔥 одна аллокация
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
            convert = dataType.converter()
        )
    }

    return tensor
}

private inline fun <T : Any> writeImageInto(
    image: Image,
    tensor: ImageTensor<T>,
    batchIndex: Int,
    convert: (Int) -> T
) {
    val width = image.width
    val height = image.height
    val pf = image.pixelFormat
    val pixels = image.getPixels()
    val total = width * height

    when (pf) {

        PixelFormat.Grayscale -> {
            for (idx in 0 until total) {
                val w = idx % width
                val h = idx / width
                val v = pixels[idx] and 0xFF

                tensor[batchIndex, 0, h, w] = convert(v)
            }
        }

        is PixelFormat.RGB -> {
            val rShift = pf.rIndex * 8
            val gShift = pf.gIndex * 8
            val bShift = pf.bIndex * 8

            for (idx in 0 until total) {
                val p = pixels[idx]
                val w = idx % width
                val h = idx / width

                tensor[batchIndex, pf.rIndex, h, w] = convert((p shr rShift) and 0xFF)
                tensor[batchIndex, pf.gIndex, h, w] = convert((p shr gShift) and 0xFF)
                tensor[batchIndex, pf.bIndex, h, w] = convert((p shr bShift) and 0xFF)
            }
        }

        is PixelFormat.RGBA -> {
            val rShift = pf.rIndex * 8
            val gShift = pf.gIndex * 8
            val bShift = pf.bIndex * 8
            val aShift = pf.aIndex * 8

            for (idx in 0 until total) {
                val p = pixels[idx]
                val w = idx % width
                val h = idx / width

                tensor[batchIndex, pf.rIndex, h, w] = convert((p shr rShift) and 0xFF)
                tensor[batchIndex, pf.gIndex, h, w] = convert((p shr gShift) and 0xFF)
                tensor[batchIndex, pf.bIndex, h, w] = convert((p shr bShift) and 0xFF)
                tensor[batchIndex, pf.aIndex, h, w] = convert((p shr aShift) and 0xFF)
            }
        }
    }
}

@Suppress("UNCHECKED_CAST")
fun <T : Any> TensorDataType<T>.converter(): (Int) -> T = when (this) {
    TensorDataType.Float32 -> { v: Int -> v.toFloat() as T }
    TensorDataType.Int32   -> { v: Int -> v as T }
    TensorDataType.Int64   -> { v: Int -> v.toLong() as T }
    TensorDataType.UInt8   -> { v: Int -> v.toUByte() as T }
}