package dev.kursor.ktensorflow.media

import dev.kursor.ktensorflow.tensor.TensorDataType
import dev.kursor.ktensorflow.tensor.set

inline fun <reified T : Any> Image.tensorize() = tensorize(
    dataType = TensorDataType.of<T>()
)

fun <T : Any> Image.tensorize(
    dataType: TensorDataType<T>,
): ImageTensor<T> = when (dataType) {
    TensorDataType.Float32 -> tensorize(
        dataType = TensorDataType.Float32,
        convert = Int::toFloat
    )
    TensorDataType.Int32 -> tensorize(
        dataType = TensorDataType.Int32,
        convert = { it }
    )
    TensorDataType.Int64 -> tensorize(
        dataType = TensorDataType.Int64,
        convert = Int::toLong
    )
    TensorDataType.UInt8 -> tensorize(
        dataType = TensorDataType.UInt8,
        convert = Int::toUByte
    )
} as ImageTensor<T>

private inline fun <T : Any> Image.tensorize(
    dataType: TensorDataType<T>,
    convert: (Int) -> T
): ImageTensor<T> {
    val tensor = ImageTensor(
        width = width,
        height = height,
        pixelFormat = pixelFormat,
        dataType = dataType
    )

    val pixelFormat = this.pixelFormat
    val pixels = getPixels() // Один вызов — весь буфер сразу!
    val totalPixels = width * height

    when (pixelFormat) {
        PixelFormat.Grayscale -> {
            for (idx in 0 until totalPixels) {
                val pixel = pixels[idx]
                val y = pixel and 0xFF
                tensor[idx / width, idx % width, 0] = convert(y) // или лучше по линейному индексу
            }
        }

        is PixelFormat.RGB -> {
            val rShift = pixelFormat.rIndex * 8
            val gShift = pixelFormat.gIndex * 8
            val bShift = pixelFormat.bIndex * 8

            for (idx in 0 until totalPixels) {
                val pixel = pixels[idx]
                val r = (pixel shr rShift) and 0xFF
                val g = (pixel shr gShift) and 0xFF
                val b = (pixel shr bShift) and 0xFF

                val x = idx % width
                val y = idx / width

                tensor[y, x, pixelFormat.rIndex] = convert(r)
                tensor[y, x, pixelFormat.gIndex] = convert(g)
                tensor[y, x, pixelFormat.bIndex] = convert(b)
            }
        }

        is PixelFormat.RGBA -> {
            val rShift = pixelFormat.rIndex * 8
            val gShift = pixelFormat.gIndex * 8
            val bShift = pixelFormat.bIndex * 8
            val aShift = pixelFormat.aIndex * 8

            for (idx in 0 until totalPixels) {
                val pixel = pixels[idx]
                val r = (pixel shr rShift) and 0xFF
                val g = (pixel shr gShift) and 0xFF
                val b = (pixel shr bShift) and 0xFF
                val a = (pixel shr aShift) and 0xFF

                val x = idx % width
                val y = idx / width

                tensor[y, x, pixelFormat.rIndex] = convert(r)
                tensor[y, x, pixelFormat.gIndex] = convert(g)
                tensor[y, x, pixelFormat.bIndex] = convert(b)
                tensor[y, x, pixelFormat.aIndex] = convert(a)
            }
        }
    }

    return tensor
}