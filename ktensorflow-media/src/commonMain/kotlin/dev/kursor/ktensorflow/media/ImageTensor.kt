package dev.kursor.ktensorflow.media

import dev.kursor.ktensorflow.tensor.Tensor
import dev.kursor.ktensorflow.tensor.TensorDataType
import dev.kursor.ktensorflow.tensor.TensorShape

class ImageTensor<T : Any>(
    tensor: Tensor<T>,
    val pixelFormat: PixelFormat
) : Tensor<T> by tensor {

    init {
        require(shape.rank == 3) { "ImageTensor must have 3 dimensions" }
    }

    val width get() = shape.dimensions[0]
    val height get() = shape.dimensions[1]
    val channels get() = shape.dimensions[2]

    companion object
}

inline fun <reified T : Any> ImageTensor(
    width: Int,
    height: Int,
    pixelFormat: PixelFormat
): ImageTensor<T> = ImageTensor(
    tensor = Tensor(
        dataType = TensorDataType.of<T>(),
        shape = TensorShape(width, height, pixelFormat.channels)
    ),
    pixelFormat = pixelFormat
)

fun <T : Any> ImageTensor(
    width: Int,
    height: Int,
    dataType: TensorDataType<T>,
    pixelFormat: PixelFormat
) = ImageTensor(
    tensor = Tensor(
        dataType = dataType,
        shape = TensorShape(width, height, pixelFormat.channels)
    ),
    pixelFormat = pixelFormat
)

fun ImageTensor(pixelFormat: PixelFormat, data: Array<Array<FloatArray>>) = ImageTensor(
    tensor = Tensor<Float>(data),
    pixelFormat = pixelFormat
)

fun <T : Any> Tensor<T>.toImageTensor(pixelFormat: PixelFormat): ImageTensor<T> {
    return (this as? ImageTensor<T>)
        ?.let {
            if (it.pixelFormat == pixelFormat) {
                it
            } else {
                null
            }
        }
        ?: ImageTensor(this, pixelFormat)
}