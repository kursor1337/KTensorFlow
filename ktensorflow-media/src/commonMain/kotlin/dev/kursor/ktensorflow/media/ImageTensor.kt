package dev.kursor.ktensorflow.media

import dev.kursor.ktensorflow.tensor.Tensor
import dev.kursor.ktensorflow.tensor.TensorDataType
import dev.kursor.ktensorflow.tensor.TensorShape
import dev.kursor.ktensorflow.tensor.reshape
import dev.kursor.ktensorflow.tensor.squeeze
import kotlin.coroutines.coroutineContext

class ImageTensor<T : Any>(
    tensor: Tensor<T>,
    val pixelFormat: PixelFormat,
    val layout: ImageTensorLayout
) : Tensor<T> by normalize(tensor, layout) {

    init {
        require(shape.rank == 4) {
            "ImageTensor must have 4 dimensions: batch, width, height, channels"
        }
    }

    val batch get() = shape.dimensions[layout.nIndex]
    val width get() = shape.dimensions[layout.wIndex]
    val height get() = shape.dimensions[layout.hIndex]
    val channels get() = shape.dimensions[layout.cIndex]

    operator fun get(
        n: Int,
        h: Int,
        w: Int,
        c: Int
    ): T {
        val coords = IntArray(4)
        coords[layout.nIndex] = n
        coords[layout.hIndex] = h
        coords[layout.wIndex] = w
        coords[layout.cIndex] = c
        return get(coords)
    }

    operator fun set(
        n: Int,
        h: Int,
        w: Int,
        c: Int,
        value: T
    ) {
        val coords = IntArray(4)
        coords[layout.nIndex] = n
        coords[layout.hIndex] = h
        coords[layout.wIndex] = w
        coords[layout.cIndex] = c
        set(coords, value)
    }

    companion object {
        private fun <T : Any> normalize(tensor: Tensor<T>, layout: ImageTensorLayout): Tensor<T> {
            return when (tensor.shape.rank) {
                4 -> tensor
                3 -> {
                    val expanded = with(tensor.shape.dimensions) {
                        take(layout.nIndex) + 1 + takeLast(lastIndex - layout.nIndex)
                    }
                        .toIntArray()
                        .let(::TensorShape)
                    tensor.reshape(expanded)
                }
                else -> throw IllegalArgumentException(
                    "ImageTensor must have 3 or 4 dimensions: batch, width, height, channels"
                )
            }
        }
    }
}

inline fun <reified T : Any> ImageTensor(
    width: Int,
    height: Int,
    pixelFormat: PixelFormat,
    layout: ImageTensorLayout = ImageTensorLayout.NHWC,
    batchSize: Int = 1
): ImageTensor<T> = ImageTensor(
    tensor = Tensor(
        dataType = TensorDataType.of<T>(),
        shape = TensorShape(
            n = batchSize,
            h = height,
            w = width,
            c = pixelFormat.channels,
            layout = layout
        )
    ),
    pixelFormat = pixelFormat,
    layout = layout
)

fun <T : Any> ImageTensor(
    width: Int,
    height: Int,
    dataType: TensorDataType<T>,
    pixelFormat: PixelFormat,
    layout: ImageTensorLayout = ImageTensorLayout.NHWC,
    batchSize: Int = 1
) = ImageTensor(
    tensor = Tensor(
        dataType = dataType,
        shape = TensorShape(
            n = batchSize,
            h = height,
            w = width,
            c = pixelFormat.channels,
            layout = layout
        )
    ),
    pixelFormat = pixelFormat,
    layout = layout
)

fun ImageTensor(
    layout: ImageTensorLayout,
    pixelFormat: PixelFormat,
    data: Array<Array<FloatArray>>
) = ImageTensor(
    tensor = Tensor<Float>(data),
    pixelFormat = pixelFormat,
    layout = layout
)

fun <T : Any> Tensor<T>.toImageTensor(
    pixelFormat: PixelFormat,
    layout: ImageTensorLayout
): ImageTensor<T> {
    return (this as? ImageTensor<T>)
        ?.let {
            if (it.pixelFormat == pixelFormat && it.layout == layout) {
                it
            } else {
                null
            }
        }
        ?: ImageTensor(this, pixelFormat, layout)
}
