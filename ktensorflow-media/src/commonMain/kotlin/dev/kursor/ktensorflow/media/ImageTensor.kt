package dev.kursor.ktensorflow.media

import dev.kursor.ktensorflow.tensor.Tensor
import dev.kursor.ktensorflow.tensor.TensorDataType

interface ImageTensor<T : Any> : Tensor<T> {
    val pixelFormat: PixelFormat
    val layout: ImageTensorLayout

    val batches: Int
    val width: Int
    val height: Int
    val channels: Int

    operator fun get(n: Int, h: Int, w: Int, c: Int): T
    operator fun get(h: Int, w: Int, c: Int) = get(0, h, w, c)
    operator fun set(n: Int, h: Int, w: Int, c: Int, value: T)
    operator fun set(h: Int, w: Int, c: Int, value: T) = set(0, h, w, c, value)
}

fun <T : Any> ImageTensor(
    tensor: Tensor<T>,
    pixelFormat: PixelFormat,
    layout: ImageTensorLayout
): ImageTensor<T> = ImageTensorImpl(
    tensor = tensor,
    pixelFormat = pixelFormat,
    layout = layout
)

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
