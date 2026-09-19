package dev.kursor.ktensorflow.vision

import dev.kursor.ktensorflow.tensor.Tensor
import dev.kursor.ktensorflow.tensor.TensorShape
import dev.kursor.ktensorflow.tensor.reshape
import dev.kursor.ktensorflow.tensor.strides

internal class ImageTensorImpl<T : Any>(
    tensor: Tensor<T>,
    override val pixelFormat: PixelFormat,
    override val layout: ImageTensorLayout
) : ImageTensor<T>, Tensor<T> by normalize(tensor, layout) {

    init {
        require(shape.rank == 4) {
            "ImageTensor must have 4 dimensions: batch, width, height, channels"
        }
    }

    private val strides = shape.strides()

    // Страйды и размеры неизменны для конкретного тензора, поэтому считаются один раз:
    // в геттерах они обходились в 4 чтения массива на КАЖДОЕ обращение к элементу,
    // а тензоризация изображения делает сотни тысяч таких обращений на кадр.
    private val nStride: Int = strides[layout.nIndex]
    private val hStride: Int = strides[layout.hIndex]
    private val wStride: Int = strides[layout.wIndex]
    private val cStride: Int = strides[layout.cIndex]

    override val batch: Int = shape.dimensions[layout.nIndex]

    override val width: Int = shape.dimensions[layout.wIndex]

    override val height: Int = shape.dimensions[layout.hIndex]

    override val channels: Int = shape.dimensions[layout.cIndex]

    override operator fun get(
        n: Int,
        h: Int,
        w: Int,
        c: Int
    ): T = getFlat(offset(n, h, w, c))

    override operator fun set(
        n: Int,
        h: Int,
        w: Int,
        c: Int,
        value: T
    ) = setFlat(offset(n, h, w, c), value)

    private fun offset(n: Int, h: Int, w: Int, c: Int): Int {
        return n * nStride + h * hStride + w * wStride + c * cStride
    }

    companion object {
        private fun <T : Any> normalize(tensor: Tensor<T>, layout: ImageTensorLayout): Tensor<T> {
            return when (tensor.shape.rank) {
                4 -> tensor
                3 -> {
                    val expanded = with(tensor.shape.dimensions) {
                        take(layout.nIndex) + 1 + drop(layout.nIndex)
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