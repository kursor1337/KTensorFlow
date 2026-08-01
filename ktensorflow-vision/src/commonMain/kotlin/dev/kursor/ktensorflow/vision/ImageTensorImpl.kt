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

    private val strides = shape.strides()

    private val nStride: Int
        get() = strides[layout.nIndex]
    private val hStride: Int
        get() = strides[layout.hIndex]
    private val wStride: Int
        get() = strides[layout.wIndex]
    private val cStride: Int
        get() = strides[layout.cIndex]

    init {
        require(shape.rank == 4) {
            "ImageTensor must have 4 dimensions: batch, width, height, channels"
        }
    }

    override val batch: Int
        get() = shape.dimensions[layout.nIndex]

    override val width: Int
        get() = shape.dimensions[layout.wIndex]

    override val height: Int
        get() = shape.dimensions[layout.hIndex]

    override val channels: Int
        get() = shape.dimensions[layout.cIndex]

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