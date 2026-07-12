package dev.kursor.ktensorflow.tensor.views

import dev.kursor.ktensorflow.tensor.Tensor
import dev.kursor.ktensorflow.tensor.TensorDataType
import dev.kursor.ktensorflow.tensor.TensorShape
import dev.kursor.ktensorflow.tensor.TensorView
import dev.kursor.ktensorflow.tensor.strides
import dev.kursor.ktensorflow.tensor.toFlatIndex

internal class ReshapedTensorView<T : Any>(
    override val delegate: Tensor<T>,
    override val shape: TensorShape
) : TensorView<T> {

    init {
        require(shape.flatSize == delegate.shape.flatSize) {
            "Cannot reshape tensor of flat size ${delegate.shape.flatSize} to shape $shape"
        }
    }

    override val dataType: TensorDataType<T> = delegate.dataType

    private val viewStrides = shape.strides()

    override fun getFlat(index: Int): T = delegate.getFlat(index)
    override fun setFlat(index: Int, value: T) = delegate.setFlat(index, value)

    override fun get(index: IntArray): T {
        return delegate.getFlat(index.toFlatIndex(viewStrides))
    }

    override fun set(index: IntArray, value: T) {
        delegate.setFlat(index.toFlatIndex(viewStrides), value)
    }
}