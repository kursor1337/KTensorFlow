package dev.kursor.ktensorflow.tensor.views

import dev.kursor.ktensorflow.tensor.Tensor
import dev.kursor.ktensorflow.tensor.TensorDataType
import dev.kursor.ktensorflow.tensor.TensorShape
import dev.kursor.ktensorflow.tensor.TensorView
import dev.kursor.ktensorflow.tensor.strides
import dev.kursor.ktensorflow.tensor.toNestedIndex

internal class PermutedTensorView<T : Any>(
    override val delegate: Tensor<T>,
    private val permuteAxes: IntArray
) : TensorView<T> {

    override val dataType: TensorDataType<T> = delegate.dataType

    override val shape: TensorShape = TensorShape(
        *IntArray(delegate.shape.rank) { delegate.shape.dimensions[permuteAxes[it]] }
    )

    private val originalStrides = delegate.shape.strides()

    override fun get(index: IntArray): T {
        var physicalOffset = 0
        for (i in index.indices) {
            // Магия: мапим ось View обратно на оригинальную ось и умножаем на оригинальный страйд
            val origAxis = permuteAxes[i]
            physicalOffset += index[i] * originalStrides[origAxis]
        }
        return delegate.getFlat(physicalOffset)
    }

    override fun set(index: IntArray, value: T) {
        var physicalOffset = 0
        for (i in index.indices) {
            val origAxis = permuteAxes[i]
            physicalOffset += index[i] * originalStrides[origAxis]
        }
        delegate.setFlat(physicalOffset, value)
    }

    override fun getFlat(index: Int): T {
        return get(index.toNestedIndex(shape))
    }

    override fun setFlat(index: Int, value: T) {
        set(index.toNestedIndex(shape), value)
    }
}