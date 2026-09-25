package dev.kursor.ktensorflow.tensor.views

import dev.kursor.ktensorflow.tensor.Tensor
import dev.kursor.ktensorflow.tensor.TensorDataType
import dev.kursor.ktensorflow.tensor.TensorShape
import dev.kursor.ktensorflow.tensor.TensorView
import dev.kursor.ktensorflow.tensor.strides

internal class PermutedTensorView<T : Any>(
    override val delegate: Tensor<T>,
    private val permuteAxes: IntArray
) : TensorView<T> {

    override val dataType: TensorDataType<T> = delegate.dataType

    override val shape: TensorShape = TensorShape(
        *IntArray(delegate.shape.rank) { delegate.shape.dimensions[permuteAxes[it]] }
    )

    // Шаг оси view - это шаг той оси исходника, на которую она переставлена
    private val offsets = delegate.shape.strides().let { strides ->
        StridedOffsets(
            shape = shape,
            strides = IntArray(permuteAxes.size) { strides[permuteAxes[it]] },
            base = 0
        )
    }

    override fun get(index: IntArray): T = delegate.getFlat(offsets.of(index))

    override fun set(index: IntArray, value: T) = delegate.setFlat(offsets.of(index), value)

    override fun getFlat(index: Int): T = delegate.getFlat(offsets.of(index))

    override fun setFlat(index: Int, value: T) = delegate.setFlat(offsets.of(index), value)
}