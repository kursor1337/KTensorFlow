package dev.kursor.ktensorflow.tensor.views

import dev.kursor.ktensorflow.tensor.Tensor
import dev.kursor.ktensorflow.tensor.TensorDataType
import dev.kursor.ktensorflow.tensor.TensorShape
import dev.kursor.ktensorflow.tensor.TensorView
import dev.kursor.ktensorflow.tensor.strides
import dev.kursor.ktensorflow.tensor.toNestedIndex

internal class SlicedTensorView<T : Any>(
    override val delegate: Tensor<T>,
    private val ranges: Array<IntRange>
) : TensorView<T> {

    init {
        require(ranges.size == delegate.shape.rank) {
            "Ranges count (${ranges.size}) must match tensor rank (${delegate.shape.rank})"
        }
        // Индекс view переводится в физическое смещение без проверки по осям, поэтому диапазон
        // за пределами оси молча читал элементы соседней строки. Проверяется один раз здесь.
        ranges.forEachIndexed { axis, range ->
            val size = delegate.shape.dimensions[axis]
            require(range.first >= 0 && range.last < size && range.first <= range.last + 1) {
                "Range $range is outside axis $axis of size $size"
            }
        }
    }

    override val dataType: TensorDataType<T> = delegate.dataType

    // Новая форма — это размеры переданных диапазонов
    override val shape: TensorShape = TensorShape(
        *ranges.map { it.last - it.first + 1 }.toIntArray()
    )

    private val originalStrides = delegate.shape.strides()

    // Считаем стартовое физическое смещение один раз
    private val startPhysicalOffset: Int = ranges.mapIndexed { i, range ->
        range.first * originalStrides[i]
    }.sum()

    override fun get(index: IntArray): T {
        var physicalOffset = startPhysicalOffset
        for (i in index.indices) {
            physicalOffset += index[i] * originalStrides[i]
        }
        return delegate.getFlat(physicalOffset)
    }

    override fun set(index: IntArray, value: T) {
        var physicalOffset = startPhysicalOffset
        for (i in index.indices) {
            physicalOffset += index[i] * originalStrides[i]
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