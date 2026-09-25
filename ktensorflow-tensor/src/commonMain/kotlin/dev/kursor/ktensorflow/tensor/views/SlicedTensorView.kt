package dev.kursor.ktensorflow.tensor.views

import dev.kursor.ktensorflow.tensor.Tensor
import dev.kursor.ktensorflow.tensor.TensorDataType
import dev.kursor.ktensorflow.tensor.TensorShape
import dev.kursor.ktensorflow.tensor.TensorView
import dev.kursor.ktensorflow.tensor.strides

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

    // Оси view совпадают с осями исходника, а начало диапазонов - это постоянный сдвиг
    private val offsets = delegate.shape.strides().let { strides ->
        StridedOffsets(
            shape = shape,
            strides = strides,
            base = ranges.indices.sumOf { ranges[it].first * strides[it] }
        )
    }

    override fun get(index: IntArray): T = delegate.getFlat(offsets.of(index))

    override fun set(index: IntArray, value: T) = delegate.setFlat(offsets.of(index), value)

    override fun getFlat(index: Int): T = delegate.getFlat(offsets.of(index))

    override fun setFlat(index: Int, value: T) = delegate.setFlat(offsets.of(index), value)
}