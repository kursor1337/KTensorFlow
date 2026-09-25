package dev.kursor.ktensorflow.tensor.views

import dev.kursor.ktensorflow.tensor.TensorShape

/**
 * Переводит индексы view в плоское смещение в исходном тензоре: у каждой оси view есть свой
 * физический шаг [strides] и общий сдвиг [base].
 *
 * Плоский индекс раскладывается делением прямо здесь, без промежуточного IntArray: раньше
 * getFlat выделял массив на каждый элемент, а через getFlat идёт и toPhysical. Обход с
 * наращиваемым индексом пробовался и оказался медленнее: на Android на 40%, на iOS выигрыш 2-6%.
 */
internal class StridedOffsets(
    shape: TensorShape,
    private val strides: IntArray,
    private val base: Int
) {
    private val dimensions = shape.dimensions.copyOf()
    private val flatSize = shape.flatSize

    fun of(index: IntArray): Int {
        var offset = base
        for (i in index.indices) {
            offset += index[i] * strides[i]
        }
        return offset
    }

    fun of(flatIndex: Int): Int {
        require(flatIndex in 0 until flatSize) {
            "Flat index $flatIndex out of bounds for shape ${dimensions.contentToString()}"
        }
        var remainder = flatIndex
        var offset = base
        for (axis in dimensions.size - 1 downTo 0) {
            val size = dimensions[axis]
            offset += (remainder % size) * strides[axis]
            remainder /= size
        }
        return offset
    }
}
