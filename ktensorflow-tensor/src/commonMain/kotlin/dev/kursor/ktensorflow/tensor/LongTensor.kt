package dev.kursor.ktensorflow.tensor

import dev.kursor.ktensorflow.tensor.impl.toShapedAndTypedArray

/**
 * Represents a [Tensor] of kotlin type [Long] and [TensorDataType.Int64]
 *
 * @param shape - shape of the tensor
 * @param data - raw data of the tensor
 */
class LongTensor(
    override val shape: TensorShape,
    override val data: ByteArray
) : Tensor<Long> {

    override val dataType = TensorDataType.Int64

    override fun get(index: IntArray): Long {
        return data.readLong(index.toFlatIndex(shape))
    }

    override fun set(index: IntArray, value: Long) {
        data.writeLong(index.toFlatIndex(shape), value)
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other == null || this::class != other::class) return false

        other as LongTensor

        if (shape != other.shape) return false
        if (!data.contentEquals(other.data)) return false
        if (dataType != other.dataType) return false

        return true
    }

    override fun hashCode(): Int {
        var result = shape.hashCode()
        result = 31 * result + data.contentHashCode()
        result = 31 * result + dataType.hashCode()
        return result
    }

    override fun toString(): String {
        val array = data.toShapedAndTypedArray(dataType, shape) as Array<*>
        return array.contentDeepToString()
    }
}