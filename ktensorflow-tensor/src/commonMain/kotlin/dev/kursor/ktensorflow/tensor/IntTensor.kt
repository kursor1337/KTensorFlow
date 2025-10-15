package dev.kursor.ktensorflow.tensor

import dev.kursor.ktensorflow.tensor.impl.toShapedAndTypedArray

class IntTensor(
    override val shape: TensorShape,
    override val data: ByteArray
) : Tensor<Int> {

    override val dataType = TensorDataType.Int32

    override fun get(index: IntArray): Int {
        return data.readInt(index.toFlatIndex(shape))
    }

    override fun set(index: IntArray, value: Int) {
        data.writeInt(index.toFlatIndex(shape), value)
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other == null || this::class != other::class) return false

        other as IntTensor

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