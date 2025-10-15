package dev.kursor.ktensorflow.tensor

import dev.kursor.ktensorflow.tensor.impl.toShapedAndTypedArray

class FloatTensor(
    override val shape: TensorShape,
    override val data: ByteArray
) : Tensor<Float> {

    override val dataType = TensorDataType.Float32

    override fun get(index: IntArray): Float {
        return data.readFloat(index.toFlatIndex(shape))
    }

    override fun set(index: IntArray, value: Float) {
        data.writeFloat(index.toFlatIndex(shape), value)
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other == null || this::class != other::class) return false

        other as FloatTensor

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
