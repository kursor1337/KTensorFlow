package dev.kursor.ktensorflow.tensor.physical

import dev.kursor.ktensorflow.tensor.PhysicalTensor
import dev.kursor.ktensorflow.tensor.TensorDataType
import dev.kursor.ktensorflow.tensor.TensorShape
import dev.kursor.ktensorflow.tensor.impl.toShapedAndTypedArray
import dev.kursor.ktensorflow.tensor.readFloat
import dev.kursor.ktensorflow.tensor.strides
import dev.kursor.ktensorflow.tensor.toFlatIndex
import dev.kursor.ktensorflow.tensor.writeFloat

/**
 * Represents a [dev.kursor.ktensorflow.tensor.Tensor] of kotlin type [Float] and [dev.kursor.ktensorflow.tensor.TensorDataType.Float32]
 *
 * @param shape - shape of the tensor
 * @param data - raw data of the tensor
 */
class FloatTensor(
    override val shape: TensorShape,
    override val data: ByteArray
) : PhysicalTensor<Float> {

    private val strides = shape.strides()

    override val dataType = TensorDataType.Float32

    override fun getFlat(index: Int): Float {
        return data.readFloat(index)
    }

    override fun setFlat(index: Int, value: Float) {
        data.writeFloat(index, value)
    }

    override fun get(index: IntArray): Float {
        return data.readFloat(index.toFlatIndex(strides))
    }

    override fun set(index: IntArray, value: Float) {
        data.writeFloat(index.toFlatIndex(strides), value)
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
