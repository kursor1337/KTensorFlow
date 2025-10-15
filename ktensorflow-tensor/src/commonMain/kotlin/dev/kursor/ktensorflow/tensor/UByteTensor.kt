package dev.kursor.ktensorflow.tensor

import dev.kursor.ktensorflow.tensor.impl.toShapedAndTypedArray

class UByteTensor(
    override val shape: TensorShape,
    override val data: ByteArray
) : Tensor<UByte> {

    override val dataType = TensorDataType.UInt8

    override fun get(index: IntArray): UByte {
        return data.readUByte(index.toFlatIndex(shape))
    }

    override fun set(index: IntArray, value: UByte) {
        data.writeUByte(index.toFlatIndex(shape), value)
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other == null || this::class != other::class) return false

        other as UByteTensor

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