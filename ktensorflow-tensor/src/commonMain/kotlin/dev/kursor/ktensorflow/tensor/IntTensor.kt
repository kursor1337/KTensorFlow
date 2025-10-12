package dev.kursor.ktensorflow.tensor

import kotlin.reflect.KClass

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
}