package dev.kursor.ktensorflow.tensor

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
}