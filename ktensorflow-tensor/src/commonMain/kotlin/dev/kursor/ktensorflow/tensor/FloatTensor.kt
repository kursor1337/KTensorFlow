package dev.kursor.ktensorflow.tensor

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
}
