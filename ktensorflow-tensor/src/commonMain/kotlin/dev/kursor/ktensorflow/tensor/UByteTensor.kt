package dev.kursor.ktensorflow.tensor

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
}