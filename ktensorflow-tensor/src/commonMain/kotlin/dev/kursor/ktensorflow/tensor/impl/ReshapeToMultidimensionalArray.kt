package dev.kursor.ktensorflow.tensor.impl

import dev.kursor.ktensorflow.tensor.TensorDataType
import dev.kursor.ktensorflow.tensor.TensorShape

@OptIn(ExperimentalUnsignedTypes::class)
internal fun <T : Any> ByteArray.toShapedAndTypedArray(
    dataType: TensorDataType<T>,
    shape: TensorShape
): Any {
    // Скаляр (ранг 0) отдаётся массивом из одного элемента. Раньше он падал на обеих
    // платформах, причём с разными исключениями
    val dimensions = if (shape.rank == 0) intArrayOf(1) else shape.dimensions
    return when (dataType) {
        TensorDataType.Float32 -> reshapeArray(readFloatArray(this), dimensions)
        TensorDataType.Int32 -> reshapeArray(readIntArray(this), dimensions)
        TensorDataType.UInt8 -> reshapeArray(readUByteArray(this), dimensions)
        TensorDataType.Int64 -> reshapeArray(readLongArray(this), dimensions)
    }
}

private fun ByteArray.intAt(offset: Int): Int =
    (this[offset].toInt() and 0xFF) or
        ((this[offset + 1].toInt() and 0xFF) shl 8) or
        ((this[offset + 2].toInt() and 0xFF) shl 16) or
        ((this[offset + 3].toInt() and 0xFF) shl 24)

private fun readIntArray(bytes: ByteArray): IntArray =
    IntArray(bytes.size / 4) { i -> bytes.intAt(i * 4) }

// Сразу во FloatArray: раньше значения сначала читались в промежуточный IntArray того же размера
private fun readFloatArray(bytes: ByteArray): FloatArray =
    FloatArray(bytes.size / 4) { i -> Float.fromBits(bytes.intAt(i * 4)) }

@OptIn(ExperimentalUnsignedTypes::class)
private fun readUByteArray(bytes: ByteArray): UByteArray =
    bytes.toUByteArray()

private fun readLongArray(bytes: ByteArray): LongArray {
    val count = bytes.size / 8
    val result = LongArray(count)
    for (i in 0 until count) {
        val offset = i * 8
        result[i] =
            (bytes[offset + 0].toLong() and 0xFF) or
                    ((bytes[offset + 1].toLong() and 0xFF) shl 8) or
                    ((bytes[offset + 2].toLong() and 0xFF) shl 16) or
                    ((bytes[offset + 3].toLong() and 0xFF) shl 24) or
                    ((bytes[offset + 4].toLong() and 0xFF) shl 32) or
                    ((bytes[offset + 5].toLong() and 0xFF) shl 40) or
                    ((bytes[offset + 6].toLong() and 0xFF) shl 48) or
                    ((bytes[offset + 7].toLong() and 0xFF) shl 56)
    }
    return result
}
