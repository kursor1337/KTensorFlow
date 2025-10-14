package dev.kursor.ktensorflow.tensor.impl

import dev.kursor.ktensorflow.tensor.TensorShape
import kotlin.reflect.KClass

@OptIn(ExperimentalUnsignedTypes::class)
internal fun <T : Any> ByteArray.toShapedAndTypedArray(
    dataType: KClass<T>,
    shape: TensorShape
): Any {

    return when (dataType) {
        Float::class -> reshapeArray(
            readFloatArray(this),
            shape.dimensions
        )
        Int::class -> reshapeArray(
            readIntArray(this),
            shape.dimensions
        )
        UByte::class -> reshapeArray(
            readUByteArray(this),
            shape.dimensions
        )
        Long::class -> reshapeArray(
            readLongArray(this),
            shape.dimensions
        )

        else -> throw IllegalArgumentException("Unsupported data type: $dataType")
    }
}

private fun readIntArray(bytes: ByteArray): IntArray {
    val count = bytes.size / 4
    val result = IntArray(count)
    for (i in 0 until count) {
        val offset = i * 4
        result[i] =
            (bytes[offset + 0].toInt() and 0xFF) or
                    ((bytes[offset + 1].toInt() and 0xFF) shl 8) or
                    ((bytes[offset + 2].toInt() and 0xFF) shl 16) or
                    ((bytes[offset + 3].toInt() and 0xFF) shl 24)
    }
    return result
}

private fun readFloatArray(bytes: ByteArray): FloatArray {
    val count = bytes.size / 4
    val ints = readIntArray(bytes)
    return FloatArray(count) { i -> Float.fromBits(ints[i]) }
}

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
