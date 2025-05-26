package dev.kursor.ktensorflow.impl

import dev.kursor.ktensorflow.api.TensorDataType
import dev.kursor.ktensorflow.api.TensorShape

internal fun ByteArray.toShapedAndTypedArray(
    dataType: TensorDataType,
    shape: TensorShape
): Any {
    val totalElements = shape.flatSize

    return when (dataType) {
        TensorDataType.Float32 -> reshapeArray(
            readFloatArray(this, totalElements),
            shape.dimensions
        )
        TensorDataType.Int32 -> reshapeArray(
            readIntArray(this, totalElements),
            shape.dimensions
        )
        TensorDataType.Uint8 -> reshapeArray(
            readByteArray(this, totalElements),
            shape.dimensions
        )
        TensorDataType.Int64 -> reshapeArray(
            readLongArray(this, totalElements),
            shape.dimensions
        )
    }
}

private fun readIntArray(bytes: ByteArray, count: Int): IntArray {
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

private fun readFloatArray(bytes: ByteArray, count: Int): FloatArray {
    val ints = readIntArray(bytes, count)
    return FloatArray(count) { i -> Float.fromBits(ints[i]) }
}

private fun readByteArray(bytes: ByteArray, count: Int): ByteArray =
    bytes.copyOf(count)

private fun readLongArray(bytes: ByteArray, count: Int): LongArray {
    val result = LongArray(count)
    for (i in 0 until count) {
        var value = 0L
        val offset = i * 8
        for (b in 0 until 8) {
            value = value or ((bytes[offset + b].toLong() and 0xFF) shl (b * 8))
        }
        result[i] = value
    }
    return result
}

private fun reshapeArray(flat: Any, dims: IntArray): Any {
    var index = 0

    fun build(shape: IntArray): Any {
        if (shape.size == 1) {
            val size = shape[0]
            return when (flat) {
                is FloatArray -> FloatArray(size) { flat[index++] }
                is IntArray -> IntArray(size) { flat[index++] }
                is ByteArray -> ByteArray(size) { flat[index++] }
                is LongArray -> LongArray(size) { flat[index++] }
                else -> error("Unsupported flat array type: ${flat::class}")
            }
        } else {
            val size = shape[0]
            val rest = shape.copyOfRange(1, shape.size)
            return Array(size) { build(rest) }
        }
    }

    return build(dims)
}
