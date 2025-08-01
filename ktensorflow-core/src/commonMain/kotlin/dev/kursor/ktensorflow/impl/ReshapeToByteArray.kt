package dev.kursor.ktensorflow.impl

import dev.kursor.ktensorflow.TensorDataType
import dev.kursor.ktensorflow.TensorShape

internal fun Any.toByteArray(
    dataType: TensorDataType,
    shape: TensorShape
): ByteArray {
    val totalElements = shape.flatSize
    val byteArray = ByteArray(totalElements * dataType.byteSize)

    when (dataType) {
        TensorDataType.Float32 -> writeFlatFloatArray(flattenFloatArray(this, shape), byteArray)
        TensorDataType.Int32 -> writeFlatIntArray(flattenIntArray(this, shape), byteArray)
        TensorDataType.Uint8 -> writeFlatByteArray(flattenByteArray(this, shape), byteArray)
        TensorDataType.Int64 -> writeFlatLongArray(flattenLongArray(this, shape), byteArray)
    }

    return byteArray
}

private fun flattenFloatArray(array: Any, shape: TensorShape): FloatArray {
    val flat = FloatArray(shape.flatSize)
    var index = 0
    fun recurse(curr: Any) {
        when (curr) {
            is FloatArray -> {
                for (f in curr) flat[index++] = f
            }
            is Array<*> -> {
                for (sub in curr) recurse(sub!!)
            }
            else -> error("Unsupported type in FloatArray: ${curr::class}")
        }
    }
    recurse(array)
    return flat
}

private fun flattenIntArray(array: Any, shape: TensorShape): IntArray {
    val flat = IntArray(shape.flatSize)
    var index = 0
    fun recurse(curr: Any) {
        when (curr) {
            is IntArray -> {
                for (v in curr) flat[index++] = v
            }
            is Array<*> -> {
                for (sub in curr) recurse(sub!!)
            }
            else -> error("Unsupported type in IntArray: ${curr::class}")
        }
    }
    recurse(array)
    return flat
}

private fun flattenByteArray(array: Any, shape: TensorShape): ByteArray {
    val flat = ByteArray(shape.flatSize)
    var index = 0
    fun recurse(curr: Any) {
        when (curr) {
            is ByteArray -> {
                for (v in curr) flat[index++] = v
            }
            is Array<*> -> {
                for (sub in curr) recurse(sub!!)
            }
            else -> error("Unsupported type in ByteArray: ${curr::class}")
        }
    }
    recurse(array)
    return flat
}

private fun flattenLongArray(array: Any, shape: TensorShape): LongArray {
    val flat = LongArray(shape.flatSize)
    var index = 0
    fun recurse(curr: Any) {
        when (curr) {
            is LongArray -> {
                for (v in curr) flat[index++] = v
            }
            is Array<*> -> {
                for (sub in curr) recurse(sub!!)
            }
            else -> error("Unsupported type in LongArray: ${curr::class}")
        }
    }
    recurse(array)
    return flat
}

private fun writeFlatFloatArray(src: FloatArray, dest: ByteArray) {
    var i = 0
    for (f in src) {
        val bits = f.toBits()
        dest[i++] = (bits and 0xFF).toByte()
        dest[i++] = ((bits shr 8) and 0xFF).toByte()
        dest[i++] = ((bits shr 16) and 0xFF).toByte()
        dest[i++] = ((bits shr 24) and 0xFF).toByte()
    }
}

private fun writeFlatIntArray(src: IntArray, dest: ByteArray) {
    var i = 0
    for (v in src) {
        dest[i++] = (v and 0xFF).toByte()
        dest[i++] = ((v shr 8) and 0xFF).toByte()
        dest[i++] = ((v shr 16) and 0xFF).toByte()
        dest[i++] = ((v shr 24) and 0xFF).toByte()
    }
}

private fun writeFlatByteArray(src: ByteArray, dest: ByteArray) {
    for (i in src.indices) {
        dest[i] = src[i]
    }
}

private fun writeFlatLongArray(src: LongArray, dest: ByteArray) {
    var i = 0
    for (v in src) {
        for (b in 0 until 8) {
            dest[i++] = ((v shr (b * 8)) and 0xFF).toByte()
        }
    }
}
