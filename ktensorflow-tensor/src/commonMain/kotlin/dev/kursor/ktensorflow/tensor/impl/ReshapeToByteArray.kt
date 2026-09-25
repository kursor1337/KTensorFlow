package dev.kursor.ktensorflow.tensor.impl

import dev.kursor.ktensorflow.tensor.TensorDataType
import dev.kursor.ktensorflow.tensor.TensorShape

@OptIn(ExperimentalUnsignedTypes::class)
internal fun <T : Any> Any.toByteArray(
    dataType: TensorDataType<T>,
    shape: TensorShape
): ByteArray {
    val totalElements = shape.flatSize
    val byteArray = ByteArray(totalElements * dataType.byteSize)

    when (dataType) {
        TensorDataType.Float32 -> writeFlatFloatArray(flattenFloatArray(this, shape), byteArray)
        TensorDataType.Int32 -> writeFlatIntArray(flattenIntArray(this, shape), byteArray)
        TensorDataType.UInt8 -> writeFlatUByteArray(flattenUByteArray(this, shape), byteArray)
        TensorDataType.Int64 -> writeFlatLongArray(flattenLongArray(this, shape), byteArray)
    }

    return byteArray
}

private fun flattenFloatArray(array: Any, shape: TensorShape): FloatArray {
    val flat = FloatArray(shape.flatSize)
    var index = 0
    fun recurse(curr: Any?, depth: Int) {
        when (curr) {
            is FloatArray -> {
                requireLength(curr.size, shape, depth)
                for (v in curr) flat[index++] = v
            }
            is Array<*> -> {
                requireLength(curr.size, shape, depth)
                for (sub in curr) recurse(sub, depth + 1)
            }
            else -> unsupportedElement(curr, "FloatArray")
        }
    }
    recurse(array, 0)
    return flat
}

private fun flattenIntArray(array: Any, shape: TensorShape): IntArray {
    val flat = IntArray(shape.flatSize)
    var index = 0
    fun recurse(curr: Any?, depth: Int) {
        when (curr) {
            is IntArray -> {
                requireLength(curr.size, shape, depth)
                for (v in curr) flat[index++] = v
            }
            is Array<*> -> {
                requireLength(curr.size, shape, depth)
                for (sub in curr) recurse(sub, depth + 1)
            }
            else -> unsupportedElement(curr, "IntArray")
        }
    }
    recurse(array, 0)
    return flat
}

@OptIn(ExperimentalUnsignedTypes::class)
private fun flattenUByteArray(array: Any, shape: TensorShape): UByteArray {
    val flat = UByteArray(shape.flatSize)
    var index = 0
    fun recurse(curr: Any?, depth: Int) {
        when (curr) {
            is UByteArray -> {
                requireLength(curr.size, shape, depth)
                for (v in curr) flat[index++] = v
            }
            is Array<*> -> {
                requireLength(curr.size, shape, depth)
                for (sub in curr) recurse(sub, depth + 1)
            }
            else -> unsupportedElement(curr, "UByteArray")
        }
    }
    recurse(array, 0)
    return flat
}

private fun flattenLongArray(array: Any, shape: TensorShape): LongArray {
    val flat = LongArray(shape.flatSize)
    var index = 0
    fun recurse(curr: Any?, depth: Int) {
        when (curr) {
            is LongArray -> {
                requireLength(curr.size, shape, depth)
                for (v in curr) flat[index++] = v
            }
            is Array<*> -> {
                requireLength(curr.size, shape, depth)
                for (sub in curr) recurse(sub, depth + 1)
            }
            else -> unsupportedElement(curr, "LongArray")
        }
    }
    recurse(array, 0)
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

@OptIn(ExperimentalUnsignedTypes::class)
private fun writeFlatUByteArray(src: UByteArray, dest: ByteArray) {
    for (i in src.indices) {
        dest[i] = src[i].toByte()
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

/**
 * Проверяет длину подмассива на глубине [depth]. Форма берётся по первому элементу каждого
 * уровня, поэтому рваный массив раньше проходил: короткая строка молча дополнялась нулями
 * (и все следующие строки сдвигались), а длинная падала невнятным выходом за массив.
 */
private fun requireLength(size: Int, shape: TensorShape, depth: Int) {
    require(depth < shape.rank && size == shape.dimensions[depth]) {
        "Nested arrays must form a regular tensor of shape $shape, " +
            "but an array at depth $depth has $size elements"
    }
}

private fun unsupportedElement(element: Any?, expected: String): Nothing =
    throw IllegalArgumentException(
        "Expected nested arrays of $expected, got ${element?.let { it::class.simpleName } ?: "null"}"
    )
