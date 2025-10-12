package dev.kursor.ktensorflow.tensor.impl

import dev.kursor.ktensorflow.tensor.TensorShape

@OptIn(ExperimentalUnsignedTypes::class)
internal fun inferTensorShape(data: Any): TensorShape {
    val dims = mutableListOf<Int>()
    var current = data
    while (current is Array<*>) {
        dims += current.size
        current = if (current.isNotEmpty()) current[0]!! else break
    }
    dims += when (current) {
        is FloatArray -> current.size
        is IntArray -> current.size
        is UByteArray -> current.size
        is LongArray -> current.size
        else -> error("Unsupported tensor data type: ${current::class}")
    }
    return TensorShape(dims.toIntArray())
}
