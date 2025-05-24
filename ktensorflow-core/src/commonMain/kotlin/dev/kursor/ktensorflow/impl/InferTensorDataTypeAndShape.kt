package dev.kursor.ktensorflow.impl

import dev.kursor.ktensorflow.api.TensorDataType
import dev.kursor.ktensorflow.api.TensorShape

internal fun inferTensorDataType(data: Any): TensorDataType = when (data) {
    is FloatArray -> TensorDataType.Float32
    is IntArray -> TensorDataType.Int32
    is ByteArray -> TensorDataType.Uint8
    is LongArray -> TensorDataType.Int64
    is Array<*> -> {
        if (data.isEmpty()) error("Cannot infer data type from empty array")
        inferTensorDataType(data[0]!!)
    }
    else -> error("Unsupported tensor data type: ${data::class}")
}

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
        is ByteArray -> current.size
        is LongArray -> current.size
        else -> error("Unsupported tensor data type: ${current::class}")
    }
    return TensorShape(dims.toIntArray())
}
