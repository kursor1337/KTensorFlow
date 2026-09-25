package dev.kursor.ktensorflow.tensor.impl

import dev.kursor.ktensorflow.tensor.TensorShape

@OptIn(ExperimentalUnsignedTypes::class)
internal fun inferTensorShape(data: Any): TensorShape {
    val dims = mutableListOf<Int>()
    var current = data
    while (current is Array<*>) {
        dims += current.size
        // У пустого массива нет элемента, по которому видны внутренние измерения; раньше
        // здесь падало невнятное "Unsupported tensor data type: Array"
        require(current.isNotEmpty()) {
            "Cannot infer the shape of an empty nested array: its inner dimensions are unknown. " +
                "Create the tensor with an explicit TensorShape instead"
        }
        current = requireNotNull(current[0]) { "Nested arrays must not contain null" }
    }
    dims += when (current) {
        is FloatArray -> current.size
        is IntArray -> current.size
        is UByteArray -> current.size
        is LongArray -> current.size
        else -> throw IllegalArgumentException(
            "Unsupported tensor element type ${current::class.simpleName}: " +
                "use FloatArray, IntArray, UByteArray or LongArray innermost"
        )
    }
    return TensorShape(dims.toIntArray())
}
