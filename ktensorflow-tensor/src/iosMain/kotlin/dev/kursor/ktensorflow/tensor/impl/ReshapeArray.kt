package dev.kursor.ktensorflow.tensor.impl

internal actual fun reshapeArray(flat: Any, dimentions: IntArray): Any {
    var index = 0
    fun build(shape: IntArray): Any {
        if (shape.size == 1) {
            val size = shape[0]
            return when (flat) {
                is FloatArray -> FloatArray(size) { flat[index++] }
                is IntArray -> IntArray(size) { flat[index++] }
                is UByteArray -> UByteArray(size) { flat[index++] }
                is LongArray -> LongArray(size) { flat[index++] }
                else -> error("Unsupported flat array type: ${flat::class}")
            }
        } else {
            val size = shape[0]
            val rest = shape.copyOfRange(1, shape.size)
            return Array(size) { build(rest) }
        }
    }

    return build(dimentions)
}