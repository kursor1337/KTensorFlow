package dev.kursor.ktensorflow

/**
 * Tensor data types.
 */
enum class TensorDataType {
    /**
     * 32-bit floating point. Corresponds to [Float] in Kotlin.
     */
    Float32,

    /**
     * 32-bit signed integer. Corresponds to [Int] in Kotlin.
     */
    Int32,

    /**
     * 8-bit unsigned integer. Corresponds to [Byte] in Kotlin.
     */
    Uint8,

    /**
     * 64-bit signed integer. Corresponds to [Long] in Kotlin.
     */
    Int64;

    val byteSize: Int
        get() = when (this) {
            Float32 -> 4
            Int32 -> 4
            Uint8 -> 1
            Int64 -> 8
        }
}
