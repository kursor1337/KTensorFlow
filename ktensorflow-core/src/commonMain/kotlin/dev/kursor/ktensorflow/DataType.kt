package dev.kursor.ktensorflow

/**
 * Data type of a model tensor, as reported in [ModelMeta].
 *
 * It covers every type TensorFlow Lite can report on either platform, so the metadata of any
 * model can be read. Tensors of the types without a [dev.kursor.ktensorflow.tensor.Tensor]
 * counterpart can still be run through the raw `ByteArray` overloads of [Interpreter.run].
 * [Float16], [BFloat16] and [Float64] are reported only on iOS, whose runtime exposes them;
 * [String] is reported only on Android.
 *
 * @property byteSize Size of one element in bytes, or 0 for [String], whose elements have a
 * variable length.
 */
enum class DataType(val byteSize: Int) {
    /** 32-bit floating point. */
    Float32(4),

    /** 32-bit signed integer. */
    Int32(4),

    /** 64-bit signed integer. */
    Int64(8),

    /** 8-bit unsigned integer, used by uint8-quantized models. */
    UInt8(1),

    /** 8-bit signed integer, used by fully int8-quantized models. */
    Int8(1),

    /** 16-bit signed integer. */
    Int16(2),

    /** 16-bit half-precision floating point. */
    Float16(2),

    /** 16-bit bfloat16 floating point. */
    BFloat16(2),

    /** 64-bit floating point. */
    Float64(8),

    /** Boolean, one byte per element. */
    Bool(1),

    /** Variable-length string. */
    String(0)
}
