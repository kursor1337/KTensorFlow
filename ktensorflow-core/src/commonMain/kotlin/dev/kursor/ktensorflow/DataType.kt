package dev.kursor.ktensorflow

enum class DataType(val byteSize: Int) {
    Float32(4),
    Int32(4),
    Int64(8),
    UInt8(1)
}