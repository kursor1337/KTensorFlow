package dev.kursor.ktensorflow.tensor

import kotlin.reflect.KClass

internal fun IntArray.toFlatIndex(shape: TensorShape): Int {
    val strides = shape
        .dimensions
        .reversed()
        .runningFold(1) { acc, element ->
            acc * element
        }
        .reversed()
    return zip(strides).sumOf { it.first * it.second }
}

internal fun ByteArray.readFloat(index: Int): Float {
    return Float.fromBits(
        this[4 * index].toInt() or
                (this[4 * index + 1].toInt() shl 8) or
                (this[4 * index + 2].toInt() shl 16) or
                (this[4 * index + 3].toInt() shl 24)
    )
}

internal fun ByteArray.readInt(index: Int): Int {
    return this[4 * index].toInt() or
            (this[4 * index + 1].toInt() shl 8) or
            (this[4 * index + 2].toInt() shl 16) or
            (this[4 * index + 3].toInt() shl 24)
}

internal fun ByteArray.readUByte(index: Int): UByte {
    return this[index].toUByte()
}

internal fun ByteArray.readLong(index: Int): Long {
    return this[8 * index].toLong() or
            (this[8 * index + 1].toLong() shl 8) or
            (this[8 * index + 2].toLong() shl 16) or
            (this[8 * index + 3].toLong() shl 24) or
            (this[8 * index + 4].toLong() shl 32) or
            (this[8 * index + 5].toLong() shl 40) or
            (this[8 * index + 6].toLong() shl 48) or
            (this[8 * index + 7].toLong() shl 56)
}

internal fun ByteArray.writeFloat(index: Int, value: Float) {
    val bits = value.toBits()
    this[4 * index] = (bits and 0xFF).toByte()
    this[4 * index + 1] = ((bits shr 8) and 0xFF).toByte()
    this[4 * index + 2] = ((bits shr 16) and 0xFF).toByte()
    this[4 * index + 3] = ((bits shr 24) and 0xFF).toByte()
}

internal fun ByteArray.writeInt(index: Int, value: Int) {
    val bits = value
    this[4 * index] = (bits and 0xFF).toByte()
    this[4 * index + 1] = ((bits shr 8) and 0xFF).toByte()
    this[4 * index + 2] = ((bits shr 16) and 0xFF).toByte()
    this[4 * index + 3] = ((bits shr 24) and 0xFF).toByte()
}

internal fun ByteArray.writeUByte(index: Int, value: UByte) {
    this[index] = value.toByte()
}

internal fun ByteArray.writeLong(index: Int, value: Long) {
    this[8 * index] = (value and 0xFF).toByte()
    this[8 * index + 1] = ((value shr 8) and 0xFF).toByte()
    this[8 * index + 2] = ((value shr 16) and 0xFF).toByte()
    this[8 * index + 3] = ((value shr 24) and 0xFF).toByte()
    this[8 * index + 4] = ((value shr 32) and 0xFF).toByte()
    this[8 * index + 5] = ((value shr 40) and 0xFF).toByte()
    this[8 * index + 6] = ((value shr 48) and 0xFF).toByte()
    this[8 * index + 7] = ((value shr 56) and 0xFF).toByte()
}

val <T : Any> KClass<T>.byteSize: Int
    get() = when (this) {
        Float::class -> 4
        Int::class -> 4
        UByte::class -> 1
        Long::class -> 8
        else -> throw IllegalArgumentException("Unsupported type: $this")
    }