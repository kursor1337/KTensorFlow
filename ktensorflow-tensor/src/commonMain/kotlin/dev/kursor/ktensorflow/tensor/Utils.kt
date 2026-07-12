package dev.kursor.ktensorflow.tensor

fun IntArray.incrementIndex(shape: TensorShape) {
    for (d in size - 1 downTo 0) {
        val next = this[d] + 1
        if (next < shape.dimensions[d]) {
            this[d] = next
            break
        } else {
            this[d] = 0
        }
    }
}

fun IntArray.toFlatIndex(shape: TensorShape): Int {
    require(size == shape.rank) {
        "Index rank $size doesn't match tensor rank ${shape.rank}"
    }

    var flatIndex = 0
    var stride = 1
    for (i in shape.rank - 1 downTo 0) {
        flatIndex += this[i] * stride
        stride *= shape.dimensions[i]
    }
    return flatIndex
}

fun IntArray.toFlatIndex(strides: IntArray): Int {
    var flatIndex = 0
    for (i in indices) {
        flatIndex += this[i] * strides[i]
    }
    return flatIndex
}

fun Int.toNestedIndex(shape: TensorShape): IntArray {
    require(this in 0 until shape.flatSize) {
        "Flat index $this out of bounds for shape $shape"
    }

    val index = IntArray(shape.rank)
    var remainder = this
    for (i in shape.rank - 1 downTo 0) {
        index[i] = remainder % shape.dimensions[i]
        remainder /= shape.dimensions[i]
    }
    return index
}

fun TensorShape.strides(): IntArray {
    val strides = IntArray(rank)
    var currentStride = 1
    for (i in rank - 1 downTo 0) {
        strides[i] = currentStride
        currentStride *= dimensions[i]
    }
    return strides
}

internal fun ByteArray.readFloat(index: Int): Float {
    return Float.fromBits(readInt(index))
}

internal fun ByteArray.readInt(index: Int): Int {
    return (this[4 * index].toInt() and 0xFF) or
            ((this[4 * index + 1].toInt() and 0xFF) shl 8) or
            ((this[4 * index + 2].toInt() and 0xFF) shl 16) or
            ((this[4 * index + 3].toInt() and 0xFF) shl 24)
}

internal fun ByteArray.readUByte(index: Int): UByte {
    return this[index].toUByte()
}

internal fun ByteArray.readLong(index: Int): Long {
    return ((this[8 * index].toLong() and 0xFF) or
            ((this[8 * index + 1].toLong() and 0xFF) shl 8) or
            ((this[8 * index + 2].toLong() and 0xFF) shl 16) or
            ((this[8 * index + 3].toLong() and 0xFF) shl 24) or
            ((this[8 * index + 4].toLong() and 0xFF) shl 32) or
            ((this[8 * index + 5].toLong() and 0xFF) shl 40) or
            ((this[8 * index + 6].toLong() and 0xFF) shl 48) or
            ((this[8 * index + 7].toLong() and 0xFF) shl 56))
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