package dev.kursor.ktensorflow.tensor.impl

import dev.kursor.ktensorflow.tensor.readFloat
import dev.kursor.ktensorflow.tensor.readInt
import dev.kursor.ktensorflow.tensor.readLong

internal fun FloatArray.toByteArray(): ByteArray {
    val dest = ByteArray(this.size * 4)
    var i = 0
    for (f in this) {
        val bits = f.toBits()
        dest[i++] = (bits and 0xFF).toByte()
        dest[i++] = ((bits shr 8) and 0xFF).toByte()
        dest[i++] = ((bits shr 16) and 0xFF).toByte()
        dest[i++] = ((bits shr 24) and 0xFF).toByte()
    }
    return dest
}

internal fun IntArray.toByteArray(): ByteArray {
    val dest = ByteArray(this.size * 4)
    var i = 0
    for (v in this) {
        dest[i++] = (v and 0xFF).toByte()
        dest[i++] = ((v shr 8) and 0xFF).toByte()
        dest[i++] = ((v shr 16) and 0xFF).toByte()
        dest[i++] = ((v shr 24) and 0xFF).toByte()
    }
    return dest
}

internal fun LongArray.toByteArray(): ByteArray {
    val dest = ByteArray(this.size * 8)
    var i = 0
    for (v in this) {
        for (b in 0 until 8) {
            dest[i++] = ((v shr (b * 8)) and 0xFF).toByte()
        }
    }
    return dest
}

internal fun ByteArray.toFloatArray(): FloatArray {
    val dest = FloatArray(this.size / 4)
    for (i in dest.indices) {
        dest[i] = this.readFloat(i)
    }
    return dest
}

internal fun ByteArray.toIntArray(): IntArray {
    val dest = IntArray(this.size / 4)
    for (i in dest.indices) {
        dest[i] = this.readInt(i)
    }
    return dest
}

internal fun ByteArray.toLongArray(): LongArray {
    val dest = LongArray(this.size / 8)
    for (i in dest.indices) {
        dest[i] = this.readLong(i)
    }
    return dest
}
