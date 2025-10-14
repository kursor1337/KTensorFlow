package dev.kursor.ktensorflow.tensor.impl

import java.lang.reflect.Array

// Using reflection to build an array because Array builds boxed array
// and we need primitive arrays
// If we don't do this, there will be class cast exception when we cast this to multidimensional array
@OptIn(ExperimentalUnsignedTypes::class)
internal actual fun reshapeArray(flat: Any, dimentions: IntArray): Any {
    require(dimentions.isNotEmpty())



    var index = 0

    // Determine primitive component type
    val componentType: Class<*> = when (flat) {
        is FloatArray -> Float::class.javaPrimitiveType!!
        is IntArray -> Int::class.javaPrimitiveType!!
        is LongArray -> Long::class.javaPrimitiveType!!
        is UByteArray -> UByteArray::class.java
        else -> error("Unsupported type: ${flat::class}")
    }

    val dims = when (flat) {
        is UByteArray -> dimentions.dropLast(1).toIntArray()
        else -> dimentions
    }

    // Allocate nested array of the correct primitive type
    val result = Array.newInstance(componentType, *dims)

    fun fillArray(array: Any, shape: IntArray) {
        if (shape.size == 1) {
            val size = shape[0]
            when (flat) {
                is FloatArray -> {
                    val prim = array as FloatArray
                    for (i in 0 until size) prim[i] = flat[index++]
                }

                is IntArray -> {
                    val prim = array as IntArray
                    for (i in 0 until size) prim[i] = flat[index++]
                }

                is LongArray -> {
                    val prim = array as LongArray
                    for (i in 0 until size) prim[i] = flat[index++]
                }

                is UByteArray -> {
                    val prim = array as kotlin.Array<UByteArray>
                    for (i in 0 until size) {
                        prim[i] = UByteArray(dimentions.last()) { flat[index++] }
                    }
                }
            }
        } else {
            val subShape = shape.copyOfRange(1, shape.size)
            val len = shape[0]
            for (i in 0 until len) {
                fillArray(Array.get(array, i), subShape)
            }
        }
    }

    fillArray(result, dims)
    return result
}