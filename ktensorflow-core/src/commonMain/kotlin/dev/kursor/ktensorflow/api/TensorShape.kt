package dev.kursor.ktensorflow.api

import kotlin.jvm.JvmInline

/**
 * Represents the shape of a tensor.
 */
@JvmInline
value class TensorShape(
    val dimensions: IntArray
) {

    /**
     * Returns the rank of the tensor, which is the number of dimensions.
     */
    val rank: Int
        get() = dimensions.size

    /**
     * Returns the size of the tensor if it was flattened to a single dimension.
     */
    val flatSize: Int
        get() = dimensions.reduce { acc, i -> acc * i }

    override fun toString(): String {
        return dimensions.joinToString(prefix = "(", separator = ", ", postfix = ")") { it.toString() }
    }
}

/**
 * Creates a new [TensorShape] with the specified dimensions.
 *
 * @param dimensions The dimensions of the tensor.
 */
inline fun TensorShape(vararg dimensions: Int) = TensorShape(dimensions)
