package dev.kursor.ktensorflow.tensor

/**
 * Represents the shape of a tensor.
 *
 * This is deliberately a regular class rather than a value class: the only property is an
 * [IntArray], whose own `equals` is identity, and a value class may not override `equals`.
 * Wrapping it inline would therefore make two identical shapes compare as different, which
 * silently lets element-wise operations run over tensors of different shapes.
 */
class TensorShape(
    val dimensions: IntArray
) {

    /**
     * Returns the rank of the tensor, which is the number of dimensions.
     */
    val rank: Int
        get() = dimensions.size

    /**
     * Returns the size of the tensor if it was flattened to a single dimension.
     *
     * A shape without dimensions describes a scalar and holds exactly one element, hence the
     * fold over an empty product: reducing would throw on it, and `squeeze()` produces such a
     * shape whenever every dimension of the source is 1.
     */
    val flatSize: Int
        get() = dimensions.fold(1) { acc, i -> acc * i }

    /**
     * Two shapes are equal when they describe the same dimensions.
     *
     * The comparison has to be spelled out: the single property of this value class is an
     * [IntArray], whose own `equals` is identity, so the generated implementation would
     * report two identical shapes as different.
     */
    override fun equals(other: Any?): Boolean =
        other is TensorShape && dimensions.contentEquals(other.dimensions)

    override fun hashCode(): Int = dimensions.contentHashCode()

    override fun toString(): String {
        return dimensions.joinToString(prefix = "(", separator = ", ", postfix = ")") { it.toString() }
    }
}

/**
 * Creates a new [TensorShape] with the specified dimensions.
 *
 * @param dimensions The dimensions of the tensor.
 */
fun TensorShape(vararg dimensions: Int) = TensorShape(dimensions)
