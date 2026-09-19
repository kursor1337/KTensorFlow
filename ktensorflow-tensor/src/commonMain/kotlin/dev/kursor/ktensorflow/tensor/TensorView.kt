package dev.kursor.ktensorflow.tensor

/**
 * Represents a virtual or transformed view of another [Tensor].
 *
 * A [TensorView] does not typically store its own data but instead provides an interface
 * to access or manipulate the data of its [delegate] tensor. This is commonly used for
 * operations like slicing, reshaping, or transposing without copying the underlying memory.
 *
 * @param T The type of the elements contained within the tensor.
 */
interface TensorView<T : Any> : Tensor<T> {
    /**
     * The underlying [Tensor] that this view is based on.
     */
    val delegate: Tensor<T>

    override fun toPhysical(): PhysicalTensor<T> {
        val result = Tensor(dataType, shape)
        for (i in 0 until shape.flatSize) {
            result.setFlat(i, this.getFlat(i))
        }
        return result
    }
}