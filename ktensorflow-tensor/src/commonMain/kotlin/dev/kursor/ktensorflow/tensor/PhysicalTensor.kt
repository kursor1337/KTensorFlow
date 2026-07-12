package dev.kursor.ktensorflow.tensor

import dev.kursor.ktensorflow.tensor.impl.toShapedAndTypedArray

interface PhysicalTensor<T : Any> : Tensor<T> {
    /**
     * Raw data of the [Tensor]
     */
    val data: ByteArray

    override fun toPhysical(): PhysicalTensor<T> = this
}

/**
 * Converts this [PhysicalTensor] to a multidimensional array of type [R].
 *
 * @param R - type of the array
 */
fun <R : Any> PhysicalTensor<*>.toArray(): R =
    (data.toShapedAndTypedArray(dataType, shape) as? R)
        ?: throw IllegalArgumentException("Unsupported data type: $dataType")