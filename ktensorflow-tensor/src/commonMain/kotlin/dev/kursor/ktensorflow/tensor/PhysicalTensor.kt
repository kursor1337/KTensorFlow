package dev.kursor.ktensorflow.tensor

import dev.kursor.ktensorflow.InternalKTensorFlowApi

import dev.kursor.ktensorflow.tensor.impl.toShapedAndTypedArray

/**
 * Represents a [Tensor] that is physically backed by a [ByteArray].
 *
 * Unlike tensor views, a [PhysicalTensor] contains the actual raw data
 * in memory, allowing for direct access and conversion to multidimensional arrays.
 *
 * @param T the type of the elements contained in this tensor.
 */
@SubclassOptInRequired(InternalKTensorFlowApi::class)
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
 * A tensor of rank N becomes N nested arrays with a primitive array innermost, for example
 * `Array<FloatArray>` for a 2D [Float] tensor or `UByteArray` for a 1D [UByte] tensor.
 * A scalar (rank 0) is returned as a one-element array.
 *
 * @param R - type of the array
 */
fun <R : Any> PhysicalTensor<*>.toArray(): R =
    (data.toShapedAndTypedArray(dataType, shape) as? R)
        ?: throw IllegalArgumentException("Unsupported data type: $dataType")