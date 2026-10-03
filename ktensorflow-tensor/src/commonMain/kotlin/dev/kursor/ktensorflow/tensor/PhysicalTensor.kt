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

/**
 * Sets an element by its flat index, like [Tensor.setFlat], but without boxing the value.
 *
 * [Tensor.setFlat] is generic, so every call through [Tensor] or [PhysicalTensor] wraps the
 * value into an object. In a loop over every element, such as filling an input tensor from
 * camera pixels, that allocation is most of the cost: this function writes straight into
 * [PhysicalTensor.data] and is several times faster. A view has no data of its own; call
 * [Tensor.toPhysical] first.
 *
 * @param index The flat index of the element.
 * @param value The value to set.
 * @throws IndexOutOfBoundsException if [index] is outside the tensor.
 */
@Suppress("NOTHING_TO_INLINE")
inline fun PhysicalTensor<Float>.setFlatUnboxed(index: Int, value: Float) = data.writeFloat(index, value)

/**
 * Sets an element by its flat index, like [Tensor.setFlat], but without boxing the value.
 *
 * [Tensor.setFlat] is generic, so every call through [Tensor] or [PhysicalTensor] wraps the
 * value into an object. In a loop over every element that allocation is most of the cost: this
 * function writes straight into [PhysicalTensor.data] and is several times faster. A view has no
 * data of its own; call [Tensor.toPhysical] first.
 *
 * @param index The flat index of the element.
 * @param value The value to set.
 * @throws IndexOutOfBoundsException if [index] is outside the tensor.
 */
@Suppress("NOTHING_TO_INLINE")
inline fun PhysicalTensor<Int>.setFlatUnboxed(index: Int, value: Int) = data.writeInt(index, value)

/**
 * Sets an element by its flat index, like [Tensor.setFlat], but without boxing the value.
 *
 * [Tensor.setFlat] is generic, so every call through [Tensor] or [PhysicalTensor] wraps the
 * value into an object. In a loop over every element that allocation is most of the cost: this
 * function writes straight into [PhysicalTensor.data] and is several times faster. A view has no
 * data of its own; call [Tensor.toPhysical] first.
 *
 * @param index The flat index of the element.
 * @param value The value to set.
 * @throws IndexOutOfBoundsException if [index] is outside the tensor.
 */
@Suppress("NOTHING_TO_INLINE")
inline fun PhysicalTensor<Long>.setFlatUnboxed(index: Int, value: Long) = data.writeLong(index, value)

/**
 * Sets an element by its flat index, like [Tensor.setFlat], but without boxing the value.
 *
 * [Tensor.setFlat] is generic, so every call through [Tensor] or [PhysicalTensor] wraps the
 * value into an object. In a loop over every element that allocation is most of the cost: this
 * function writes straight into [PhysicalTensor.data] and is several times faster. A view has no
 * data of its own; call [Tensor.toPhysical] first.
 *
 * @param index The flat index of the element.
 * @param value The value to set.
 * @throws IndexOutOfBoundsException if [index] is outside the tensor.
 */
@Suppress("NOTHING_TO_INLINE")
inline fun PhysicalTensor<UByte>.setFlatUnboxed(index: Int, value: UByte) = data.writeUByte(index, value)
