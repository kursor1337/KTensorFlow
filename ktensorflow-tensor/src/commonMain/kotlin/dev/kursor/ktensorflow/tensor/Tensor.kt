package dev.kursor.ktensorflow.tensor

import dev.kursor.ktensorflow.tensor.impl.inferTensorShape
import dev.kursor.ktensorflow.tensor.impl.toByteArray
import dev.kursor.ktensorflow.tensor.physical.FloatTensor
import dev.kursor.ktensorflow.tensor.physical.IntTensor
import dev.kursor.ktensorflow.tensor.physical.LongTensor
import dev.kursor.ktensorflow.tensor.physical.UByteTensor

/**
 * Represents a [Tensor] - multidimensional array of data
 * 
 * @param T The type of the data.
 */
interface Tensor<T : Any> {

    /**
     * Data type of the [Tensor]
     */
    val dataType: TensorDataType<T>

    /**
     * Shape of the [Tensor]
     * @see TensorShape
     */
    val shape: TensorShape

    /**
     * Gets a typed element from this [Tensor]
     * 
     * @param index - coordinates of the element
     */
    operator fun get(index: IntArray): T

    /**
     * Sets a typed element to this [Tensor]
     * 
     * @param index - coordinates of the element
     * @param value - value to set
     */
    operator fun set(index: IntArray, value: T)

    /**
     * Gets a typed element from this [Tensor] using its flat index.
     *
     * @param index the linear index of the element
     */
    fun getFlat(index: Int): T

    /**
     * Sets a typed element in this [Tensor] using its flattened index.
     *
     * @param index - the 1D index of the element
     * @param value - value to set
     */
    fun setFlat(index: Int, value: T)

    /**
     * Converts this [Tensor] to a [PhysicalTensor].
     *
     * If this tensor is already a [PhysicalTensor], it may return itself.
     * Otherwise, it creates a new [PhysicalTensor] containing the same data.
     *
     * @return A [PhysicalTensor] representation of this tensor.
     */
    fun toPhysical(): PhysicalTensor<T>
}

/**
 * Gets a typed element from this [Tensor]
 *
 * @param index - coordinates of the element
 */
operator fun <T : Any> Tensor<T>.get(vararg index: Int): T {
    return this[index]
}

/**
 * Sets a typed element to this [Tensor]
 *
 * @param index - coordinates of the element
 * @param value - value to set
 */
operator fun <T : Any> Tensor<T>.set(vararg index: Int, value: T) {
    this[index] = value
}

/**
 * Creates a [Tensor] with the specified data type, shape and data.
 * 
 *
 * @param dataType - data type of the [Tensor]
 * @param shape - shape of the [Tensor]
 * @param data - raw data of the [Tensor]. Initialized to ByteArray of zeros by default.
 */
@Suppress("UNCHECKED_CAST")
fun <T : Any> Tensor(
    dataType: TensorDataType<T>,
    shape: TensorShape,
    data: ByteArray = ByteArray(shape.flatSize * dataType.byteSize)
): PhysicalTensor<T> = when (dataType) {
    TensorDataType.Float32 -> FloatTensor(shape, data)
    TensorDataType.Int32 -> IntTensor(shape, data)
    TensorDataType.UInt8 -> UByteTensor(shape, data)
    TensorDataType.Int64 -> LongTensor(shape, data)
} as PhysicalTensor<T>

/**
 * Creates a [Tensor] with the specified shape and data.
 *
 * @param T - type of the data
 * @param shape - shape of the [Tensor]
 * @param data - raw data of the [Tensor]. Initialized to ByteArray of zeros by default.
 */
inline fun <reified T : Any> Tensor(
    shape: TensorShape,
    data: ByteArray = ByteArray(shape.flatSize * TensorDataType.of<T>().byteSize)
): PhysicalTensor<T> = Tensor(TensorDataType.of<T>(), shape, data)

/**
 * Creates a [Tensor] with the specified data type and data.
 * Data can be represented as a multidimensional array of types [Float], [Int], [UByte], [Long].
 * For example: Array<Array<Array<FloatArray>>>
 * Boxed arrays (for example Array<Float>) are not supported,
 * use primitive arrays (like FloatArray) instead.
 * 
 * @param dataType - data type of the [Tensor]
 * @param data - raw data of the [Tensor]
 */
@Suppress("UNCHECKED_CAST")
fun <T : Any> Tensor(
    dataType: TensorDataType<T>,
    data: Any
): PhysicalTensor<T> {
    val shape = inferTensorShape(data)
    return when (dataType) {
        TensorDataType.Float32 -> {
            FloatTensor(shape, data.toByteArray(dataType, shape))
        }

        TensorDataType.Int32 -> {
            IntTensor(shape, data.toByteArray(dataType, shape))
        }

        TensorDataType.UInt8 -> {
            UByteTensor(shape, data.toByteArray(dataType, shape))
        }

        TensorDataType.Int64 -> {
            LongTensor(shape, data.toByteArray(dataType, shape))
        }
    } as PhysicalTensor<T>
}

/**
 * Creates a [Tensor] with the specified data type and data.
 * Data can be represented as a multidimensional array of types [Float], [Int], [UByte], [Long].
 * For example: Array<Array<Array<FloatArray>>>
 * Boxed arrays (for example Array<Float>) are not supported,
 * use primitive arrays (like FloatArray) instead.
 *
 * @param T - type of the data
 * @param data - raw data of the [Tensor]
 */
inline fun <reified T : Any> Tensor(
    data: Any
): PhysicalTensor<T> {
    return Tensor(TensorDataType.of<T>(), data)
}