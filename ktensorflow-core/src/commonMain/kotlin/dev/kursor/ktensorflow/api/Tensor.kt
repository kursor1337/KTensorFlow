package dev.kursor.ktensorflow.api

import dev.kursor.ktensorflow.impl.TensorImpl
import dev.kursor.ktensorflow.impl.inferTensorDataType
import dev.kursor.ktensorflow.impl.inferTensorShape
import dev.kursor.ktensorflow.impl.toByteArray
import dev.kursor.ktensorflow.impl.toShapedAndTypedArray


/**
 * Represents a multi-dimensional array of data with a specific data type and shape.
 */
interface Tensor {
    /**
     * The data type of the tensor.
     */
    val dataType: TensorDataType

    /**
     * The shape of the tensor.
     */

    val shape: TensorShape

    /**
     * The data of the tensor as a byte array.
     */
    val data: ByteArray
}

/**
 * Returns the data of the tensor as a typed array.
 * The type [T] of the returned data must be compatible with the data type and shape of the tensor.
 * @param T The type of the array.
 * Must be one of the supported types: FloatArray, IntArray, ByteArray, LongArray.
 * or a nested array of these types, e.g. Array<FloatArray>, Array<Array<FloatArray>>.
 * @return The data of the tensor as a typed array.
 * @throws IllegalArgumentException If the type is not supported or if the data cannot be converted
 * to the specified type.
 */
fun <T : Any> Tensor.typedData(): T =
    (data.toShapedAndTypedArray(dataType, shape) as? T)
        ?: throw IllegalArgumentException(
            "Requested type is not compatible with tensor " +
                    "data type $dataType and shape $shape"
        )


/**
 * Creates a tensor with the specified data type and shape.
 * @param dataType The data type of the tensor.
 * @param shape The shape of the tensor.
 * @param data The data of the tensor as a byte array. If not provided, a zero-initialized byte array of the appropriate size will be created.
 */
fun Tensor(
    dataType: TensorDataType,
    shape: TensorShape,
    data: ByteArray = ByteArray(shape.flatSize * dataType.byteSize)
): Tensor {
    return TensorImpl(dataType, shape, data)
}

/**
 * Creates a tensor with the specified data.
 * The data type and shape are inferred from the data.
 * @param data The data of the tensor. Can be a multi- or single-dimensional primitive array,
 * e.g. `FloatArray`, `IntArray`, `ByteArray`, `LongArray`,
 * or a nested array of these types, e.g. Array<FloatArray>, Array<Array<FloatArray>>.
 */
fun Tensor(
    data: Any
): Tensor {
    val dataType = inferTensorDataType(data)
    val shape = inferTensorShape(data)
    val data = data.toByteArray(dataType, shape)
    return TensorImpl(dataType, shape, data)
}
