package dev.kursor.ktensorflow.tensor

import dev.kursor.ktensorflow.tensor.impl.inferTensorShape
import dev.kursor.ktensorflow.tensor.impl.toByteArray
import dev.kursor.ktensorflow.tensor.impl.toShapedAndTypedArray
import kotlin.reflect.KClass

sealed interface Tensor<T : Any> {

    val dataType: TensorDataType<T>

    val shape: TensorShape

    val data: ByteArray

    operator fun get(index: IntArray): T

    operator fun set(index: IntArray, value: T)
}

operator fun <T : Any> Tensor<T>.get(vararg index: Int): T {
    return this[index]
}

operator fun <T : Any> Tensor<T>.set(vararg index: Int, value: T) {
    this[index] = value
}

@Suppress("UNCHECKED_CAST")
fun <T : Any> Tensor(
    dataType: TensorDataType<T>,
    shape: TensorShape,
    data: ByteArray = ByteArray(shape.flatSize * dataType.byteSize)
): Tensor<T> = when (dataType) {
    TensorDataType.Float32 -> FloatTensor(shape, data)
    TensorDataType.Int32 -> IntTensor(shape, data)
    TensorDataType.UInt8 -> UByteTensor(shape, data)
    TensorDataType.Int64 -> LongTensor(shape, data)
} as Tensor<T>

inline fun <reified T : Any> Tensor(
    shape: TensorShape,
    data: ByteArray = ByteArray(shape.flatSize * TensorDataType.of<T>().byteSize)
): Tensor<T> = Tensor(TensorDataType.of<T>(), shape, data)

@Suppress("UNCHECKED_CAST")
fun <T : Any> Tensor(
    dataType: TensorDataType<T>,
    data: Any
): Tensor<T> {
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
    } as Tensor<T>
}

inline fun <reified T : Any> Tensor(
    data: Any
): Tensor<T> {
    return Tensor(TensorDataType.of<T>(), data)
}

fun <R : Any> Tensor<*>.toArray(): R =
    (data.toShapedAndTypedArray(dataType, shape) as? R)
        ?: throw IllegalArgumentException("Unsupported data type: $dataType")