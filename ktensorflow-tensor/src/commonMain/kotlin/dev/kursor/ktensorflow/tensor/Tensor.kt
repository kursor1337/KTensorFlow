package dev.kursor.ktensorflow.tensor

import dev.kursor.ktensorflow.tensor.impl.inferTensorShape
import dev.kursor.ktensorflow.tensor.impl.toByteArray
import dev.kursor.ktensorflow.tensor.impl.toShapedAndTypedArray
import kotlin.reflect.KClass

sealed interface Tensor<T : Any> {

    val dataType: KClass<T>

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
    dataType: KClass<T>,
    shape: TensorShape,
    data: ByteArray = ByteArray(shape.flatSize * dataType.byteSize)
): Tensor<T> = when (dataType) {
    Float::class -> FloatTensor(shape, data)
    Int::class -> IntTensor(shape, data)
    UByte::class -> UByteTensor(shape, data)
    Long::class -> LongTensor(shape, data)
    else -> throw IllegalArgumentException("Unsupported data type: $dataType")
} as Tensor<T>

inline fun <reified T : Any> Tensor(
    shape: TensorShape,
    data: ByteArray = ByteArray(shape.flatSize * T::class.byteSize)
) = Tensor(T::class, shape, data)

@Suppress("UNCHECKED_CAST")
fun <T : Any> Tensor(
    dataType: KClass<T>,
    data: Any
): Tensor<T> {
    val shape = inferTensorShape(data)
    return when (dataType) {
        Float::class -> {
            FloatTensor(shape, data.toByteArray(Float::class, shape))
        }

        Int::class -> {
            IntTensor(shape, data.toByteArray(Int::class, shape))
        }

        UByte::class -> {
            UByteTensor(shape, data.toByteArray(UByte::class, shape))
        }

        Long::class -> {
            LongTensor(shape, data.toByteArray(Long::class, shape))
        }

        else -> throw IllegalArgumentException("Unsupported data type: $dataType")
    } as Tensor<T>
}

inline fun <reified T : Any> Tensor(
    data: Any
): Tensor<T> {
    return Tensor(T::class, data)
}

fun <R : Any> Tensor<*>.toArray(): R =
    (data.toShapedAndTypedArray(dataType, shape) as? R)
        ?: throw IllegalArgumentException("Unsupported data type: $dataType")