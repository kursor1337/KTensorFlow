package dev.kursor.ktensorflow.tensor

import kotlin.reflect.KClass

sealed interface TensorDataType<T : Any> {
    
    val kotlinType: KClass<T>
    val byteSize: Int

    data object Float32 : TensorDataType<Float> {
        override val kotlinType: KClass<Float> = Float::class
        override val byteSize: Int = 4
    }

    data object Int32 : TensorDataType<Int> {
        override val kotlinType: KClass<Int> = Int::class
        override val byteSize: Int = 4
    }

    data object UInt8 : TensorDataType<UByte> {
        override val kotlinType: KClass<UByte> = UByte::class
        override val byteSize: Int = 1
    }

    data object Int64 : TensorDataType<Long> {
        override val kotlinType: KClass<Long> = Long::class
        override val byteSize: Int = 8
    }

    companion object {

        fun <T : Any> of(klass: KClass<T>): TensorDataType<T> = when (klass) {
            Float::class -> Float32
            Int::class -> Int32
            UByte::class -> UInt8
            Long::class -> Int64
            else -> throw IllegalArgumentException("Unsupported tensor data type: ${klass}")
        } as TensorDataType<T>

        inline fun <reified T : Any> of(): TensorDataType<T> = of(T::class)
    }
}
