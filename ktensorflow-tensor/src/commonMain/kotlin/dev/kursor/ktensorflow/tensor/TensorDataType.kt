package dev.kursor.ktensorflow.tensor

import kotlin.reflect.KClass

/**
 * Represents a data type of a tensor.
 * @param T The kotlin type of the data.
 */
sealed interface TensorDataType<T : Any> {

    /**
     * Kotlin type corresponding to this TensorDataType
     */
    val kotlinType: KClass<T>

    /**
     * Size of the data type in bytes.
     */
    val byteSize: Int

    /**
     * Float32 data type.
     * Corresponds to the Float type in Kotlin
     */
    data object Float32 : TensorDataType<Float> {
        override val kotlinType: KClass<Float> = Float::class
        override val byteSize: Int = 4
    }

    /**
     * Int32 data type.
     * Corresponds to the Int type in Kotlin
     */
    data object Int32 : TensorDataType<Int> {
        override val kotlinType: KClass<Int> = Int::class
        override val byteSize: Int = 4
    }

    /**
     * UInt8 data type.
     * Corresponds to the UByte type in Kotlin
     */
    data object UInt8 : TensorDataType<UByte> {
        override val kotlinType: KClass<UByte> = UByte::class
        override val byteSize: Int = 1
    }

    /**
     * Int64 data type.
     * Corresponds to the Long type in Kotlin
     */
    data object Int64 : TensorDataType<Long> {
        override val kotlinType: KClass<Long> = Long::class
        override val byteSize: Int = 8
    }

    companion object {

        /**
         * Returns the TensorDataType corresponding to the given kotlin type.
         * Only supports Float, Int, UByte and Long type parameters
         *
         * @param klass The kotlin type to get the TensorDataType for.
         */
        fun <T : Any> of(klass: KClass<T>): TensorDataType<T> = when (klass) {
            Float::class -> Float32
            Int::class -> Int32
            UByte::class -> UInt8
            Long::class -> Int64
            else -> throw IllegalArgumentException("Unsupported tensor data type: ${klass}")
        } as TensorDataType<T>

        /**
         * Returns the TensorDataType corresponding to the given kotlin type.
         * Only supports Float, Int, UByte and Long type parameters
         *
         * @param T The kotlin type to get the TensorDataType for.
         */
        inline fun <reified T : Any> of(): TensorDataType<T> = of(T::class)
    }
}
