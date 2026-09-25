package dev.kursor.ktensorflow.tensor

import kotlin.jvm.JvmName

/**
 * Adds two [Tensor]s element-wise.
 *
 * @param other - [Tensor] to add
 */
@JvmName("floatTensorPlusFloatTensor")
operator fun Tensor<Float>.plus(other: Tensor<Float>): Tensor<Float> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a + b }
}

/**
 * Adds two [Tensor]s element-wise.
 *
 * @param other - [Tensor] to add
 */
@JvmName("floatTensorPlusIntTensor")
operator fun Tensor<Float>.plus(other: Tensor<Int>): Tensor<Float> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a + b }
}

/**
 * Adds two [Tensor]s element-wise.
 *
 * @param other - [Tensor] to add
 */
@JvmName("floatTensorPlusUByteTensor")
operator fun Tensor<Float>.plus(other: Tensor<UByte>): Tensor<Float> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a + b.toFloat() }
}

/**
 * Adds two [Tensor]s element-wise.
 *
 * @param other - [Tensor] to add
 */
@JvmName("floatTensorPlusLongTensor")
operator fun Tensor<Float>.plus(other: Tensor<Long>): Tensor<Float> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a + b }
}

/**
 * Adds two [Tensor]s element-wise.
 *
 * @param other - [Tensor] to add
 */
@JvmName("intTensorPlusFloatTensor")
operator fun Tensor<Int>.plus(other: Tensor<Float>): Tensor<Float> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a + b }
}

/**
 * Adds two [Tensor]s element-wise.
 *
 * @param other - [Tensor] to add
 */
@JvmName("intTensorPlusIntTensor")
operator fun Tensor<Int>.plus(other: Tensor<Int>): Tensor<Int> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a + b }
}

/**
 * Adds two [Tensor]s element-wise.
 *
 * @param other - [Tensor] to add
 */
@JvmName("intTensorPlusUByteTensor")
operator fun Tensor<Int>.plus(other: Tensor<UByte>): Tensor<Int> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a + b.toInt() }
}

/**
 * Adds two [Tensor]s element-wise.
 *
 * @param other - [Tensor] to add
 */
@JvmName("intTensorPlusLongTensor")
operator fun Tensor<Int>.plus(other: Tensor<Long>): Tensor<Long> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a + b }
}

/**
 * Adds two [Tensor]s element-wise.
 *
 * @param other - [Tensor] to add
 */
@JvmName("uByteTensorPlusFloatTensor")
operator fun Tensor<UByte>.plus(other: Tensor<Float>): Tensor<Float> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a.toFloat() + b }
}

/**
 * Adds two [Tensor]s element-wise.
 *
 * @param other - [Tensor] to add
 */
@JvmName("uByteTensorPlusIntTensor")
operator fun Tensor<UByte>.plus(other: Tensor<Int>): Tensor<Int> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a.toInt() + b }
}

/**
 * Adds two [Tensor]s element-wise.
 *
 * The result wraps around on overflow, as [UByte] arithmetic does in Kotlin:
 * `200 + 100` gives `44`. Convert with [toFloatTensor] first to keep the full range.
 *
 * @param other - [Tensor] to add
 */
@JvmName("uByteTensorPlusUByteTensor")
operator fun Tensor<UByte>.plus(other: Tensor<UByte>): Tensor<UByte> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> (a + b).toUByte() }
}

/**
 * Adds two [Tensor]s element-wise.
 *
 * @param other - [Tensor] to add
 */
@JvmName("uByteTensorPlusLongTensor")
operator fun Tensor<UByte>.plus(other: Tensor<Long>): Tensor<Long> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a.toLong() + b }
}

/**
 * Adds two [Tensor]s element-wise.
 *
 * @param other - [Tensor] to add
 */
@JvmName("longTensorPlusFloatTensor")
operator fun Tensor<Long>.plus(other: Tensor<Float>): Tensor<Float> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a + b }
}

/**
 * Adds two [Tensor]s element-wise.
 *
 * @param other - [Tensor] to add
 */
@JvmName("longTensorPlusIntTensor")
operator fun Tensor<Long>.plus(other: Tensor<Int>): Tensor<Long> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a + b }
}

/**
 * Adds two [Tensor]s element-wise.
 *
 * @param other - [Tensor] to add
 */
@JvmName("longTensorPlusUByteTensor")
operator fun Tensor<Long>.plus(other: Tensor<UByte>): Tensor<Long> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a + b.toLong() }
}

/**
 * Adds two [Tensor]s element-wise.
 *
 * @param other - [Tensor] to add
 */
@JvmName("longTensorPlusLongTensor")
operator fun Tensor<Long>.plus(other: Tensor<Long>): Tensor<Long> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a + b }
}

/**
 * Adds a number to all elements of this tensor
 *
 * @param other - number to add
 */
@JvmName("floatTensorPlusFloat")
operator fun Tensor<Float>.plus(other: Float): Tensor<Float> = map { it + other }

/**
 * Adds a number to all elements of this tensor
 *
 * @param other - number to add
 */
@JvmName("floatTensorPlusInt")
operator fun Tensor<Float>.plus(other: Int): Tensor<Float> = map { it + other }

/**
 * Adds a number to all elements of this tensor
 *
 * @param other - number to add
 */
@JvmName("floatTensorPlusUByte")
operator fun Tensor<Float>.plus(other: UByte): Tensor<Float> = map { it + other.toFloat() }

/**
 * Adds a number to all elements of this tensor
 *
 * @param other - number to add
 */
@JvmName("floatTensorPlusLong")
operator fun Tensor<Float>.plus(other: Long): Tensor<Float> = map { it + other }

/**
 * Adds a number to all elements of this tensor
 *
 * @param other - number to add
 */
@JvmName("intTensorPlusFloat")
operator fun Tensor<Int>.plus(other: Float): Tensor<Float> = map { it + other }

/**
 * Adds a number to all elements of this tensor
 *
 * @param other - number to add
 */
@JvmName("intTensorPlusInt")
operator fun Tensor<Int>.plus(other: Int): Tensor<Int> = map { it + other }

/**
 * Adds a number to all elements of this tensor
 *
 * @param other - number to add
 */
@JvmName("intTensorPlusUByte")
operator fun Tensor<Int>.plus(other: UByte): Tensor<Int> = map { it + other.toInt() }

/**
 * Adds a number to all elements of this tensor
 *
 * @param other - number to add
 */
@JvmName("intTensorPlusLong")
operator fun Tensor<Int>.plus(other: Long): Tensor<Long> = map { it + other }

/**
 * Adds a number to all elements of this tensor
 *
 * @param other - number to add
 */
@JvmName("uByteTensorPlusFloat")
operator fun Tensor<UByte>.plus(other: Float): Tensor<Float> = map { it.toFloat() + other }

/**
 * Adds a number to all elements of this tensor
 *
 * @param other - number to add
 */
@JvmName("uByteTensorPlusInt")
operator fun Tensor<UByte>.plus(other: Int): Tensor<Int> = map { it.toInt() + other }

/**
 * Adds a number to all elements of this tensor
 *
 * The result wraps around on overflow, as [UByte] arithmetic does in Kotlin:
 * `200 + 100` gives `44`. Convert with [toFloatTensor] first to keep the full range.
 *
 * @param other - number to add
 */
@JvmName("uByteTensorPlusUByte")
operator fun Tensor<UByte>.plus(other: UByte): Tensor<UByte> = map { (it + other).toUByte() }

/**
 * Adds a number to all elements of this tensor
 *
 * @param other - number to add
 */
@JvmName("uByteTensorPlusLong")
operator fun Tensor<UByte>.plus(other: Long): Tensor<Long> = map { it.toLong() + other }

/**
 * Adds a number to all elements of this tensor
 *
 * @param other - number to add
 */
@JvmName("longTensorPlusFloat")
operator fun Tensor<Long>.plus(other: Float): Tensor<Float> = map { it + other }

/**
 * Adds a number to all elements of this tensor
 *
 * @param other - number to add
 */
@JvmName("longTensorPlusInt")
operator fun Tensor<Long>.plus(other: Int): Tensor<Long> = map { it + other }

/**
 * Adds a number to all elements of this tensor
 *
 * @param other - number to add
 */
@JvmName("longTensorPlusUByte")
operator fun Tensor<Long>.plus(other: UByte): Tensor<Long> = map { it + other.toLong() }

/**
 * Adds a number to all elements of this tensor
 *
 * @param other - number to add
 */
@JvmName("longTensorPlusLong")
operator fun Tensor<Long>.plus(other: Long): Tensor<Long> = map { it + other }

/**
 * Subtracts one [Tensor] from another element-wise.
 *
 * @param other - [Tensor] to subtract
 */
@JvmName("floatTensorMinusFloatTensor")
operator fun Tensor<Float>.minus(other: Tensor<Float>): Tensor<Float> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a - b }
}

/**
 * Subtracts one [Tensor] from another element-wise.
 *
 * @param other - [Tensor] to subtract
 */
@JvmName("floatTensorMinusIntTensor")
operator fun Tensor<Float>.minus(other: Tensor<Int>): Tensor<Float> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a - b }
}

/**
 * Subtracts one [Tensor] from another element-wise.
 *
 * @param other - [Tensor] to subtract
 */
@JvmName("floatTensorMinusUByteTensor")
operator fun Tensor<Float>.minus(other: Tensor<UByte>): Tensor<Float> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a - b.toFloat() }
}

/**
 * Subtracts one [Tensor] from another element-wise.
 *
 * @param other - [Tensor] to subtract
 */
@JvmName("floatTensorMinusLongTensor")
operator fun Tensor<Float>.minus(other: Tensor<Long>): Tensor<Float> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a - b }
}

/**
 * Subtracts one [Tensor] from another element-wise.
 *
 * @param other - [Tensor] to subtract
 */
@JvmName("intTensorMinusFloatTensor")
operator fun Tensor<Int>.minus(other: Tensor<Float>): Tensor<Float> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a - b }
}

/**
 * Subtracts one [Tensor] from another element-wise.
 *
 * @param other - [Tensor] to subtract
 */
@JvmName("intTensorMinusIntTensor")
operator fun Tensor<Int>.minus(other: Tensor<Int>): Tensor<Int> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a - b }
}

/**
 * Subtracts one [Tensor] from another element-wise.
 *
 * @param other - [Tensor] to subtract
 */
@JvmName("intTensorMinusUByteTensor")
operator fun Tensor<Int>.minus(other: Tensor<UByte>): Tensor<Int> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a - b.toInt() }
}

/**
 * Subtracts one [Tensor] from another element-wise.
 *
 * @param other - [Tensor] to subtract
 */
@JvmName("intTensorMinusLongTensor")
operator fun Tensor<Int>.minus(other: Tensor<Long>): Tensor<Long> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a - b }
}

/**
 * Subtracts one [Tensor] from another element-wise.
 *
 * @param other - [Tensor] to subtract
 */
@JvmName("uByteTensorMinusFloatTensor")
operator fun Tensor<UByte>.minus(other: Tensor<Float>): Tensor<Float> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a.toFloat() - b }
}

/**
 * Subtracts one [Tensor] from another element-wise.
 *
 * @param other - [Tensor] to subtract
 */
@JvmName("uByteTensorMinusIntTensor")
operator fun Tensor<UByte>.minus(other: Tensor<Int>): Tensor<Int> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a.toInt() - b }
}

/**
 * Subtracts one [Tensor] from another element-wise.
 *
 * The result wraps around on overflow, as [UByte] arithmetic does in Kotlin:
 * `10 - 20` gives `246`. Convert with [toFloatTensor] first to keep the full range.
 *
 * @param other - [Tensor] to subtract
 */
@JvmName("uByteTensorMinusUByteTensor")
operator fun Tensor<UByte>.minus(other: Tensor<UByte>): Tensor<UByte> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> (a - b).toUByte() }
}

/**
 * Subtracts one [Tensor] from another element-wise.
 *
 * @param other - [Tensor] to subtract
 */
@JvmName("uByteTensorMinusLongTensor")
operator fun Tensor<UByte>.minus(other: Tensor<Long>): Tensor<Long> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a.toLong() - b }
}

/**
 * Subtracts one [Tensor] from another element-wise.
 *
 * @param other - [Tensor] to subtract
 */
@JvmName("longTensorMinusFloatTensor")
operator fun Tensor<Long>.minus(other: Tensor<Float>): Tensor<Float> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a - b }
}

/**
 * Subtracts one [Tensor] from another element-wise.
 *
 * @param other - [Tensor] to subtract
 */
@JvmName("longTensorMinusIntTensor")
operator fun Tensor<Long>.minus(other: Tensor<Int>): Tensor<Long> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a - b }
}

/**
 * Subtracts one [Tensor] from another element-wise.
 *
 * @param other - [Tensor] to subtract
 */
@JvmName("longTensorMinusUByteTensor")
operator fun Tensor<Long>.minus(other: Tensor<UByte>): Tensor<Long> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a - b.toLong() }
}

/**
 * Subtracts one [Tensor] from another element-wise.
 *
 * @param other - [Tensor] to subtract
 */
@JvmName("longTensorMinusLongTensor")
operator fun Tensor<Long>.minus(other: Tensor<Long>): Tensor<Long> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a - b }
}

/**
 * Subtracts a number from all elements of this tensor
 *
 * @param other - number to subtract
 */
@JvmName("floatTensorMinusFloat")
operator fun Tensor<Float>.minus(other: Float): Tensor<Float> = map { it - other }

/**
 * Subtracts a number from all elements of this tensor
 *
 * @param other - number to subtract
 */
@JvmName("floatTensorMinusInt")
operator fun Tensor<Float>.minus(other: Int): Tensor<Float> = map { it - other }

/**
 * Subtracts a number from all elements of this tensor
 *
 * @param other - number to subtract
 */
@JvmName("floatTensorMinusUByte")
operator fun Tensor<Float>.minus(other: UByte): Tensor<Float> = map { it - other.toFloat() }

/**
 * Subtracts a number from all elements of this tensor
 *
 * @param other - number to subtract
 */
@JvmName("floatTensorMinusLong")
operator fun Tensor<Float>.minus(other: Long): Tensor<Float> = map { it - other }

/**
 * Subtracts a number from all elements of this tensor
 *
 * @param other - number to subtract
 */
@JvmName("intTensorMinusFloat")
operator fun Tensor<Int>.minus(other: Float): Tensor<Float> = map { it - other }

/**
 * Subtracts a number from all elements of this tensor
 *
 * @param other - number to subtract
 */
@JvmName("intTensorMinusInt")
operator fun Tensor<Int>.minus(other: Int): Tensor<Int> = map { it - other }

/**
 * Subtracts a number from all elements of this tensor
 *
 * @param other - number to subtract
 */
@JvmName("intTensorMinusUByte")
operator fun Tensor<Int>.minus(other: UByte): Tensor<Int> = map { it - other.toInt() }

/**
 * Subtracts a number from all elements of this tensor
 *
 * @param other - number to subtract
 */
@JvmName("intTensorMinusLong")
operator fun Tensor<Int>.minus(other: Long): Tensor<Long> = map { it - other }

/**
 * Subtracts a number from all elements of this tensor
 *
 * @param other - number to subtract
 */
@JvmName("uByteTensorMinusFloat")
operator fun Tensor<UByte>.minus(other: Float): Tensor<Float> = map { it.toFloat() - other }

/**
 * Subtracts a number from all elements of this tensor
 *
 * @param other - number to subtract
 */
@JvmName("uByteTensorMinusInt")
operator fun Tensor<UByte>.minus(other: Int): Tensor<Int> = map { it.toInt() - other }

/**
 * Subtracts a number from all elements of this tensor
 *
 * The result wraps around on overflow, as [UByte] arithmetic does in Kotlin:
 * `10 - 20` gives `246`. Convert with [toFloatTensor] first to keep the full range.
 *
 * @param other - number to subtract
 */
@JvmName("uByteTensorMinusUByte")
operator fun Tensor<UByte>.minus(other: UByte): Tensor<UByte> = map { (it - other).toUByte() }

/**
 * Subtracts a number from all elements of this tensor
 *
 * @param other - number to subtract
 */
@JvmName("uByteTensorMinusLong")
operator fun Tensor<UByte>.minus(other: Long): Tensor<Long> = map { it.toLong() - other }

/**
 * Subtracts a number from all elements of this tensor
 *
 * @param other - number to subtract
 */
@JvmName("longTensorMinusFloat")
operator fun Tensor<Long>.minus(other: Float): Tensor<Float> = map { it - other }

/**
 * Subtracts a number from all elements of this tensor
 *
 * @param other - number to subtract
 */
@JvmName("longTensorMinusInt")
operator fun Tensor<Long>.minus(other: Int): Tensor<Long> = map { it - other }

/**
 * Subtracts a number from all elements of this tensor
 *
 * @param other - number to subtract
 */
@JvmName("longTensorMinusUByte")
operator fun Tensor<Long>.minus(other: UByte): Tensor<Long> = map { it - other.toLong() }

/**
 * Subtracts a number from all elements of this tensor
 *
 * @param other - number to subtract
 */
@JvmName("longTensorMinusLong")
operator fun Tensor<Long>.minus(other: Long): Tensor<Long> = map { it - other }

/**
 * Multiplies one [Tensor] with another element-wise.
 *
 * @param other - [Tensor] to multiply with
 */
@JvmName("floatTensorTimesFloatTensor")
operator fun Tensor<Float>.times(other: Tensor<Float>): Tensor<Float> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a * b }
}

/**
 * Multiplies one [Tensor] with another element-wise.
 *
 * @param other - [Tensor] to multiply with
 */
@JvmName("floatTensorTimesIntTensor")
operator fun Tensor<Float>.times(other: Tensor<Int>): Tensor<Float> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a * b }
}

/**
 * Multiplies one [Tensor] with another element-wise.
 *
 * @param other - [Tensor] to multiply with
 */
@JvmName("floatTensorTimesUByteTensor")
operator fun Tensor<Float>.times(other: Tensor<UByte>): Tensor<Float> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a * b.toFloat() }
}

/**
 * Multiplies one [Tensor] with another element-wise.
 *
 * @param other - [Tensor] to multiply with
 */
@JvmName("floatTensorTimesLongTensor")
operator fun Tensor<Float>.times(other: Tensor<Long>): Tensor<Float> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a * b }
}

/**
 * Multiplies one [Tensor] with another element-wise.
 *
 * @param other - [Tensor] to multiply with
 */
@JvmName("intTensorTimesFloatTensor")
operator fun Tensor<Int>.times(other: Tensor<Float>): Tensor<Float> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a * b }
}

/**
 * Multiplies one [Tensor] with another element-wise.
 *
 * @param other - [Tensor] to multiply with
 */
@JvmName("intTensorTimesIntTensor")
operator fun Tensor<Int>.times(other: Tensor<Int>): Tensor<Int> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a * b }
}

/**
 * Multiplies one [Tensor] with another element-wise.
 *
 * @param other - [Tensor] to multiply with
 */
@JvmName("intTensorTimesUByteTensor")
operator fun Tensor<Int>.times(other: Tensor<UByte>): Tensor<Int> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a * b.toInt() }
}

/**
 * Multiplies one [Tensor] with another element-wise.
 *
 * @param other - [Tensor] to multiply with
 */
@JvmName("intTensorTimesLongTensor")
operator fun Tensor<Int>.times(other: Tensor<Long>): Tensor<Long> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a * b }
}

/**
 * Multiplies one [Tensor] with another element-wise.
 *
 * @param other - [Tensor] to multiply with
 */
@JvmName("uByteTensorTimesFloatTensor")
operator fun Tensor<UByte>.times(other: Tensor<Float>): Tensor<Float> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a.toFloat() * b }
}

/**
 * Multiplies one [Tensor] with another element-wise.
 *
 * @param other - [Tensor] to multiply with
 */
@JvmName("uByteTensorTimesIntTensor")
operator fun Tensor<UByte>.times(other: Tensor<Int>): Tensor<Int> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a.toInt() * b }
}

/**
 * Multiplies one [Tensor] with another element-wise.
 *
 * The result wraps around on overflow, as [UByte] arithmetic does in Kotlin:
 * `16 * 20` gives `64`. Convert with [toFloatTensor] first to keep the full range.
 *
 * @param other - [Tensor] to multiply with
 */
@JvmName("uByteTensorTimesUByteTensor")
operator fun Tensor<UByte>.times(other: Tensor<UByte>): Tensor<UByte> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> (a * b).toUByte() }
}

/**
 * Multiplies one [Tensor] with another element-wise.
 *
 * @param other - [Tensor] to multiply with
 */
@JvmName("uByteTensorTimesLongTensor")
operator fun Tensor<UByte>.times(other: Tensor<Long>): Tensor<Long> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a.toLong() * b }
}

/**
 * Multiplies one [Tensor] with another element-wise.
 *
 * @param other - [Tensor] to multiply with
 */
@JvmName("longTensorTimesFloatTensor")
operator fun Tensor<Long>.times(other: Tensor<Float>): Tensor<Float> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a.toFloat() * b }
}

/**
 * Multiplies one [Tensor] with another element-wise.
 *
 * @param other - [Tensor] to multiply with
 */
@JvmName("longTensorTimesIntTensor")
operator fun Tensor<Long>.times(other: Tensor<Int>): Tensor<Long> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a * b }
}

/**
 * Multiplies one [Tensor] with another element-wise.
 *
 * @param other - [Tensor] to multiply with
 */
@JvmName("longTensorTimesUByteTensor")
operator fun Tensor<Long>.times(other: Tensor<UByte>): Tensor<Long> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a * b.toLong() }
}

/**
 * Multiplies one [Tensor] with another element-wise.
 *
 * @param other - [Tensor] to multiply with
 */
@JvmName("longTensorTimesLongTensor")
operator fun Tensor<Long>.times(other: Tensor<Long>): Tensor<Long> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a * b }
}

/**
 * Multiplies all elements of the [Tensor] with a number.
 *
 * @param other - number to multiply with
 */
@JvmName("floatTensorTimesFloat")
operator fun Tensor<Float>.times(other: Float): Tensor<Float> = map { it * other }

/**
 * Multiplies all elements of the [Tensor] with a number.
 *
 * @param other - number to multiply with
 */
@JvmName("floatTensorTimesInt")
operator fun Tensor<Float>.times(other: Int): Tensor<Float> = map { it * other }

/**
 * Multiplies all elements of the [Tensor] with a number.
 *
 * @param other - number to multiply with
 */
@JvmName("floatTensorTimesUByte")
operator fun Tensor<Float>.times(other: UByte): Tensor<Float> = map { it * other.toFloat() }

/**
 * Multiplies all elements of the [Tensor] with a number.
 *
 * @param other - number to multiply with
 */
@JvmName("floatTensorTimesLong")
operator fun Tensor<Float>.times(other: Long): Tensor<Float> = map { it * other }

/**
 * Multiplies all elements of the [Tensor] with a number.
 *
 * @param other - number to multiply with
 */
@JvmName("intTensorTimesFloat")
operator fun Tensor<Int>.times(other: Float): Tensor<Float> = map { it * other }

/**
 * Multiplies all elements of the [Tensor] with a number.
 *
 * @param other - number to multiply with
 */
@JvmName("intTensorTimesInt")
operator fun Tensor<Int>.times(other: Int): Tensor<Int> = map { it * other }

/**
 * Multiplies all elements of the [Tensor] with a number.
 *
 * @param other - number to multiply with
 */
@JvmName("intTensorTimesUByte")
operator fun Tensor<Int>.times(other: UByte): Tensor<Int> = map { it * other.toInt() }

/**
 * Multiplies all elements of the [Tensor] with a number.
 *
 * @param other - number to multiply with
 */
@JvmName("intTensorTimesLong")
operator fun Tensor<Int>.times(other: Long): Tensor<Long> = map { it * other }

/**
 * Multiplies all elements of the [Tensor] with a number.
 *
 * @param other - number to multiply with
 */
@JvmName("uByteTensorTimesFloat")
operator fun Tensor<UByte>.times(other: Float): Tensor<Float> = map { it.toFloat() * other }

/**
 * Multiplies all elements of the [Tensor] with a number.
 *
 * @param other - number to multiply with
 */
@JvmName("uByteTensorTimesInt")
operator fun Tensor<UByte>.times(other: Int): Tensor<Int> = map { it.toInt() * other }

/**
 * Multiplies all elements of the [Tensor] with a number.
 *
 * The result wraps around on overflow, as [UByte] arithmetic does in Kotlin:
 * `16 * 20` gives `64`. Convert with [toFloatTensor] first to keep the full range.
 *
 * @param other - number to multiply with
 */
@JvmName("uByteTensorTimesUByte")
operator fun Tensor<UByte>.times(other: UByte): Tensor<UByte> = map { (it * other).toUByte() }

/**
 * Multiplies all elements of the [Tensor] with a number.
 *
 * @param other - number to multiply with
 */
@JvmName("uByteTensorTimesLong")
operator fun Tensor<UByte>.times(other: Long): Tensor<Long> = map { it.toLong() * other }

/**
 * Multiplies all elements of the [Tensor] with a number.
 *
 * @param other - number to multiply with
 */
@JvmName("longTensorTimesFloat")
operator fun Tensor<Long>.times(other: Float): Tensor<Float> = map { it * other }

/**
 * Multiplies all elements of the [Tensor] with a number.
 *
 * @param other - number to multiply with
 */
@JvmName("longTensorTimesInt")
operator fun Tensor<Long>.times(other: Int): Tensor<Long> = map { it * other }

/**
 * Multiplies all elements of the [Tensor] with a number.
 *
 * @param other - number to multiply with
 */
@JvmName("longTensorTimesUByte")
operator fun Tensor<Long>.times(other: UByte): Tensor<Long> = map { it * other.toLong() }

/**
 * Multiplies all elements of the [Tensor] with a number.
 *
 * @param other - number to multiply with
 */
@JvmName("longTensorTimesLong")
operator fun Tensor<Long>.times(other: Long): Tensor<Long> = map { it * other }

/**
 * Divides one [Tensor] with another element-wise.
 *
 * @param other - [Tensor] to divide with
 */
@JvmName("floatTensorDivFloatTensor")
operator fun Tensor<Float>.div(other: Tensor<Float>): Tensor<Float> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a / b }
}

/**
 * Divides one [Tensor] with another element-wise.
 *
 * @param other - [Tensor] to divide with
 */
@JvmName("floatTensorDivIntTensor")
operator fun Tensor<Float>.div(other: Tensor<Int>): Tensor<Float> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a / b }
}

/**
 * Divides one [Tensor] with another element-wise.
 *
 * @param other - [Tensor] to divide with
 */
@JvmName("floatTensorDivUByteTensor")
operator fun Tensor<Float>.div(other: Tensor<UByte>): Tensor<Float> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a / b.toFloat() }
}

/**
 * Divides one [Tensor] with another element-wise.
 *
 * @param other - [Tensor] to divide with
 */
@JvmName("floatTensorDivLongTensor")
operator fun Tensor<Float>.div(other: Tensor<Long>): Tensor<Float> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a / b }
}

/**
 * Divides one [Tensor] with another element-wise.
 *
 * @param other - [Tensor] to divide with
 */
@JvmName("intTensorDivFloatTensor")
operator fun Tensor<Int>.div(other: Tensor<Float>): Tensor<Float> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a / b }
}

/**
 * Divides one [Tensor] with another element-wise.
 *
 * @param other - [Tensor] to divide with
 */
@JvmName("intTensorDivIntTensor")
operator fun Tensor<Int>.div(other: Tensor<Int>): Tensor<Int> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a / b }
}

/**
 * Divides one [Tensor] with another element-wise.
 *
 * @param other - [Tensor] to divide with
 */
@JvmName("intTensorDivUByteTensor")
operator fun Tensor<Int>.div(other: Tensor<UByte>): Tensor<Int> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a / b.toInt() }
}

/**
 * Divides one [Tensor] with another element-wise.
 *
 * @param other - [Tensor] to divide with
 */
@JvmName("intTensorDivLongTensor")
operator fun Tensor<Int>.div(other: Tensor<Long>): Tensor<Long> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a / b }
}

/**
 * Divides one [Tensor] with another element-wise.
 *
 * @param other - [Tensor] to divide with
 */
@JvmName("uByteTensorDivFloatTensor")
operator fun Tensor<UByte>.div(other: Tensor<Float>): Tensor<Float> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a.toFloat() / b }
}

/**
 * Divides one [Tensor] with another element-wise.
 *
 * @param other - [Tensor] to divide with
 */
@JvmName("uByteTensorDivIntTensor")
operator fun Tensor<UByte>.div(other: Tensor<Int>): Tensor<Int> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a.toInt() / b }
}

/**
 * Divides one [Tensor] with another element-wise.
 *
 * @param other - [Tensor] to divide with
 */
@JvmName("uByteTensorDivUByteTensor")
operator fun Tensor<UByte>.div(other: Tensor<UByte>): Tensor<UByte> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> (a / b).toUByte() }
}

/**
 * Divides one [Tensor] with another element-wise.
 *
 * @param other - [Tensor] to divide with
 */
@JvmName("uByteTensorDivLongTensor")
operator fun Tensor<UByte>.div(other: Tensor<Long>): Tensor<Long> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a.toLong() / b }
}

/**
 * Divides one [Tensor] with another element-wise.
 *
 * @param other - [Tensor] to divide with
 */
@JvmName("longTensorDivFloatTensor")
operator fun Tensor<Long>.div(other: Tensor<Float>): Tensor<Float> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a / b }
}

/**
 * Divides one [Tensor] with another element-wise.
 *
 * @param other - [Tensor] to divide with
 */
@JvmName("longTensorDivIntTensor")
operator fun Tensor<Long>.div(other: Tensor<Int>): Tensor<Long> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a / b }
}

/**
 * Divides one [Tensor] with another element-wise.
 *
 * @param other - [Tensor] to divide with
 */
@JvmName("longTensorDivUByteTensor")
operator fun Tensor<Long>.div(other: Tensor<UByte>): Tensor<Long> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a / b.toLong() }
}

/**
 * Divides one [Tensor] with another element-wise.
 *
 * @param other - [Tensor] to divide with
 */
@JvmName("longTensorDivLongTensor")
operator fun Tensor<Long>.div(other: Tensor<Long>): Tensor<Long> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a / b }
}

/**
 * Divides all elements of one [Tensor] with a number.
 *
 * @param other - number to divide with
 */
@JvmName("floatTensorDivFloat")
operator fun Tensor<Float>.div(other: Float): Tensor<Float> = map { it / other }

/**
 * Divides all elements of one [Tensor] with a number.
 *
 * @param other - number to divide with
 */
@JvmName("floatTensorDivInt")
operator fun Tensor<Float>.div(other: Int): Tensor<Float> = map { it / other }

/**
 * Divides all elements of one [Tensor] with a number.
 *
 * @param other - number to divide with
 */
@JvmName("floatTensorDivUByte")
operator fun Tensor<Float>.div(other: UByte): Tensor<Float> = map { it / other.toFloat() }

/**
 * Divides all elements of one [Tensor] with a number.
 *
 * @param other - number to divide with
 */
@JvmName("floatTensorDivLong")
operator fun Tensor<Float>.div(other: Long): Tensor<Float> = map { it / other }

/**
 * Divides all elements of one [Tensor] with a number.
 *
 * @param other - number to divide with
 */
@JvmName("intTensorDivFloat")
operator fun Tensor<Int>.div(other: Float): Tensor<Float> = map { it / other }

/**
 * Divides all elements of one [Tensor] with a number.
 *
 * @param other - number to divide with
 */
@JvmName("intTensorDivInt")
operator fun Tensor<Int>.div(other: Int): Tensor<Int> = map { it / other }

/**
 * Divides all elements of one [Tensor] with a number.
 *
 * @param other - number to divide with
 */
@JvmName("intTensorDivUByte")
operator fun Tensor<Int>.div(other: UByte): Tensor<Int> = map { it / other.toInt() }

/**
 * Divides all elements of one [Tensor] with a number.
 *
 * @param other - number to divide with
 */
@JvmName("intTensorDivLong")
operator fun Tensor<Int>.div(other: Long): Tensor<Long> = map { it / other }

/**
 * Divides all elements of one [Tensor] with a number.
 *
 * @param other - number to divide with
 */
@JvmName("uByteTensorDivFloat")
operator fun Tensor<UByte>.div(other: Float): Tensor<Float> = map { it.toFloat() / other }

/**
 * Divides all elements of one [Tensor] with a number.
 *
 * @param other - number to divide with
 */
@JvmName("uByteTensorDivInt")
operator fun Tensor<UByte>.div(other: Int): Tensor<Int> = map { it.toInt() / other }

/**
 * Divides all elements of one [Tensor] with a number.
 *
 * @param other - number to divide with
 */
@JvmName("uByteTensorDivUByte")
operator fun Tensor<UByte>.div(other: UByte): Tensor<UByte> = map { (it / other).toUByte() }

/**
 * Divides all elements of one [Tensor] with a number.
 *
 * @param other - number to divide with
 */
@JvmName("uByteTensorDivLong")
operator fun Tensor<UByte>.div(other: Long): Tensor<Long> = map { it.toLong() / other }

/**
 * Divides all elements of one [Tensor] with a number.
 *
 * @param other - number to divide with
 */
@JvmName("longTensorDivFloat")
operator fun Tensor<Long>.div(other: Float): Tensor<Float> = map { it / other }

/**
 * Divides all elements of one [Tensor] with a number.
 *
 * @param other - number to divide with
 */
@JvmName("longTensorDivInt")
operator fun Tensor<Long>.div(other: Int): Tensor<Long> = map { it / other }

/**
 * Divides all elements of one [Tensor] with a number.
 *
 * @param other - number to divide with
 */
@JvmName("longTensorDivUByte")
operator fun Tensor<Long>.div(other: UByte): Tensor<Long> = map { it / other.toLong() }

/**
 * Divides all elements of one [Tensor] with a number.
 *
 * @param other - number to divide with
 */
@JvmName("longTensorDivLong")
operator fun Tensor<Long>.div(other: Long): Tensor<Long> = map { it / other }

/**
 * Returns the remainder of one [Tensor] divided by another element-wise.
 *
 * @param other - [Tensor] to divide with
 */
@JvmName("floatTensorRemFloatTensor")
operator fun Tensor<Float>.rem(other: Tensor<Float>): Tensor<Float> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a % b }
}

/**
 * Returns the remainder of one [Tensor] divided by another element-wise.
 *
 * @param other - [Tensor] to divide with
 */
@JvmName("floatTensorRemIntTensor")
operator fun Tensor<Float>.rem(other: Tensor<Int>): Tensor<Float> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a % b }
}

/**
 * Returns the remainder of one [Tensor] divided by another element-wise.
 *
 * @param other - [Tensor] to divide with
 */
@JvmName("floatTensorRemUByteTensor")
operator fun Tensor<Float>.rem(other: Tensor<UByte>): Tensor<Float> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a % b.toFloat() }
}

/**
 * Returns the remainder of one [Tensor] divided by another element-wise.
 *
 * @param other - [Tensor] to divide with
 */
@JvmName("floatTensorRemLongTensor")
operator fun Tensor<Float>.rem(other: Tensor<Long>): Tensor<Float> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a % b }
}

/**
 * Returns the remainder of one [Tensor] divided by another element-wise.
 *
 * @param other - [Tensor] to divide with
 */
@JvmName("intTensorRemFloatTensor")
operator fun Tensor<Int>.rem(other: Tensor<Float>): Tensor<Float> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a % b }
}

/**
 * Returns the remainder of one [Tensor] divided by another element-wise.
 *
 * @param other - [Tensor] to divide with
 */
@JvmName("intTensorRemIntTensor")
operator fun Tensor<Int>.rem(other: Tensor<Int>): Tensor<Int> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a % b }
}

/**
 * Returns the remainder of one [Tensor] divided by another element-wise.
 *
 * @param other - [Tensor] to divide with
 */
@JvmName("intTensorRemUByteTensor")
operator fun Tensor<Int>.rem(other: Tensor<UByte>): Tensor<Int> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a % b.toInt() }
}

/**
 * Returns the remainder of one [Tensor] divided by another element-wise.
 *
 * @param other - [Tensor] to divide with
 */
@JvmName("intTensorRemLongTensor")
operator fun Tensor<Int>.rem(other: Tensor<Long>): Tensor<Long> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a % b }
}

/**
 * Returns the remainder of one [Tensor] divided by another element-wise.
 *
 * @param other - [Tensor] to divide with
 */
@JvmName("uByteTensorRemFloatTensor")
operator fun Tensor<UByte>.rem(other: Tensor<Float>): Tensor<Float> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a.toFloat() % b }
}

/**
 * Returns the remainder of one [Tensor] divided by another element-wise.
 *
 * @param other - [Tensor] to divide with
 */
@JvmName("uByteTensorRemIntTensor")
operator fun Tensor<UByte>.rem(other: Tensor<Int>): Tensor<Int> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a.toInt() % b }
}

/**
 * Returns the remainder of one [Tensor] divided by another element-wise.
 *
 * @param other - [Tensor] to divide with
 */
@JvmName("uByteTensorRemUByteTensor")
operator fun Tensor<UByte>.rem(other: Tensor<UByte>): Tensor<UByte> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> (a % b).toUByte() }
}

/**
 * Returns the remainder of one [Tensor] divided by another element-wise.
 *
 * @param other - [Tensor] to divide with
 */
@JvmName("uByteTensorRemLongTensor")
operator fun Tensor<UByte>.rem(other: Tensor<Long>): Tensor<Long> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a.toLong() % b }
}

/**
 * Returns the remainder of one [Tensor] divided by another element-wise.
 *
 * @param other - [Tensor] to divide with
 */
@JvmName("longTensorRemFloatTensor")
operator fun Tensor<Long>.rem(other: Tensor<Float>): Tensor<Float> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a % b }
}

/**
 * Returns the remainder of one [Tensor] divided by another element-wise.
 *
 * @param other - [Tensor] to divide with
 */
@JvmName("longTensorRemIntTensor")
operator fun Tensor<Long>.rem(other: Tensor<Int>): Tensor<Long> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a % b }
}

/**
 * Returns the remainder of one [Tensor] divided by another element-wise.
 *
 * @param other - [Tensor] to divide with
 */
@JvmName("longTensorRemUByteTensor")
operator fun Tensor<Long>.rem(other: Tensor<UByte>): Tensor<Long> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a % b.toLong() }
}

/**
 * Returns the remainder of one [Tensor] divided by another element-wise.
 *
 * @param other - [Tensor] to divide with
 */
@JvmName("longTensorRemLongTensor")
operator fun Tensor<Long>.rem(other: Tensor<Long>): Tensor<Long> {
    requireSameShape(other)
    return zipFlat(other) { a, b -> a % b }
}

/**
 * Returns the remainder of one [Tensor] divided by a number.
 *
 * @param other - number to divide with
 */
@JvmName("floatTensorRemFloat")
operator fun Tensor<Float>.rem(other: Float): Tensor<Float> = map { it % other }

/**
 * Returns the remainder of one [Tensor] divided by a number.
 *
 * @param other - number to divide with
 */
@JvmName("floatTensorRemInt")
operator fun Tensor<Float>.rem(other: Int): Tensor<Float> = map { it % other }

/**
 * Returns the remainder of one [Tensor] divided by a number.
 *
 * @param other - number to divide with
 */
@JvmName("floatTensorRemUByte")
operator fun Tensor<Float>.rem(other: UByte): Tensor<Float> = map { it % other.toFloat() }

/**
 * Returns the remainder of one [Tensor] divided by a number.
 *
 * @param other - number to divide with
 */
@JvmName("floatTensorRemLong")
operator fun Tensor<Float>.rem(other: Long): Tensor<Float> = map { it % other }

/**
 * Returns the remainder of one [Tensor] divided by a number.
 *
 * @param other - number to divide with
 */
@JvmName("intTensorRemFloat")
operator fun Tensor<Int>.rem(other: Float): Tensor<Float> = map { it % other }

/**
 * Returns the remainder of one [Tensor] divided by a number.
 *
 * @param other - number to divide with
 */
@JvmName("intTensorRemInt")
operator fun Tensor<Int>.rem(other: Int): Tensor<Int> = map { it % other }

/**
 * Returns the remainder of one [Tensor] divided by a number.
 *
 * @param other - number to divide with
 */
@JvmName("intTensorRemUByte")
operator fun Tensor<Int>.rem(other: UByte): Tensor<Int> = map { it % other.toInt() }

/**
 * Returns the remainder of one [Tensor] divided by a number.
 *
 * @param other - number to divide with
 */
@JvmName("intTensorRemLong")
operator fun Tensor<Int>.rem(other: Long): Tensor<Long> = map { it % other }

/**
 * Returns the remainder of one [Tensor] divided by a number.
 *
 * @param other - number to divide with
 */
@JvmName("uByteTensorRemFloat")
operator fun Tensor<UByte>.rem(other: Float): Tensor<Float> = map { it.toFloat() % other }

/**
 * Returns the remainder of one [Tensor] divided by a number.
 *
 * @param other - number to divide with
 */
@JvmName("uByteTensorRemInt")
operator fun Tensor<UByte>.rem(other: Int): Tensor<Int> = map { it.toInt() % other }

/**
 * Returns the remainder of one [Tensor] divided by a number.
 *
 * @param other - number to divide with
 */
@JvmName("uByteTensorRemUByte")
operator fun Tensor<UByte>.rem(other: UByte): Tensor<UByte> = map { (it % other).toUByte() }

/**
 * Returns the remainder of one [Tensor] divided by a number.
 *
 * @param other - number to divide with
 */
@JvmName("uByteTensorRemLong")
operator fun Tensor<UByte>.rem(other: Long): Tensor<Long> = map { it.toLong() % other }

/**
 * Returns the remainder of one [Tensor] divided by a number.
 *
 * @param other - number to divide with
 */
@JvmName("longTensorRemFloat")
operator fun Tensor<Long>.rem(other: Float): Tensor<Float> = map { it % other }

/**
 * Returns the remainder of one [Tensor] divided by a number.
 *
 * @param other - number to divide with
 */
@JvmName("longTensorRemInt")
operator fun Tensor<Long>.rem(other: Int): Tensor<Long> = map { it % other }

/**
 * Returns the remainder of one [Tensor] divided by a number.
 *
 * @param other - number to divide with
 */
@JvmName("longTensorRemUByte")
operator fun Tensor<Long>.rem(other: UByte): Tensor<Long> = map { it % other.toLong() }

/**
 * Returns the remainder of one [Tensor] divided by a number.
 *
 * @param other - number to divide with
 */
@JvmName("longTensorRemLong")
operator fun Tensor<Long>.rem(other: Long): Tensor<Long> = map { it % other }

/**
 * Increments all elements of the [Tensor] by 1.
 */
@JvmName("floatTensorInc")
operator fun Tensor<Float>.inc(): Tensor<Float> = map { it + 1 }

/**
 * Increments all elements of the [Tensor] by 1.
 */
@JvmName("intTensorInc")
operator fun Tensor<Int>.inc(): Tensor<Int> = map { it + 1 }

/**
 * Increments all elements of the [Tensor] by 1.
 */
@JvmName("uByteTensorInc")
operator fun Tensor<UByte>.inc(): Tensor<UByte> = map { (it + 1U).toUByte() }

/**
 * Increments all elements of the [Tensor] by 1.
 */
@JvmName("longTensorInc")
operator fun Tensor<Long>.inc(): Tensor<Long> = map { it + 1 }

/**
 * Decrements all elements of the [Tensor] by 1.
 */
@JvmName("floatTensorDec")
operator fun Tensor<Float>.dec(): Tensor<Float> = map { it - 1 }

/**
 * Decrements all elements of the [Tensor] by 1.
 */
@JvmName("intTensorDec")
operator fun Tensor<Int>.dec(): Tensor<Int> = map { it - 1 }

/**
 * Decrements all elements of the [Tensor] by 1.
 */
@JvmName("uByteTensorDec")
operator fun Tensor<UByte>.dec(): Tensor<UByte> = map { (it - 1U).toUByte() }

/**
 * Decrements all elements of the [Tensor] by 1.
 */
@JvmName("longTensorDec")
operator fun Tensor<Long>.dec(): Tensor<Long> = map { it - 1 }

/**
 * Returns the same [Tensor].
 */
@JvmName("floatTensorUnaryPlus")
operator fun Tensor<Float>.unaryPlus(): Tensor<Float> = map { it }

/**
 * Returns the same [Tensor].
 */
@JvmName("intTensorUnaryPlus")
operator fun Tensor<Int>.unaryPlus(): Tensor<Int> = map { it }

/**
 * Returns the same [Tensor].
 */
@JvmName("longTensorUnaryPlus")
operator fun Tensor<Long>.unaryPlus(): Tensor<Long> = map { it }

/**
 * Returns the [Tensor] with negated elements.
 */
@JvmName("floatTensorUnaryMinus")
operator fun Tensor<Float>.unaryMinus(): Tensor<Float> = map { -it }

/**
 * Returns the [Tensor] with negated elements.
 */
@JvmName("intTensorUnaryMinus")
operator fun Tensor<Int>.unaryMinus(): Tensor<Int> = map { -it }

/**
 * Returns the [Tensor] with negated elements.
 */
@JvmName("longTensorUnaryMinus")
operator fun Tensor<Long>.unaryMinus(): Tensor<Long> = map { -it }

/**
 * Проверяет, что формы совпадают.
 *
 * Поэлементные операции обходят индексы левого тензора и читают правый по тем же координатам.
 * При разных формах координата чаще всего остаётся в пределах плоского размера, поэтому
 * результат получался не ошибкой, а молча неверными числами.
 */
private fun Tensor<*>.requireSameShape(other: Tensor<*>) {
    require(shape == other.shape) {
        "Element-wise operations require tensors of the same shape, got $shape and ${other.shape}"
    }
}

/**
 * Поэлементно объединяет два тензора одной формы. Оба обходятся по плоскому индексу: у views
 * getFlat логический, так что порядок элементов совпадает с вложенным. Вложенный индекс с
 * пересчётом через страйды на каждый элемент был медленнее на 12-30%.
 */
private inline fun <A : Any, B : Any, reified R : Any> Tensor<A>.zipFlat(
    other: Tensor<B>,
    transform: (A, B) -> R
): Tensor<R> {
    val result = Tensor(TensorDataType.of<R>(), shape)
    for (i in 0 until shape.flatSize) {
        result.setFlat(i, transform(getFlat(i), other.getFlat(i)))
    }
    return result
}
