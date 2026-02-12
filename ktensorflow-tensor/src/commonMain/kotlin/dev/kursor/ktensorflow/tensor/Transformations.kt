package dev.kursor.ktensorflow.tensor

import dev.kursor.ktensorflow.tensor.impl.toByteArray
import kotlin.jvm.JvmName
import kotlin.math.sqrt

/**
 * Iterates over all elements of the [Tensor].
 *
 * @param action The action to perform on each element.
 */
inline fun <T : Any> Tensor<T>.forEach(action: (T) -> Unit) {
    val index = IntArray(shape.rank)
    for (i in 0 until shape.flatSize) {
        action(get(index))
        index.incrementIndex(shape)
    }
}

/**
 * Iterates over all elements of the [Tensor] with their indices.
 *
 * @param action The action to perform on each element.
 */
inline fun <T : Any> Tensor<T>.forEachIndexed(action: (IntArray, T) -> Unit) {
    val index = IntArray(shape.rank)
    for (i in 0 until shape.flatSize) {
        action(index, get(index))
        index.incrementIndex(shape)
    }
}

/**
 * Creates a new [Tensor] by applying a transformation to each element of the [Tensor].
 *
 * @param dataType The data type of the new [Tensor].
 * @param transform The transformation to apply to each element.
 */
inline fun <T : Any, R : Any> Tensor<T>.map(
    dataType: TensorDataType<R>,
    crossinline transform: (T) -> R
): Tensor<R> {
    val result = Tensor(dataType, shape)
    forEachIndexed { index, value ->
        result[index] = transform(value)
    }
    return result
}

/**
 * Creates a new [Tensor] by applying a transformation to each element of the [Tensor].
 *
 * @param transform The transformation to apply to each element.
 */
inline fun <T : Any, reified R : Any> Tensor<T>.map(
    noinline transform: (T) -> R
): Tensor<R> = map(TensorDataType.of<R>(), transform)

/**
 * Creates a new [Tensor] by applying a transformation to each element of the [Tensor] with their indices.
 *
 * @param dataType The data type of the new [Tensor].
 * @param transform The transformation to apply to each element.
 */
inline fun <T : Any, R : Any> Tensor<T>.mapIndexed(
    dataType: TensorDataType<R>,
    crossinline transform: (IntArray, T) -> R
): Tensor<R> {
    val result = Tensor(dataType, shape)
    forEachIndexed { index, value ->
        result[index] = transform(index, value)
    }
    return result
}

/**
 * Creates a new [Tensor] by applying a transformation to each element of the [Tensor] with their indices.
 *
 * @param transform The transformation to apply to each element.
 */
inline fun <T : Any, reified R : Any> Tensor<T>.mapIndexed(
    noinline transform: (IntArray, T) -> R
): Tensor<R> = mapIndexed(TensorDataType.of<R>(), transform)

/**
 * Applies a transformation to each element of the [Tensor] in place (without copying).
 *
 * @param transform The transformation to apply to each element.
 */
inline fun <T : Any> Tensor<T>.mapInPlace(crossinline transform: (T) -> T) {
    forEachIndexed { index, value ->
        this[index] = transform(value)
    }
}

/**
 * Applies a transformation to each element of the [Tensor] in place (without copying) with their indices.
 *
 * @param transform The transformation to apply to each element.
 */
inline fun <T : Any> Tensor<T>.mapInPlaceIndexed(crossinline transform: (IntArray, T) -> T) {
    forEachIndexed { index, value ->
        this[index] = transform(index, value)
    }
}

/**
 * Creates a new [Tensor] out of this [Tensor] with the new shape.
 *
 * @param newShape The new shape of the [Tensor].
 */
fun <T : Any> Tensor<T>.reshape(newShape: TensorShape): Tensor<T> {
    require(newShape.flatSize == shape.flatSize) {
        "Cannot reshape tensor of shape $shape to $newShape (different element count)"
    }
    return Tensor(dataType, newShape, data.copyOf())
}

/**
 * Creates a new [Tensor] out of this [Tensor] with the flattened shape.
 */
fun <T : Any> Tensor<T>.flatten(): Tensor<T> =
    reshape(TensorShape(shape.flatSize))

/**
 * Creates a new [Tensor] out of this [Tensor] with the transposed shape.
 * Only 2D tensors can be transposed.
 *
 * @throws IllegalArgumentException if the [Tensor] is not 2D.
 */
fun <T : Any> Tensor<T>.transpose(): Tensor<T> {
    require(shape.rank == 2) { "Only 2D tensors can be transposed" }
    val (rows, cols) = shape.dimensions
    val result = Tensor(dataType, TensorShape(cols, rows))
    for (i in 0 until rows) {
        for (j in 0 until cols) {
            result[intArrayOf(j, i)] = this[intArrayOf(i, j)]
        }
    }
    return result
}

/**
 * Creates a new [Tensor] out of this [Tensor] with the specified ranges.
 *
 * @param ranges The ranges to use for slicing.
 */
fun <T : Any> Tensor<T>.slice(ranges: Array<IntRange>): Tensor<T> {
    require(ranges.size == shape.rank)
    val newShape = TensorShape(*ranges.map { it.last - it.first + 1 }.toIntArray())
    val result = Tensor(dataType, newShape)
    result.forEachIndexed { idx, _ ->
        val srcIndex = IntArray(shape.rank) { d -> ranges[d].first + idx[d] }
        result[idx] = this[srcIndex]
    }
    return result
}

fun <T : Any> Tensor<T>.squeeze(): Tensor<T> {
    val newShape = TensorShape(
        shape
            .dimensions
            .filter { it > 1 }
            .toIntArray()
    )
    return reshape(newShape)
}

/**
 * Returns the sum of all elements in the [Tensor].
 */
@JvmName("sumFloat")
fun Tensor<Float>.sum(): Float {
    var sum = 0f
    forEach { sum += it }
    return sum
}

/**
 * Returns the sum of all elements in the [Tensor].
 */
@JvmName("sumInt")
fun Tensor<Int>.sum(): Int {
    var sum = 0
    forEach { sum += it }
    return sum
}

/**
 * Returns the sum of all elements in the [Tensor].
 */
@JvmName("sumUByte")
fun Tensor<UByte>.sum(): UByte {
    var sum = 0U
    forEach { sum += it }
    return sum.toUByte()
}

/**
 * Returns the sum of all elements in the [Tensor].
 */
@JvmName("sumLong")
fun Tensor<Long>.sum(): Long {
    var sum = 0L
    forEach { sum += it }
    return sum
}

/**
 * Returns the average of all elements in the [Tensor].
 */
@JvmName("avgFloat")
fun Tensor<Float>.avg(): Float = sum() / shape.flatSize

/**
 * Returns the average of all elements in the [Tensor].
 */
@JvmName("avgInt")
fun Tensor<Int>.avg(): Int = sum() / shape.flatSize

/**
 * Returns the average of all elements in the [Tensor].
 */
@JvmName("avgUByte")
fun Tensor<UByte>.avg(): UByte = (sum() / shape.flatSize.toUInt()).toUByte()

/**
 * Returns the average of all elements in the [Tensor].
 */
@JvmName("avgLong")
fun Tensor<Long>.avg(): Long = sum() / shape.flatSize

/**
 * Returns the minimum of all elements in the [Tensor].
 */
@JvmName("minFloat")
fun Tensor<Float>.min(): Float {
    var min = Float.POSITIVE_INFINITY
    forEach { min = minOf(min, it) }
    return min
}

/**
 * Returns the minimum of all elements in the [Tensor].
 */
@JvmName("minInt")
fun Tensor<Int>.min(): Int {
    var min = Int.MAX_VALUE
    forEach { min = minOf(min, it) }
    return min
}

/**
 * Returns the minimum of all elements in the [Tensor].
 */
@JvmName("minUByte")
fun Tensor<UByte>.min(): UByte {
    var min = UByte.MAX_VALUE
    forEach { min = minOf(min, it) }
    return min
}

/**
 * Returns the minimum of all elements in the [Tensor].
 */
@JvmName("minLong")
fun Tensor<Long>.min(): Long {
    var min = Long.MAX_VALUE
    forEach { min = minOf(min, it) }
    return min
}

/**
 * Returns the maximum of all elements in the [Tensor].
 */
@JvmName("maxFloat")
fun Tensor<Float>.max(): Float {
    var max = Float.NEGATIVE_INFINITY
    forEach { max = maxOf(max, it) }
    return max
}

/**
 * Returns the maximum of all elements in the [Tensor].
 */
@JvmName("maxInt")
fun Tensor<Int>.max(): Int {
    var max = Int.MIN_VALUE
    forEach { max = maxOf(max, it) }
    return max
}

/**
 * Returns the maximum of all elements in the [Tensor].
 */
@JvmName("maxUByte")
fun Tensor<UByte>.max(): UByte {
    var max = UByte.MIN_VALUE
    forEach { max = maxOf(max, it) }
    return max
}

/**
 * Returns the maximum of all elements in the [Tensor].
 */
@JvmName("maxLong")
fun Tensor<Long>.max(): Long {
    var max = Long.MIN_VALUE
    forEach { max = maxOf(max, it) }
    return max
}

/**
 * Returns the index of the maximum element in the [Tensor].
 */
@JvmName("argMaxFloat")
fun Tensor<Float>.argmax(): IntArray {
    var max = Float.NEGATIVE_INFINITY
    var maxIndex = IntArray(shape.rank)
    forEachIndexed { index, value ->
        if (value > max) {
            max = value
            maxIndex = index
        }
    }
    return maxIndex
}

/**
 * Returns the index of the maximum element in the [Tensor].
 */
@JvmName("argMaxInt")
fun Tensor<Int>.argmax(): IntArray {
    var max = Int.MIN_VALUE
    var maxIndex = IntArray(shape.rank)
    forEachIndexed { index, value ->
        if (value > max) {
            max = value
            maxIndex = index
        }
    }
    return maxIndex
}

/**
 * Returns the index of the maximum element in the [Tensor].
 */
@JvmName("argMaxUByte")
fun Tensor<UByte>.argmax(): IntArray {
    var max = UByte.MIN_VALUE
    var maxIndex = IntArray(shape.rank)
    forEachIndexed { index, value ->
        if (value > max) {
            max = value
            maxIndex = index
        }
    }
    return maxIndex
}

/**
 * Returns the index of the maximum element in the [Tensor].
 */
@JvmName("argMaxLong")
fun Tensor<Long>.argmax(): IntArray {
    var max = Long.MIN_VALUE
    var maxIndex = IntArray(shape.rank)
    forEachIndexed { index, value ->
        if (value > max) {
            max = value
            maxIndex = index
        }
    }
    return maxIndex
}

/**
 * Returns the index of the minimum element in the [Tensor].
 */
@JvmName("argMinFloat")
fun Tensor<Float>.argmin(): IntArray {
    var min = Float.POSITIVE_INFINITY
    var minIndex = IntArray(shape.rank)
    forEachIndexed { index, value ->
        if (value < min) {
            min = value
            minIndex = index
        }
    }
    return minIndex
}

/**
 * Returns the index of the minimum element in the [Tensor].
 */
@JvmName("argMinInt")
fun Tensor<Int>.argmin(): IntArray {
    var min = Int.MAX_VALUE
    var minIndex = IntArray(shape.rank)
    forEachIndexed { index, value ->
        if (value < min) {
            min = value
            minIndex = index
        }
    }
    return minIndex
}

/**
 * Returns the index of the minimum element in the [Tensor].
 */
@JvmName("argMinUByte")
fun Tensor<UByte>.argmin(): IntArray {
    var min = UByte.MAX_VALUE
    var minIndex = IntArray(shape.rank)
    forEachIndexed { index, value ->
        if (value < min) {
            min = value
            minIndex = index
        }
    }
    return minIndex
}

/**
 * Returns the index of the minimum element in the [Tensor].
 */
@JvmName("argMinLong")
fun Tensor<Long>.argmin(): IntArray {
    var min = Long.MAX_VALUE
    var minIndex = IntArray(shape.rank)
    forEachIndexed { index, value ->
        if (value < min) {
            min = value
            minIndex = index
        }
    }
    return minIndex
}

/**
 * Normalizes the [Tensor] to the range [0, 1].
 */
@JvmName("normalizeFloat")
fun Tensor<Float>.normalize(): Tensor<Float> {
    var min = Float.POSITIVE_INFINITY
    var max = Float.NEGATIVE_INFINITY
    forEach {
        if (it < min) min = it
        if (it > max) max = it
    }
    return map { (it - min) / (max - min) }
}

/**
 * Converts the [Tensor] to a [Tensor<Float>].
 */
@Suppress("UNCHECKED_CAST")
fun <T : Any> Tensor<T>.toFloatTensor(): Tensor<Float> = when (dataType) {
    TensorDataType.UInt8 -> map { (it as UByte).toFloat() }
    TensorDataType.Float32 -> this as Tensor<Float>
    else -> map { (it as Number).toFloat() }
}

/**
 * Converts the [Tensor] to a [Tensor<Int>].
 */
@Suppress("UNCHECKED_CAST")
fun <T : Any> Tensor<T>.toIntTensor(): Tensor<Int> = when (dataType) {
    TensorDataType.UInt8 -> map { (it as UByte).toInt() }
    TensorDataType.Int32 -> this as Tensor<Int>
    else -> map { (it as Number).toInt() }
}

/**
 * Converts the [Tensor] to a [Tensor<Long>].
 */
@Suppress("UNCHECKED_CAST")
fun <T : Any> Tensor<T>.toLongTensor(): Tensor<Long> = when (dataType) {
    TensorDataType.UInt8 -> map { (it as UByte).toLong() }
    TensorDataType.Int64 -> this as Tensor<Long>
    else -> map { (it as Number).toLong() }
}

/**
 * Converts the [Tensor] to a [Tensor<UByte>].
 */
@Suppress("UNCHECKED_CAST")
fun <T : Any> Tensor<T>.toUByteTensor(): Tensor<UByte> = when (dataType) {
    TensorDataType.UInt8 -> this as Tensor<UByte>
    TensorDataType.Float32 -> map { (it as Float).toInt().toUByte() }
    TensorDataType.Int32 -> map { (it as Int).toUByte() }
    TensorDataType.Int64 -> map { (it as Long).toUByte() }
}

/**
 * Converts the [Tensor] to a [List<T>].
 *
 * @return The [List<T>] containing all elements of the [Tensor].
 */
fun <T : Any> Tensor<T>.toList(): List<T> {
    val list = ArrayList<T>(shape.flatSize)
    forEach { list.add(it) }
    return list
}

@OptIn(ExperimentalUnsignedTypes::class)
fun <R : Any> Tensor<*>.toFlatArray(): R {
    return when (dataType) {
        TensorDataType.Float32 -> {
            val array = FloatArray(shape.flatSize)
            forEachIndexed { index, value -> array[index.toFlatIndex(shape)] = value as Float }
            array
        }

        TensorDataType.Int32 -> {
            val array = IntArray(shape.flatSize)
            forEachIndexed { index, value -> array[index.toFlatIndex(shape)] = value as Int }
            array
        }

        TensorDataType.Int64 -> {
            val array = LongArray(shape.flatSize)
            forEachIndexed { index, value -> array[index.toFlatIndex(shape)] = value as Long }
            array
        }

        TensorDataType.UInt8 -> {
            val array = UByteArray(shape.flatSize)
            forEachIndexed { index, value -> array[index.toFlatIndex(shape)] = value as UByte }
            array
        }
    } as R
}
