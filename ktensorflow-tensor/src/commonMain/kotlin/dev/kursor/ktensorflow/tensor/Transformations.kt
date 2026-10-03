package dev.kursor.ktensorflow.tensor

import dev.kursor.ktensorflow.tensor.views.PermutedTensorView
import dev.kursor.ktensorflow.tensor.views.ReshapedTensorView
import dev.kursor.ktensorflow.tensor.views.SlicedTensorView
import kotlin.jvm.JvmName

/**
 * Iterates over all elements of the [Tensor].
 *
 * @param action The action to perform on each element.
 */
inline fun <T : Any> Tensor<T>.forEach(action: (T) -> Unit) {
    for (i in 0 until shape.flatSize) {
        action(getFlat(i))
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
        action(index, getFlat(i))
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
    val size = shape.flatSize
    for (i in 0 until size) {
        result.setFlat(i, transform(getFlat(i)))
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
    val index = IntArray(shape.rank)
    val size = shape.flatSize
    for (i in 0 until size) {
        result.setFlat(i, transform(index, getFlat(i)))
        index.incrementIndex(shape)
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
    val size = shape.flatSize
    for (i in 0 until size) {
        setFlat(i, transform(getFlat(i)))
    }
}

/**
 * Applies a transformation to each element of the [Tensor] in place (without copying) with their indices.
 *
 * @param transform The transformation to apply to each element.
 */
inline fun <T : Any> Tensor<T>.mapInPlaceIndexed(crossinline transform: (IntArray, T) -> T) {
    val index = IntArray(shape.rank)
    val size = shape.flatSize
    for (i in 0 until size) {
        setFlat(i, transform(index, getFlat(i)))
        index.incrementIndex(shape)
    }
}

/**
 * Creates a new [Tensor] out of this [Tensor] with the new shape.
 * Does not create a new [PhysicalTensor], instead creates a view of the original tensor
 *
 * @param newShape The new shape of the [Tensor].
 */
fun <T : Any> Tensor<T>.reshape(newShape: TensorShape): Tensor<T> {
    if (this.shape == newShape) return this
    return ReshapedTensorView(this, newShape)
}

/**
 * Creates a new [Tensor] out of this [Tensor] with the flattened shape.
 * Does not create a new [PhysicalTensor], instead creates a view of the original tensor
 */
fun <T : Any> Tensor<T>.flatten(): Tensor<T> =
    reshape(TensorShape(shape.flatSize))

/**
 * Creates a new [Tensor] out of this [Tensor] with the permuted axes.
 * Does not create a new [PhysicalTensor], instead creates a view of the original tensor
 */
fun <T : Any> Tensor<T>.permuted(vararg axes: Int): Tensor<T> {
    require(axes.size == shape.rank) { "Axes length must match tensor rank" }
    // Повтор оси не всегда выводит смещения за массив: на квадратном тензоре permuted(1, 1)
    // молча возвращал мусор, поэтому оси обязаны быть перестановкой 0 until rank
    require(axes.sorted() == (0 until shape.rank).toList()) {
        "Axes ${axes.contentToString()} are not a permutation of 0 until ${shape.rank}"
    }
    return PermutedTensorView(this, axes.copyOf())
}

/**
 * Creates a new [Tensor] out of this [Tensor] with the transposed shape.
 * Only 2D tensors can be transposed.
 * Does not create a new [PhysicalTensor], instead creates a view of the original tensor
 *
 * @throws IllegalArgumentException if the [Tensor] is not 2D.
 */
fun <T : Any> Tensor<T>.transpose(): Tensor<T> {
    require(shape.rank >= 2) { "Tensor must have at least 2 dimensions to be transposed" }
    val axes = IntArray(shape.rank) { it }
    axes[shape.rank - 1] = shape.rank - 2
    axes[shape.rank - 2] = shape.rank - 1
    return permuted(*axes)
}

/**
 * Creates a new [Tensor] out of this [Tensor] with the specified ranges.
 * Does not create a new [PhysicalTensor], instead creates a view of the original tensor
 * @param ranges The ranges to use for slicing.
 */
@JvmName("sliceArray")
fun <T : Any> Tensor<T>.slice(ranges: Array<IntRange>): Tensor<T> {
    return SlicedTensorView(this, ranges)
}

/**
 * Creates a new [Tensor] out of this [Tensor] with the specified ranges.
 * Does not create a new [PhysicalTensor], instead creates a view of the original tensor
 * @param ranges The ranges to use for slicing.
 */
fun <T : Any> Tensor<T>.slice(vararg ranges: IntRange): Tensor<T> =
    slice(arrayOf(*ranges))

/**
 * Creates a new [Tensor] out of this [Tensor] with the squeezed shape.
 * Does not create a new [PhysicalTensor], instead creates a view of the original tensor
 */
fun <T : Any> Tensor<T>.squeeze(): Tensor<T> = reshape(
    TensorShape(
        shape
            .dimensions
            // Убираются только единичные измерения: с условием `> 1` пропадали и нулевые, и
            // пустой тензор (0, 3) превращался в (3) с другим числом элементов
            .filter { it != 1 }
            .toIntArray()
    )
)

/**
 * Returns the sum of all elements in the [Tensor].
 *
 * The sum is accumulated in [Double], so it stays accurate for tensors of millions of elements.
 */
@JvmName("sumFloat")
fun Tensor<Float>.sum(): Float = sumAsDouble().toFloat()

// Во float накопленная сумма миллиона значений 0.1 уходила на 1%: каждое следующее слагаемое
// округлялось к шагу, который растёт вместе с суммой
private fun Tensor<Float>.sumAsDouble(): Double {
    var sum = 0.0
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
 *
 * The result wraps around on overflow, as [UByte] arithmetic does in Kotlin: a sum of 20000
 * gives `32`. Convert with [toIntTensor] first to get the full sum; [avg] is not affected.
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
 *
 * @throws NoSuchElementException if the tensor is empty.
 */
@JvmName("avgFloat")
fun Tensor<Float>.avg(): Float {
    requireNotEmpty("the average")
    return (sumAsDouble() / shape.flatSize).toFloat()
}

/**
 * Returns the average of all elements in the [Tensor], rounded toward zero.
 *
 * The sum is accumulated in [Long], so the average is correct even when the sum overflows [Int].
 *
 * @throws NoSuchElementException if the tensor is empty.
 */
@JvmName("avgInt")
fun Tensor<Int>.avg(): Int {
    requireNotEmpty("the average")
    // Сумма в Int переполнялась: среднее трёх Int.MAX_VALUE выходило 715827881
    var sum = 0L
    forEach { sum += it }
    return (sum / shape.flatSize).toInt()
}

/**
 * Returns the average of all elements in the [Tensor], rounded toward zero.
 *
 * @throws NoSuchElementException if the tensor is empty.
 */
@JvmName("avgUByte")
fun Tensor<UByte>.avg(): UByte {
    requireNotEmpty("the average")
    // Раньше делилась сумма, уже обрезанная до UByte: среднее ста значений 200 выходило 0
    var sum = 0L
    forEach { sum += it.toLong() }
    return (sum / shape.flatSize).toInt().toUByte()
}

/**
 * Returns the average of all elements in the [Tensor].
 *
 * @throws NoSuchElementException if the tensor is empty.
 */
@JvmName("avgLong")
fun Tensor<Long>.avg(): Long {
    requireNotEmpty("the average")
    return sum() / shape.flatSize
}

/**
 * Returns the minimum of all elements in the [Tensor].
 *
 * @throws NoSuchElementException if the tensor is empty.
 */
@JvmName("minFloat")
fun Tensor<Float>.min(): Float {
    requireNotEmpty("the minimum")
    var min = Float.POSITIVE_INFINITY
    forEach { min = minOf(min, it) }
    return min
}

/**
 * Returns the minimum of all elements in the [Tensor].
 *
 * @throws NoSuchElementException if the tensor is empty.
 */
@JvmName("minInt")
fun Tensor<Int>.min(): Int {
    requireNotEmpty("the minimum")
    var min = Int.MAX_VALUE
    forEach { min = minOf(min, it) }
    return min
}

/**
 * Returns the minimum of all elements in the [Tensor].
 *
 * @throws NoSuchElementException if the tensor is empty.
 */
@JvmName("minUByte")
fun Tensor<UByte>.min(): UByte {
    requireNotEmpty("the minimum")
    var min = UByte.MAX_VALUE
    forEach { min = minOf(min, it) }
    return min
}

/**
 * Returns the minimum of all elements in the [Tensor].
 *
 * @throws NoSuchElementException if the tensor is empty.
 */
@JvmName("minLong")
fun Tensor<Long>.min(): Long {
    requireNotEmpty("the minimum")
    var min = Long.MAX_VALUE
    forEach { min = minOf(min, it) }
    return min
}

/**
 * Returns the maximum of all elements in the [Tensor].
 *
 * @throws NoSuchElementException if the tensor is empty.
 */
@JvmName("maxFloat")
fun Tensor<Float>.max(): Float {
    requireNotEmpty("the maximum")
    var max = Float.NEGATIVE_INFINITY
    forEach { max = maxOf(max, it) }
    return max
}

/**
 * Returns the maximum of all elements in the [Tensor].
 *
 * @throws NoSuchElementException if the tensor is empty.
 */
@JvmName("maxInt")
fun Tensor<Int>.max(): Int {
    requireNotEmpty("the maximum")
    var max = Int.MIN_VALUE
    forEach { max = maxOf(max, it) }
    return max
}

/**
 * Returns the maximum of all elements in the [Tensor].
 *
 * @throws NoSuchElementException if the tensor is empty.
 */
@JvmName("maxUByte")
fun Tensor<UByte>.max(): UByte {
    requireNotEmpty("the maximum")
    var max = UByte.MIN_VALUE
    forEach { max = maxOf(max, it) }
    return max
}

/**
 * Returns the maximum of all elements in the [Tensor].
 *
 * @throws NoSuchElementException if the tensor is empty.
 */
@JvmName("maxLong")
fun Tensor<Long>.max(): Long {
    requireNotEmpty("the maximum")
    var max = Long.MIN_VALUE
    forEach { max = maxOf(max, it) }
    return max
}

/**
 * Returns the index of the maximum element in the [Tensor].
 *
 * @throws NoSuchElementException if the tensor is empty.
 */
@JvmName("argMaxFloat")
fun Tensor<Float>.argmax(): IntArray {
    requireNotEmpty("argmax")
    var max = Float.NEGATIVE_INFINITY
    var maxIndex = 0
    val size = shape.flatSize
    for (i in 0 until size) {
        val v = getFlat(i)
        if (v > max) {
            max = v
            maxIndex = i
        }
    }
    return maxIndex.toNestedIndex(shape)
}

/**
 * Returns the index of the maximum element in the [Tensor].
 *
 * @throws NoSuchElementException if the tensor is empty.
 */
@JvmName("argMaxInt")
fun Tensor<Int>.argmax(): IntArray {
    requireNotEmpty("argmax")
    var max = Int.MIN_VALUE
    var maxIndex = 0
    val size = shape.flatSize
    for (i in 0 until size) {
        val v = getFlat(i)
        if (v > max) {
            max = v
            maxIndex = i
        }
    }
    return maxIndex.toNestedIndex(shape)
}

/**
 * Returns the index of the maximum element in the [Tensor].
 *
 * @throws NoSuchElementException if the tensor is empty.
 */
@JvmName("argMaxUByte")
fun Tensor<UByte>.argmax(): IntArray {
    requireNotEmpty("argmax")
    var max = UByte.MIN_VALUE
    var maxIndex = 0
    val size = shape.flatSize
    for (i in 0 until size) {
        val v = getFlat(i)
        if (v > max) {
            max = v
            maxIndex = i
        }
    }
    return maxIndex.toNestedIndex(shape)
}

/**
 * Returns the index of the maximum element in the [Tensor].
 *
 * @throws NoSuchElementException if the tensor is empty.
 */
@JvmName("argMaxLong")
fun Tensor<Long>.argmax(): IntArray {
    requireNotEmpty("argmax")
    var max = Long.MIN_VALUE
    var maxIndex = 0
    val size = shape.flatSize
    for (i in 0 until size) {
        val v = getFlat(i)
        if (v > max) {
            max = v
            maxIndex = i
        }
    }
    return maxIndex.toNestedIndex(shape)
}

/**
 * Returns the index of the minimum element in the [Tensor].
 *
 * @throws NoSuchElementException if the tensor is empty.
 */
@JvmName("argMinFloat")
fun Tensor<Float>.argmin(): IntArray {
    requireNotEmpty("argmin")
    var min = Float.POSITIVE_INFINITY
    var minIndex = 0
    val size = shape.flatSize
    for (i in 0 until size) {
        val v = getFlat(i)
        if (v < min) {
            min = v
            minIndex = i
        }
    }
    return minIndex.toNestedIndex(shape)
}

/**
 * Returns the index of the minimum element in the [Tensor].
 *
 * @throws NoSuchElementException if the tensor is empty.
 */
@JvmName("argMinInt")
fun Tensor<Int>.argmin(): IntArray {
    requireNotEmpty("argmin")
    var min = Int.MAX_VALUE
    var minIndex = 0
    val size = shape.flatSize
    for (i in 0 until size) {
        val v = getFlat(i)
        if (v < min) {
            min = v
            minIndex = i
        }
    }
    return minIndex.toNestedIndex(shape)
}

/**
 * Returns the index of the minimum element in the [Tensor].
 *
 * @throws NoSuchElementException if the tensor is empty.
 */
@JvmName("argMinUByte")
fun Tensor<UByte>.argmin(): IntArray {
    requireNotEmpty("argmin")
    var min = UByte.MAX_VALUE
    var minIndex = 0
    val size = shape.flatSize
    for (i in 0 until size) {
        val v = getFlat(i)
        if (v < min) {
            min = v
            minIndex = i
        }
    }
    return minIndex.toNestedIndex(shape)
}

/**
 * Returns the index of the minimum element in the [Tensor].
 *
 * @throws NoSuchElementException if the tensor is empty.
 */
@JvmName("argMinLong")
fun Tensor<Long>.argmin(): IntArray {
    requireNotEmpty("argmin")
    var min = Long.MAX_VALUE
    var minIndex = 0
    val size = shape.flatSize
    for (i in 0 until size) {
        val v = getFlat(i)
        if (v < min) {
            min = v
            minIndex = i
        }
    }
    return minIndex.toNestedIndex(shape)
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

    // Все значения одинаковы - диапазона нет, и деление на ноль превратило бы совершенно
    // обычный вход (однотонный кадр, тензор из одного элемента) в сплошной NaN, который
    // дальше молча расползается по инференсу. Принятое соглашение - нули.
    val range = max - min
    if (range == 0f) return map { 0f }

    return map { (it - min) / range }
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

@Suppress("UNCHECKED_CAST")
@OptIn(ExperimentalUnsignedTypes::class)
@JvmName("toFlatArrayFloat")
fun Tensor<Float>.toFlatArray(): FloatArray {
    val size = shape.flatSize
    val array = FloatArray(size)
    for (i in 0 until size) array[i] = getFlat(i)
    return array
}

@Suppress("UNCHECKED_CAST")
@OptIn(ExperimentalUnsignedTypes::class)
@JvmName("toFlatArrayInt")
fun Tensor<Int>.toFlatArray(): IntArray {
    val size = shape.flatSize
    val array = IntArray(size)
    for (i in 0 until size) array[i] = getFlat(i)
    return array
}

@Suppress("UNCHECKED_CAST")
@OptIn(ExperimentalUnsignedTypes::class)
@JvmName("toFlatArrayUByte")
fun Tensor<UByte>.toFlatArray(): UByteArray {
    val size = shape.flatSize
    val array = UByteArray(size)
    for (i in 0 until size) array[i] = getFlat(i)
    return array
}

@Suppress("UNCHECKED_CAST")
@OptIn(ExperimentalUnsignedTypes::class)
@JvmName("toFlatArrayLong")
fun Tensor<Long>.toFlatArray(): LongArray {
    val size = shape.flatSize
    val array = LongArray(size)
    for (i in 0 until size) array[i] = getFlat(i)
    return array
}

// Раньше пустой тензор вёл себя в каждой функции по-своему: min и max возвращали заглушки
// (Infinity, Int.MAX_VALUE), argmax падал с IllegalArgumentException про индекс, avg давал NaN
// или ArithmeticException. Теперь - как min() в стандартной библиотеке
private fun Tensor<*>.requireNotEmpty(operation: String) {
    if (shape.flatSize == 0) {
        throw NoSuchElementException("Cannot compute $operation of an empty tensor of shape $shape")
    }
}
