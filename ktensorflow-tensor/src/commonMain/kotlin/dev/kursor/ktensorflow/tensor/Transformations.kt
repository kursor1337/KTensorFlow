package dev.kursor.ktensorflow.tensor

import kotlin.jvm.JvmName

inline fun <T : Any> Tensor<T>.forEach(action: (T) -> Unit) {
    for (i in 0 until shape.flatSize) action(get(i.toNestedIndex(shape)))
}

inline fun <T : Any> Tensor<T>.forEachIndexed(action: (IntArray, T) -> Unit) {
    for (i in 0 until shape.flatSize) {
        val index = i.toNestedIndex(shape)
        action(index, get(index))
    }
}

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

inline fun <T : Any, reified R : Any> Tensor<T>.map(
    noinline transform: (T) -> R
): Tensor<R> = map(TensorDataType.of<R>(), transform)

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

inline fun <T : Any, reified R : Any> Tensor<T>.mapIndexed(
    noinline transform: (IntArray, T) -> R
): Tensor<R> = mapIndexed(TensorDataType.of<R>(), transform)

inline fun <T : Any> Tensor<T>.mapInPlace(crossinline transform: (T) -> T) {
    forEachIndexed { index, value ->
        this[index] = transform(value)
    }
}

inline fun <T : Any> Tensor<T>.mapInPlaceIndexed(crossinline transform: (IntArray, T) -> T) {
    forEachIndexed { index, value ->
        this[index] = transform(index, value)
    }
}

fun <T : Any> Tensor<T>.reshape(newShape: TensorShape): Tensor<T> {
    require(newShape.flatSize == shape.flatSize) {
        "Cannot reshape tensor of shape $shape to $newShape (different element count)"
    }
    return Tensor(dataType, newShape, data.copyOf())
}

fun <T : Any> Tensor<T>.flatten(): Tensor<T> =
    reshape(TensorShape(shape.flatSize))

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

@JvmName("sumFloat")
fun Tensor<Float>.sum(): Float {
    var sum = 0f
    forEach { sum += it }
    return sum
}

@JvmName("sumInt")
fun Tensor<Int>.sum(): Int {
    var sum = 0
    forEach { sum += it }
    return sum
}

@JvmName("sumUByte")
fun Tensor<UByte>.sum(): UByte {
    var sum = 0U
    forEach { sum += it }
    return sum.toUByte()
}

@JvmName("sumLong")
fun Tensor<Long>.sum(): Long {
    var sum = 0L
    forEach { sum += it }
    return sum
}

@JvmName("avgFloat")
fun Tensor<Float>.avg(): Float = sum() / shape.flatSize

@JvmName("avgInt")
fun Tensor<Int>.avg(): Int = sum() / shape.flatSize

@JvmName("avgUByte")
fun Tensor<UByte>.avg(): UByte = (sum() / shape.flatSize.toUInt()).toUByte()

@JvmName("avgLong")
fun Tensor<Long>.avg(): Long = sum() / shape.flatSize

@JvmName("minFloat")
fun Tensor<Float>.min(): Float {
    var min = Float.MAX_VALUE
    forEach { min = minOf(min, it) }
    return min
}

@JvmName("minInt")
fun Tensor<Int>.min(): Int {
    var min = Int.MAX_VALUE
    forEach { min = minOf(min, it) }
    return min
}

@JvmName("minUByte")
fun Tensor<UByte>.min(): UByte {
    var min = UByte.MAX_VALUE
    forEach { min = minOf(min, it) }
    return min
}

@JvmName("minLong")
fun Tensor<Long>.min(): Long {
    var min = Long.MAX_VALUE
    forEach { min = minOf(min, it) }
    return min
}

@JvmName("maxFloat")
fun Tensor<Float>.max(): Float {
    var max = Float.MIN_VALUE
    forEach { max = maxOf(max, it) }
    return max
}

@JvmName("maxInt")
fun Tensor<Int>.max(): Int {
    var max = Int.MIN_VALUE
    forEach { max = maxOf(max, it) }
    return max
}

@JvmName("maxUByte")
fun Tensor<UByte>.max(): UByte {
    var max = UByte.MIN_VALUE
    forEach { max = maxOf(max, it) }
    return max
}

@JvmName("maxLong")
fun Tensor<Long>.max(): Long {
    var max = Long.MIN_VALUE
    forEach { max = maxOf(max, it) }
    return max
}

@JvmName("argMaxFloat")
fun Tensor<Float>.argMax(): IntArray {
    var max = Float.MIN_VALUE
    var maxIndex = IntArray(shape.rank)
    forEachIndexed { index, value ->
        if (value > max) {
            max = value
            maxIndex = index
        }
    }
    return maxIndex
}

@JvmName("argMaxInt")
fun Tensor<Int>.argMax(): IntArray {
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

@JvmName("argMaxUByte")
fun Tensor<UByte>.argMax(): IntArray {
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

@JvmName("argMaxLong")
fun Tensor<Long>.argMax(): IntArray {
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

@JvmName("argMinFloat")
fun Tensor<Float>.argMin(): IntArray {
    var min = Float.MAX_VALUE
    var minIndex = IntArray(shape.rank)
    forEachIndexed { index, value ->
        if (value < min) {
            min = value
            minIndex = index
        }
    }
    return minIndex
}

@JvmName("argMinInt")
fun Tensor<Int>.argMin(): IntArray {
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

@JvmName("argMinUByte")
fun Tensor<UByte>.argMin(): IntArray {
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

@JvmName("argMinLong")
fun Tensor<Long>.argMin(): IntArray {
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

@JvmName("normalizeFloat")
fun Tensor<Float>.normalize(): Tensor<Float> {
    val min = min()
    val max = max()
    return map { (it - min) / (max - min) }
}

@Suppress("UNCHECKED_CAST")
fun <T : Any> Tensor<T>.toFloatTensor(): Tensor<Float> = when (dataType) {
    TensorDataType.UInt8 -> map { (it as UByte).toFloat() }
    TensorDataType.Float32 -> this as Tensor<Float>
    else -> map { (it as Number).toFloat() }
}

@Suppress("UNCHECKED_CAST")
fun <T : Any> Tensor<T>.toIntTensor(): Tensor<Int> = when (dataType) {
    TensorDataType.UInt8 -> map { (it as UByte).toInt() }
    TensorDataType.Int32 -> this as Tensor<Int>
    else -> map { (it as Number).toInt() }
}

@Suppress("UNCHECKED_CAST")
fun <T : Any> Tensor<T>.toLongTensor(): Tensor<Long> = when (dataType) {
    TensorDataType.UInt8 -> map { (it as UByte).toLong() }
    TensorDataType.Int64 -> this as Tensor<Long>
    else -> map { (it as Number).toLong() }
}

@Suppress("UNCHECKED_CAST")
fun <T : Any> Tensor<T>.toUByteTensor(): Tensor<UByte> = when (dataType) {
    TensorDataType.UInt8 -> this as Tensor<UByte>
    TensorDataType.Float32 -> map { (it as Float).toInt().toUByte() }
    TensorDataType.Int32 -> map { (it as Int).toUByte() }
    TensorDataType.Int64 -> map { (it as Long).toUByte() }
}

fun <T : Any> Tensor<T>.toList(): List<T> {
    val list = ArrayList<T>(shape.flatSize)
    forEach { list.add(it) }
    return list
}
