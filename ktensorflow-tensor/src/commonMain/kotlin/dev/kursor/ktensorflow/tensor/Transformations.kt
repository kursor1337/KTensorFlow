package dev.kursor.ktensorflow.tensor

import kotlin.reflect.KClass

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
    dataType: KClass<R>,
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
): Tensor<R> = map(R::class, transform)

inline fun <T : Any, R : Any> Tensor<T>.mapIndexed(
    dataType: KClass<R>,
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
): Tensor<R> = mapIndexed(R::class, transform)

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

fun Tensor<Float>.sum(): Float {
    var s = 0f
    forEach { s += it }
    return s
}

fun Tensor<Float>.avg(): Float = sum() / shape.flatSize

@Suppress("UNCHECKED_CAST")
inline fun <reified T : Any> Tensor<T>.toFloatTensor(): Tensor<Float> = when (T::class) {
    UByte::class -> map { (it as UByte).toFloat() }
    Float::class -> this as Tensor<Float>
    else -> map { (it as Number).toFloat() }
}

@Suppress("UNCHECKED_CAST")
inline fun <reified T : Any> Tensor<T>.toIntTensor(): Tensor<Int> = when (T::class) {
    UByte::class -> map { (it as UByte).toInt() }
    Int::class -> this as Tensor<Int>
    else -> map { (it as Number).toInt() }
}

@Suppress("UNCHECKED_CAST")
inline fun <reified T : Any> Tensor<T>.toLongTensor(): Tensor<Long> = when (T::class) {
    UByte::class -> map { (it as UByte).toLong() }
    Long::class -> this as Tensor<Long>
    else -> map { (it as Number).toLong() }
}

@Suppress("UNCHECKED_CAST")
inline fun <reified T : Any> Tensor<T>.toUByteTensor(): Tensor<UByte> = when (T::class) {
    UByte::class -> this as Tensor<UByte>
    Float::class -> map { (it as Float).toInt().toUByte() }
    Int::class -> map { (it as Int).toUByte() }
    Long::class -> map { (it as Long).toUByte() }
    else -> map { (it as Number).toByte().toUByte() }
}

fun <T : Any> Tensor<T>.toList(): List<T> {
    val list = ArrayList<T>(shape.flatSize)
    forEach { list.add(it) }
    return list
}
