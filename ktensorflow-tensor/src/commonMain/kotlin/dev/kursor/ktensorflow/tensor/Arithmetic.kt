package dev.kursor.ktensorflow.tensor

import kotlin.jvm.JvmName

@JvmName("floatTensorPlusFloatTensor")
operator fun Tensor<Float>.plus(other: Tensor<Float>): Tensor<Float> =
    mapIndexed { index, it -> it + other[index] }

@JvmName("floatTensorPlusIntTensor")
operator fun Tensor<Float>.plus(other: Tensor<Int>): Tensor<Float> =
    mapIndexed { index, it -> it + other[index] }

@JvmName("floatTensorPlusUByteTensor")
operator fun Tensor<Float>.plus(other: Tensor<UByte>): Tensor<Float> =
    mapIndexed { index, it -> it + other[index].toFloat() }

@JvmName("floatTensorPlusLongTensor")
operator fun Tensor<Float>.plus(other: Tensor<Long>): Tensor<Float> =
    mapIndexed { index, it -> it + other[index] }

@JvmName("intTensorPlusFloatTensor")
operator fun Tensor<Int>.plus(other: Tensor<Float>): Tensor<Float> =
    mapIndexed { index, it -> it + other[index] }

@JvmName("intTensorPlusIntTensor")
operator fun Tensor<Int>.plus(other: Tensor<Int>): Tensor<Int> =
    mapIndexed { index, it -> it + other[index] }

@JvmName("intTensorPlusUByteTensor")
operator fun Tensor<Int>.plus(other: Tensor<UByte>): Tensor<Int> =
    mapIndexed { index, it -> it + other[index].toInt() }

@JvmName("intTensorPlusLongTensor")
operator fun Tensor<Int>.plus(other: Tensor<Long>): Tensor<Long> =
    mapIndexed { index, it -> it + other[index] }

@JvmName("uByteTensorPlusFloatTensor")
operator fun Tensor<UByte>.plus(other: Tensor<Float>): Tensor<Float> =
    mapIndexed { index, it -> it.toFloat() + other[index] }

@JvmName("uByteTensorPlusIntTensor")
operator fun Tensor<UByte>.plus(other: Tensor<Int>): Tensor<Int> =
    mapIndexed { index, it -> it.toInt() + other[index] }

@JvmName("uByteTensorPlusUByteTensor")
operator fun Tensor<UByte>.plus(other: Tensor<UByte>): Tensor<UByte> =
    mapIndexed { index, it -> (it + other[index]).toUByte() }

@JvmName("uByteTensorPlusLongTensor")
operator fun Tensor<UByte>.plus(other: Tensor<Long>): Tensor<Long> =
    mapIndexed { index, it -> it.toLong() + other[index] }

@JvmName("longTensorPlusFloatTensor")
operator fun Tensor<Long>.plus(other: Tensor<Float>): Tensor<Float> = mapIndexed { index, it -> it + other[index] }

@JvmName("longTensorPlusIntTensor")
operator fun Tensor<Long>.plus(other: Tensor<Int>): Tensor<Long> =
    mapIndexed { index, it -> it + other[index] }

@JvmName("longTensorPlusUByteTensor")
operator fun Tensor<Long>.plus(other: Tensor<UByte>): Tensor<Long> =
    mapIndexed { index, it -> it + other[index].toLong() }

@JvmName("longTensorPlusLongTensor")
operator fun Tensor<Long>.plus(other: Tensor<Long>): Tensor<Long> =
    mapIndexed { index, it -> it + other[index] }

@JvmName("floatTensorPlusFloat")
operator fun Tensor<Float>.plus(other: Float): Tensor<Float> = map { it + other }

@JvmName("floatTensorPlusInt")
operator fun Tensor<Float>.plus(other: Int): Tensor<Float> = map { it + other }

@JvmName("floatTensorPlusUByte")
operator fun Tensor<Float>.plus(other: UByte): Tensor<Float> = map { it + other.toFloat() }

@JvmName("floatTensorPlusLong")
operator fun Tensor<Float>.plus(other: Long): Tensor<Float> = map { it + other }

@JvmName("intTensorPlusFloat")
operator fun Tensor<Int>.plus(other: Float): Tensor<Float> = map { it + other }

@JvmName("intTensorPlusInt")
operator fun Tensor<Int>.plus(other: Int): Tensor<Int> = map { it + other }

@JvmName("intTensorPlusUByte")
operator fun Tensor<Int>.plus(other: UByte): Tensor<Int> = map { it + other.toInt() }

@JvmName("intTensorPlusLong")
operator fun Tensor<Int>.plus(other: Long): Tensor<Long> = map { it + other }

@JvmName("uByteTensorPlusFloat")
operator fun Tensor<UByte>.plus(other: Float): Tensor<Float> = map { it.toFloat() + other }

@JvmName("uByteTensorPlusInt")
operator fun Tensor<UByte>.plus(other: Int): Tensor<Int> = map { it.toInt() + other }

@JvmName("uByteTensorPlusUByte")
operator fun Tensor<UByte>.plus(other: UByte): Tensor<UByte> = map { (it + other).toUByte() }

@JvmName("uByteTensorPlusLong")
operator fun Tensor<UByte>.plus(other: Long): Tensor<Long> = map { it.toLong() + other }

@JvmName("longTensorPlusFloat")
operator fun Tensor<Long>.plus(other: Float): Tensor<Float> = map { it + other }

@JvmName("longTensorPlusInt")
operator fun Tensor<Long>.plus(other: Int): Tensor<Long> = map { it + other }

@JvmName("longTensorPlusUByte")
operator fun Tensor<Long>.plus(other: UByte): Tensor<Long> = map { it + other.toLong() }

@JvmName("longTensorPlusLong")
operator fun Tensor<Long>.plus(other: Long): Tensor<Long> = map { it + other }

@JvmName("floatTensorMinusFloatTensor")
operator fun Tensor<Float>.minus(other: Tensor<Float>): Tensor<Float> =
    mapIndexed { index, it -> it - other[index] }

@JvmName("floatTensorMinusIntTensor")
operator fun Tensor<Float>.minus(other: Tensor<Int>): Tensor<Float> =
    mapIndexed { index, it -> it - other[index] }

@JvmName("floatTensorMinusUByteTensor")
operator fun Tensor<Float>.minus(other: Tensor<UByte>): Tensor<Float> =
    mapIndexed { index, it -> it - other[index].toFloat() }

@JvmName("floatTensorMinusLongTensor")
operator fun Tensor<Float>.minus(other: Tensor<Long>): Tensor<Float> =
    mapIndexed { index, it -> it - other[index] }

@JvmName("intTensorMinusFloatTensor")
operator fun Tensor<Int>.minus(other: Tensor<Float>): Tensor<Float> =
    mapIndexed { index, it -> it - other[index] }

@JvmName("intTensorMinusIntTensor")
operator fun Tensor<Int>.minus(other: Tensor<Int>): Tensor<Int> =
    mapIndexed { index, it -> it - other[index] }

@JvmName("intTensorMinusUByteTensor")
operator fun Tensor<Int>.minus(other: Tensor<UByte>): Tensor<Int> =
    mapIndexed { index, it -> it - other[index].toInt() }

@JvmName("intTensorMinusLongTensor")
operator fun Tensor<Int>.minus(other: Tensor<Long>): Tensor<Long> =
    mapIndexed { index, it -> it - other[index] }

@JvmName("uByteTensorMinusFloatTensor")
operator fun Tensor<UByte>.minus(other: Tensor<Float>): Tensor<Float> =
    mapIndexed { index, it -> it.toFloat() - other[index] }

@JvmName("uByteTensorMinusIntTensor")
operator fun Tensor<UByte>.minus(other: Tensor<Int>): Tensor<Int> =
    mapIndexed { index, it -> it.toInt() - other[index] }

@JvmName("uByteTensorMinusUByteTensor")
operator fun Tensor<UByte>.minus(other: Tensor<UByte>): Tensor<UByte> =
    mapIndexed { index, it -> (it - other[index]).toUByte() }

@JvmName("uByteTensorMinusLongTensor")
operator fun Tensor<UByte>.minus(other: Tensor<Long>): Tensor<Long> =
    mapIndexed { index, it -> it.toLong() - other[index] }

@JvmName("longTensorMinusFloatTensor")
operator fun Tensor<Long>.minus(other: Tensor<Float>): Tensor<Float> =
    mapIndexed { index, it -> it - other[index] }

@JvmName("longTensorMinusIntTensor")
operator fun Tensor<Long>.minus(other: Tensor<Int>): Tensor<Long> =
    mapIndexed { index, it -> it - other[index] }

@JvmName("longTensorMinusUByteTensor")
operator fun Tensor<Long>.minus(other: Tensor<UByte>): Tensor<Long> =
    mapIndexed { index, it -> it - other[index].toLong() }

@JvmName("longTensorMinusLongTensor")
operator fun Tensor<Long>.minus(other: Tensor<Long>): Tensor<Long> =
    mapIndexed { index, it -> it - other[index] }

@JvmName("floatTensorMinusFloat")
operator fun Tensor<Float>.minus(other: Float): Tensor<Float> = map { it - other }

@JvmName("floatTensorMinusInt")
operator fun Tensor<Float>.minus(other: Int): Tensor<Float> = map { it - other }

@JvmName("floatTensorMinusUByte")
operator fun Tensor<Float>.minus(other: UByte): Tensor<Float> = map { it - other.toFloat() }

@JvmName("floatTensorMinusLong")
operator fun Tensor<Float>.minus(other: Long): Tensor<Float> = map { it - other }

@JvmName("intTensorMinusFloat")
operator fun Tensor<Int>.minus(other: Float): Tensor<Float> = map { it - other }

@JvmName("intTensorMinusInt")
operator fun Tensor<Int>.minus(other: Int): Tensor<Int> = map { it - other }

@JvmName("intTensorMinusUByte")
operator fun Tensor<Int>.minus(other: UByte): Tensor<Int> = map { it - other.toInt() }

@JvmName("intTensorMinusLong")
operator fun Tensor<Int>.minus(other: Long): Tensor<Long> = map { it - other }

@JvmName("uByteTensorMinusFloat")
operator fun Tensor<UByte>.minus(other: Float): Tensor<Float> = map { it.toFloat() - other }

@JvmName("uByteTensorMinusInt")
operator fun Tensor<UByte>.minus(other: Int): Tensor<Int> = map { it.toInt() - other }

@JvmName("uByteTensorMinusUByte")
operator fun Tensor<UByte>.minus(other: UByte): Tensor<UByte> = map { (it - other).toUByte() }

@JvmName("uByteTensorMinusLong")
operator fun Tensor<UByte>.minus(other: Long): Tensor<Long> = map { it.toLong() - other }

@JvmName("longTensorMinusFloat")
operator fun Tensor<Long>.minus(other: Float): Tensor<Float> = map { it - other }

@JvmName("longTensorMinusInt")
operator fun Tensor<Long>.minus(other: Int): Tensor<Long> = map { it - other }

@JvmName("longTensorMinusUByte")
operator fun Tensor<Long>.minus(other: UByte): Tensor<Long> = map { it - other.toLong() }

@JvmName("longTensorMinusLong")
operator fun Tensor<Long>.minus(other: Long): Tensor<Long> = map { it - other }

@JvmName("floatTensorTimesFloatTensor")
operator fun Tensor<Float>.times(other: Tensor<Float>): Tensor<Float> =
    mapIndexed { index, it -> it * other[index] }

@JvmName("floatTensorTimesIntTensor")
operator fun Tensor<Float>.times(other: Tensor<Int>): Tensor<Float> =
    mapIndexed { index, it -> it * other[index] }

@JvmName("floatTensorTimesUByteTensor")
operator fun Tensor<Float>.times(other: Tensor<UByte>): Tensor<Float> =
    mapIndexed { index, it -> it * other[index].toFloat() }

@JvmName("floatTensorTimesLongTensor")
operator fun Tensor<Float>.times(other: Tensor<Long>): Tensor<Float> =
    mapIndexed { index, it -> it * other[index] }

@JvmName("intTensorTimesFloatTensor")
operator fun Tensor<Int>.times(other: Tensor<Float>): Tensor<Float> =
    mapIndexed { index, it -> it * other[index] }

@JvmName("intTensorTimesIntTensor")
operator fun Tensor<Int>.times(other: Tensor<Int>): Tensor<Int> =
    mapIndexed { index, it -> it * other[index] }

@JvmName("intTensorTimesUByteTensor")
operator fun Tensor<Int>.times(other: Tensor<UByte>): Tensor<Int> =
    mapIndexed { index, it -> it * other[index].toInt() }

@JvmName("intTensorTimesLongTensor")
operator fun Tensor<Int>.times(other: Tensor<Long>): Tensor<Long> =
    mapIndexed { index, it -> it * other[index] }

@JvmName("uByteTensorTimesFloatTensor")
operator fun Tensor<UByte>.times(other: Tensor<Float>): Tensor<Float> =
    mapIndexed { index, it -> it.toFloat() * other[index] }

@JvmName("uByteTensorTimesIntTensor")
operator fun Tensor<UByte>.times(other: Tensor<Int>): Tensor<Int> =
    mapIndexed { index, it -> it.toInt() * other[index] }

@JvmName("uByteTensorTimesUByteTensor")
operator fun Tensor<UByte>.times(other: Tensor<UByte>): Tensor<UByte> =
    mapIndexed { index, it -> (it * other[index]).toUByte() }

@JvmName("uByteTensorTimesLongTensor")
operator fun Tensor<UByte>.times(other: Tensor<Long>): Tensor<Long> =
    mapIndexed { index, it -> it.toLong() * other[index] }

@JvmName("longTensorTimesFloatTensor")
operator fun Tensor<Long>.times(other: Tensor<Float>): Tensor<Float> =
    mapIndexed { index, it -> it.toFloat() * other[index] }

@JvmName("longTensorTimesIntTensor")
operator fun Tensor<Long>.times(other: Tensor<Int>): Tensor<Long> =
    mapIndexed { index, it -> it * other[index] }

@JvmName("longTensorTimesUByteTensor")
operator fun Tensor<Long>.times(other: Tensor<UByte>): Tensor<Long> =
    mapIndexed { index, it -> it * other[index].toLong() }

@JvmName("longTensorTimesLongTensor")
operator fun Tensor<Long>.times(other: Tensor<Long>): Tensor<Long> =
    mapIndexed { index, it -> it * other[index] }

@JvmName("floatTensorTimesFloat")
operator fun Tensor<Float>.times(other: Float): Tensor<Float> = map { it * other }

@JvmName("floatTensorTimesInt")
operator fun Tensor<Float>.times(other: Int): Tensor<Float> = map { it * other }

@JvmName("floatTensorTimesUByte")
operator fun Tensor<Float>.times(other: UByte): Tensor<Float> = map { it * other.toFloat() }

@JvmName("floatTensorTimesLong")
operator fun Tensor<Float>.times(other: Long): Tensor<Float> = map { it * other }

@JvmName("intTensorTimesFloat")
operator fun Tensor<Int>.times(other: Float): Tensor<Float> = map { it * other }

@JvmName("intTensorTimesInt")
operator fun Tensor<Int>.times(other: Int): Tensor<Int> = map { it * other }

@JvmName("intTensorTimesUByte")
operator fun Tensor<Int>.times(other: UByte): Tensor<Int> = map { it * other.toInt() }

@JvmName("intTensorTimesLong")
operator fun Tensor<Int>.times(other: Long): Tensor<Long> = map { it * other }

@JvmName("uByteTensorTimesFloat")
operator fun Tensor<UByte>.times(other: Float): Tensor<Float> = map { it.toFloat() * other }

@JvmName("uByteTensorTimesInt")
operator fun Tensor<UByte>.times(other: Int): Tensor<Int> = map { it.toInt() * other }

@JvmName("uByteTensorTimesUByte")
operator fun Tensor<UByte>.times(other: UByte): Tensor<UByte> = map { (it * other).toUByte() }

@JvmName("uByteTensorTimesLong")
operator fun Tensor<UByte>.times(other: Long): Tensor<Long> = map { it.toLong() * other }

@JvmName("longTensorTimesFloat")
operator fun Tensor<Long>.times(other: Float): Tensor<Float> = map { it * other }

@JvmName("longTensorTimesInt")
operator fun Tensor<Long>.times(other: Int): Tensor<Long> = map { it * other }

@JvmName("longTensorTimesUByte")
operator fun Tensor<Long>.times(other: UByte): Tensor<Long> = map { it * other.toLong() }

@JvmName("longTensorTimesLong")
operator fun Tensor<Long>.times(other: Long): Tensor<Long> = map { it * other }

@JvmName("floatTensorDivFloatTensor")
operator fun Tensor<Float>.div(other: Tensor<Float>): Tensor<Float> =
    mapIndexed { index, it -> it / other[index] }

@JvmName("floatTensorDivIntTensor")
operator fun Tensor<Float>.div(other: Tensor<Int>): Tensor<Float> =
    mapIndexed { index, it -> it / other[index] }

@JvmName("floatTensorDivUByteTensor")
operator fun Tensor<Float>.div(other: Tensor<UByte>): Tensor<Float> =
    mapIndexed { index, it -> it / other[index].toFloat() }

@JvmName("floatTensorDivLongTensor")
operator fun Tensor<Float>.div(other: Tensor<Long>): Tensor<Float> =
    mapIndexed { index, it -> it / other[index] }

@JvmName("intTensorDivFloatTensor")
operator fun Tensor<Int>.div(other: Tensor<Float>): Tensor<Float> =
    mapIndexed { index, it -> it / other[index] }

@JvmName("intTensorDivIntTensor")
operator fun Tensor<Int>.div(other: Tensor<Int>): Tensor<Int> =
    mapIndexed { index, it -> it / other[index] }

@JvmName("intTensorDivUByteTensor")
operator fun Tensor<Int>.div(other: Tensor<UByte>): Tensor<Int> =
    mapIndexed { index, it -> it / other[index].toInt() }

@JvmName("intTensorDivLongTensor")
operator fun Tensor<Int>.div(other: Tensor<Long>): Tensor<Long> =
    mapIndexed { index, it -> it / other[index] }

@JvmName("uByteTensorDivFloatTensor")
operator fun Tensor<UByte>.div(other: Tensor<Float>): Tensor<Float> =
    mapIndexed { index, it -> it.toFloat() / other[index] }

@JvmName("uByteTensorDivIntTensor")
operator fun Tensor<UByte>.div(other: Tensor<Int>): Tensor<Int> =
    mapIndexed { index, it -> it.toInt() / other[index] }

@JvmName("uByteTensorDivUByteTensor")
operator fun Tensor<UByte>.div(other: Tensor<UByte>): Tensor<UByte> =
    mapIndexed { index, it -> (it / other[index]).toUByte() }

@JvmName("uByteTensorDivLongTensor")
operator fun Tensor<UByte>.div(other: Tensor<Long>): Tensor<Long> =
    mapIndexed { index, it -> it.toLong() / other[index] }

@JvmName("longTensorDivFloatTensor")
operator fun Tensor<Long>.div(other: Tensor<Float>): Tensor<Float> =
    mapIndexed { index, it -> it / other[index] }

@JvmName("longTensorDivIntTensor")
operator fun Tensor<Long>.div(other: Tensor<Int>): Tensor<Long> =
    mapIndexed { index, it -> it / other[index] }

@JvmName("longTensorDivUByteTensor")
operator fun Tensor<Long>.div(other: Tensor<UByte>): Tensor<Long> =
    mapIndexed { index, it -> it / other[index].toLong() }

@JvmName("longTensorDivLongTensor")
operator fun Tensor<Long>.div(other: Tensor<Long>): Tensor<Long> =
    mapIndexed { index, it -> it / other[index] }

@JvmName("floatTensorDivFloat")
operator fun Tensor<Float>.div(other: Float): Tensor<Float> = map { it / other }

@JvmName("floatTensorDivInt")
operator fun Tensor<Float>.div(other: Int): Tensor<Float> = map { it / other }

@JvmName("floatTensorDivUByte")
operator fun Tensor<Float>.div(other: UByte): Tensor<Float> = map { it / other.toFloat() }

@JvmName("floatTensorDivLong")
operator fun Tensor<Float>.div(other: Long): Tensor<Float> = map { it / other }

@JvmName("intTensorDivFloat")
operator fun Tensor<Int>.div(other: Float): Tensor<Float> = map { it / other }

@JvmName("intTensorDivInt")
operator fun Tensor<Int>.div(other: Int): Tensor<Int> = map { it / other }

@JvmName("intTensorDivUByte")
operator fun Tensor<Int>.div(other: UByte): Tensor<Int> = map { it / other.toInt() }

@JvmName("intTensorDivLong")
operator fun Tensor<Int>.div(other: Long): Tensor<Long> = map { it / other }

@JvmName("uByteTensorDivFloat")
operator fun Tensor<UByte>.div(other: Float): Tensor<Float> = map { it.toFloat() / other }

@JvmName("uByteTensorDivInt")
operator fun Tensor<UByte>.div(other: Int): Tensor<Int> = map { it.toInt() / other }

@JvmName("uByteTensorDivUByte")
operator fun Tensor<UByte>.div(other: UByte): Tensor<UByte> = map { (it / other).toUByte() }

@JvmName("uByteTensorDivLong")
operator fun Tensor<UByte>.div(other: Long): Tensor<Long> = map { it.toLong() / other }

@JvmName("longTensorDivFloat")
operator fun Tensor<Long>.div(other: Float): Tensor<Float> = map { it / other }

@JvmName("longTensorDivInt")
operator fun Tensor<Long>.div(other: Int): Tensor<Long> = map { it / other }

@JvmName("longTensorDivUByte")
operator fun Tensor<Long>.div(other: UByte): Tensor<Long> = map { it / other.toLong() }

@JvmName("longTensorDivLong")
operator fun Tensor<Long>.div(other: Long): Tensor<Long> = map { it / other }

@JvmName("floatTensorRemFloatTensor")
operator fun Tensor<Float>.rem(other: Tensor<Float>): Tensor<Float> =
    mapIndexed { index, it -> it % other[index] }

@JvmName("floatTensorRemIntTensor")
operator fun Tensor<Float>.rem(other: Tensor<Int>): Tensor<Float> =
    mapIndexed { index, it -> it % other[index] }

@JvmName("floatTensorRemUByteTensor")
operator fun Tensor<Float>.rem(other: Tensor<UByte>): Tensor<Float> =
    mapIndexed { index, it -> it % other[index].toFloat() }

@JvmName("floatTensorRemLongTensor")
operator fun Tensor<Float>.rem(other: Tensor<Long>): Tensor<Float> =
    mapIndexed { index, it -> it % other[index] }

@JvmName("intTensorRemFloatTensor")
operator fun Tensor<Int>.rem(other: Tensor<Float>): Tensor<Float> =
    mapIndexed { index, it -> it % other[index] }

@JvmName("intTensorRemIntTensor")
operator fun Tensor<Int>.rem(other: Tensor<Int>): Tensor<Int> =
    mapIndexed { index, it -> it % other[index] }

@JvmName("intTensorRemUByteTensor")
operator fun Tensor<Int>.rem(other: Tensor<UByte>): Tensor<Int> =
    mapIndexed { index, it -> it % other[index].toInt() }

@JvmName("intTensorRemLongTensor")
operator fun Tensor<Int>.rem(other: Tensor<Long>): Tensor<Long> =
    mapIndexed { index, it -> it % other[index] }

@JvmName("uByteTensorRemFloatTensor")
operator fun Tensor<UByte>.rem(other: Tensor<Float>): Tensor<Float> =
    mapIndexed { index, it -> it.toFloat() % other[index] }

@JvmName("uByteTensorRemIntTensor")
operator fun Tensor<UByte>.rem(other: Tensor<Int>): Tensor<Int> =
    mapIndexed { index, it -> it.toInt() % other[index] }

@JvmName("uByteTensorRemUByteTensor")
operator fun Tensor<UByte>.rem(other: Tensor<UByte>): Tensor<UByte> =
    mapIndexed { index, it -> (it % other[index]).toUByte() }

@JvmName("uByteTensorRemLongTensor")
operator fun Tensor<UByte>.rem(other: Tensor<Long>): Tensor<Long> =
    mapIndexed { index, it -> it.toLong() % other[index] }

@JvmName("longTensorRemFloatTensor")
operator fun Tensor<Long>.rem(other: Tensor<Float>): Tensor<Float> =
    mapIndexed { index, it -> it % other[index] }

@JvmName("longTensorRemIntTensor")
operator fun Tensor<Long>.rem(other: Tensor<Int>): Tensor<Long> =
    mapIndexed { index, it -> it % other[index] }

@JvmName("longTensorRemUByteTensor")
operator fun Tensor<Long>.rem(other: Tensor<UByte>): Tensor<Long> =
    mapIndexed { index, it -> it % other[index].toLong() }

@JvmName("longTensorRemLongTensor")
operator fun Tensor<Long>.rem(other: Tensor<Long>): Tensor<Long> =
    mapIndexed { index, it -> it % other[index] }

@JvmName("floatTensorRemFloat")
operator fun Tensor<Float>.rem(other: Float): Tensor<Float> = map { it % other }

@JvmName("floatTensorRemInt")
operator fun Tensor<Float>.rem(other: Int): Tensor<Float> = map { it % other }

@JvmName("floatTensorRemUByte")
operator fun Tensor<Float>.rem(other: UByte): Tensor<Float> = map { it % other.toFloat() }

@JvmName("floatTensorRemLong")
operator fun Tensor<Float>.rem(other: Long): Tensor<Float> = map { it % other }

@JvmName("intTensorRemFloat")
operator fun Tensor<Int>.rem(other: Float): Tensor<Float> = map { it % other }

@JvmName("intTensorRemInt")
operator fun Tensor<Int>.rem(other: Int): Tensor<Int> = map { it % other }

@JvmName("intTensorRemUByte")
operator fun Tensor<Int>.rem(other: UByte): Tensor<Int> = map { it % other.toInt() }

@JvmName("intTensorRemLong")
operator fun Tensor<Int>.rem(other: Long): Tensor<Long> = map { it % other }

@JvmName("uByteTensorRemFloat")
operator fun Tensor<UByte>.rem(other: Float): Tensor<Float> = map { it.toFloat() % other }

@JvmName("uByteTensorRemInt")
operator fun Tensor<UByte>.rem(other: Int): Tensor<Int> = map { it.toInt() % other }

@JvmName("uByteTensorRemUByte")
operator fun Tensor<UByte>.rem(other: UByte): Tensor<UByte> = map { (it % other).toUByte() }

@JvmName("uByteTensorRemLong")
operator fun Tensor<UByte>.rem(other: Long): Tensor<Long> = map { it.toLong() % other }

@JvmName("longTensorRemFloat")
operator fun Tensor<Long>.rem(other: Float): Tensor<Float> = map { it % other }

@JvmName("longTensorRemInt")
operator fun Tensor<Long>.rem(other: Int): Tensor<Long> = map { it % other }

@JvmName("longTensorRemUByte")
operator fun Tensor<Long>.rem(other: UByte): Tensor<Long> = map { it % other.toLong() }

@JvmName("longTensorRemLong")
operator fun Tensor<Long>.rem(other: Long): Tensor<Long> = map { it % other }

@JvmName("floatTensorInc")
operator fun Tensor<Float>.inc(): Tensor<Float> = map { it + 1 }

@JvmName("intTensorInc")
operator fun Tensor<Int>.inc(): Tensor<Int> = map { it + 1 }

@JvmName("uByteTensorInc")
operator fun Tensor<UByte>.inc(): Tensor<UByte> = map { (it + 1U).toUByte() }

@JvmName("longTensorInc")
operator fun Tensor<Long>.inc(): Tensor<Long> = map { it + 1 }

@JvmName("floatTensorDec")
operator fun Tensor<Float>.dec(): Tensor<Float> = map { it - 1 }

@JvmName("intTensorDec")
operator fun Tensor<Int>.dec(): Tensor<Int> = map { it - 1 }

@JvmName("uByteTensorDec")
operator fun Tensor<UByte>.dec(): Tensor<UByte> = map { (it - 1U).toUByte() }

@JvmName("longTensorDec")
operator fun Tensor<Long>.dec(): Tensor<Long> = map { it - 1 }

@JvmName("floatTensorUnaryPlus")
operator fun Tensor<Float>.unaryPlus(): Tensor<Float> = map { it }

@JvmName("intTensorUnaryPlus")
operator fun Tensor<Int>.unaryPlus(): Tensor<Int> = map { it }

@JvmName("longTensorUnaryPlus")
operator fun Tensor<Long>.unaryPlus(): Tensor<Long> = map { it }

@JvmName("floatTensorUnaryMinus")
operator fun Tensor<Float>.unaryMinus(): Tensor<Float> = map { -it }

@JvmName("intTensorUnaryMinus")
operator fun Tensor<Int>.unaryMinus(): Tensor<Int> = map { -it }

@JvmName("longTensorUnaryMinus")
operator fun Tensor<Long>.unaryMinus(): Tensor<Long> = map { -it }
