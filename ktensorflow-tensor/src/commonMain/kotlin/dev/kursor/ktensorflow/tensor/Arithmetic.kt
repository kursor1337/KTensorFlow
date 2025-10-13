package dev.kursor.ktensorflow.tensor

operator fun Tensor<Float>.plus(other: Tensor<Float>): Tensor<Float> =
    mapIndexed { index, it ->
        val i = index.toFlatIndex(shape)
        it + other[i]
    }

operator fun Tensor<Float>.plus(other: Tensor<Int>): Tensor<Float> =
    mapIndexed { index, it ->
        val i = index.toFlatIndex(shape)
        it + other[i].toFloat()
    }


operator fun Tensor<Float>.plus(other: Tensor<UByte>): Tensor<Float> =
    mapIndexed { index, it ->
        val i = index.toFlatIndex(shape)
        it + other[i].toFloat()
    }

operator fun Tensor<Float>.plus(other: Tensor<Long>): Tensor<Float> =
    mapIndexed { index, it ->
        val i = index.toFlatIndex(shape)
        it + other[i].toFloat()
    }

operator fun Tensor<Int>.plus(other: Tensor<Float>): Tensor<Float> =
    mapIndexed { index, it ->
        val i = index.toFlatIndex(shape)
        it + other[i]
    }

operator fun Tensor<Int>.plus(other: Tensor<Int>): Tensor<Int> =
    mapIndexed { index, it ->
        val i = index.toFlatIndex(shape)
        it + other[i]
    }

operator fun Tensor<Int>.plus(other: Tensor<UByte>): Tensor<Int> =
    mapIndexed { index, it ->
        val i = index.toFlatIndex(shape)
        it + other[i].toInt()
    }

operator fun Tensor<Int>.plus(other: Tensor<Long>): Tensor<Long> =
    mapIndexed { index, it ->
        val i = index.toFlatIndex(shape)
        it + other[i]
    }

operator fun Tensor<UByte>.plus(other: Tensor<Float>): Tensor<Float> =
    mapIndexed { index, it ->
        val i = index.toFlatIndex(shape)
        it.toFloat() + other[i]
    }

operator fun Tensor<UByte>.plus(other: Tensor<Int>): Tensor<Int> =
    mapIndexed { index, it ->
        val i = index.toFlatIndex(shape)
        it.toInt() + other[i]
    }

operator fun Tensor<UByte>.plus(other: Tensor<UByte>): Tensor<UByte> =
    mapIndexed { index, it ->
        val i = index.toFlatIndex(shape)
        (it + other[i]).toUByte()
    }

operator fun Tensor<UByte>.plus(other: Tensor<Long>): Tensor<Long> =
    mapIndexed { index, it ->
        val i = index.toFlatIndex(shape)
        it.toLong() + other[i]
    }

operator fun Tensor<Long>.plus(other: Tensor<Float>): Tensor<Float> =
    mapIndexed { index, it ->
        val i = index.toFlatIndex(shape)
        it.toFloat() + other[i]
    }

operator fun Tensor<Long>.plus(other: Tensor<Int>): Tensor<Long> =
    mapIndexed { index, it ->
        val i = index.toFlatIndex(shape)
        it + other[i]
    }

operator fun Tensor<Long>.plus(other: Tensor<UByte>): Tensor<Long> =
    mapIndexed { index, it ->
        val i = index.toFlatIndex(shape)
        it + other[i].toLong()
    }

operator fun Tensor<Long>.plus(other: Tensor<Long>): Tensor<Long> =
    mapIndexed { index, it ->
        val i = index.toFlatIndex(shape)
        it + other[i]
    }

operator fun Tensor<Float>.plus(other: Float): Tensor<Float> = map { it + other }

operator fun Tensor<Float>.plus(other: Int): Tensor<Float> = map { it + other }

operator fun Tensor<Float>.plus(other: UByte): Tensor<Float> = map { it + other.toFloat() }

operator fun Tensor<Float>.plus(other: Long): Tensor<Float> = map { it + other }

operator fun Tensor<Int>.plus(other: Float): Tensor<Float> = map { it + other }

operator fun Tensor<Int>.plus(other: Int): Tensor<Int> = map { it + other }

operator fun Tensor<Int>.plus(other: UByte): Tensor<Int> = map { it + other.toInt() }

operator fun Tensor<Int>.plus(other: Long): Tensor<Long> = map { it + other }

operator fun Tensor<UByte>.plus(other: Float): Tensor<Float> = map { it.toFloat() + other }

operator fun Tensor<UByte>.plus(other: Int): Tensor<Int> = map { it.toInt() + other }

operator fun Tensor<UByte>.plus(other: UByte): Tensor<UByte> = map { (it + other).toUByte() }

operator fun Tensor<UByte>.plus(other: Long): Tensor<Long> = map { it.toLong() + other }

operator fun Tensor<Long>.plus(other: Float): Tensor<Float> = map { it + other }

operator fun Tensor<Long>.plus(other: Int): Tensor<Long> = map { it + other }

operator fun Tensor<Long>.plus(other: UByte): Tensor<Long> = map { it + other.toLong() }

operator fun Tensor<Long>.plus(other: Long): Tensor<Long> = map { it + other }


operator fun Tensor<Float>.times(other: Tensor<Float>): Tensor<Float> =
    mapIndexed { index, it ->
        val i = index.toFlatIndex(shape)
        it * other[i]
    }

operator fun Tensor<Float>.times(other: Tensor<Int>): Tensor<Float> =
    mapIndexed { index, it ->
        val i = index.toFlatIndex(shape)
        it * other[i].toFloat()
    }


operator fun Tensor<Float>.times(other: Tensor<UByte>): Tensor<Float> =
    mapIndexed { index, it ->
        val i = index.toFlatIndex(shape)
        it * other[i].toFloat()
    }

operator fun Tensor<Float>.times(other: Tensor<Long>): Tensor<Float> =
    mapIndexed { index, it ->
        val i = index.toFlatIndex(shape)
        it * other[i].toFloat()
    }

operator fun Tensor<Int>.times(other: Tensor<Float>): Tensor<Float> =
    mapIndexed { index, it ->
        val i = index.toFlatIndex(shape)
        it * other[i]
    }

operator fun Tensor<Int>.times(other: Tensor<Int>): Tensor<Int> =
    mapIndexed { index, it ->
        val i = index.toFlatIndex(shape)
        it * other[i]
    }

operator fun Tensor<Int>.times(other: Tensor<UByte>): Tensor<Int> =
    mapIndexed { index, it ->
        val i = index.toFlatIndex(shape)
        it * other[i].toInt()
    }

operator fun Tensor<Int>.times(other: Tensor<Long>): Tensor<Long> =
    mapIndexed { index, it ->
        val i = index.toFlatIndex(shape)
        it * other[i]
    }

operator fun Tensor<UByte>.times(other: Tensor<Float>): Tensor<Float> =
    mapIndexed { index, it ->
        val i = index.toFlatIndex(shape)
        it.toFloat() * other[i]
    }

operator fun Tensor<UByte>.times(other: Tensor<Int>): Tensor<Int> =
    mapIndexed { index, it ->
        val i = index.toFlatIndex(shape)
        it.toInt() * other[i]
    }

operator fun Tensor<UByte>.times(other: Tensor<UByte>): Tensor<UByte> =
    mapIndexed { index, it ->
        val i = index.toFlatIndex(shape)
        (it * other[i]).toUByte()
    }

operator fun Tensor<UByte>.times(other: Tensor<Long>): Tensor<Long> =
    mapIndexed { index, it ->
        val i = index.toFlatIndex(shape)
        it.toLong() * other[i]
    }

operator fun Tensor<Long>.times(other: Tensor<Float>): Tensor<Float> =
    mapIndexed { index, it ->
        val i = index.toFlatIndex(shape)
        it.toFloat() * other[i]
    }

operator fun Tensor<Long>.times(other: Tensor<Int>): Tensor<Long> =
    mapIndexed { index, it ->
        val i = index.toFlatIndex(shape)
        it * other[i]
    }

operator fun Tensor<Long>.times(other: Tensor<UByte>): Tensor<Long> =
    mapIndexed { index, it ->
        val i = index.toFlatIndex(shape)
        it * other[i].toLong()
    }

operator fun Tensor<Long>.times(other: Tensor<Long>): Tensor<Long> =
    mapIndexed { index, it ->
        val i = index.toFlatIndex(shape)
        it * other[i]
    }

operator fun Tensor<Float>.times(other: Float): Tensor<Float> = map { it * other }

operator fun Tensor<Float>.times(other: Int): Tensor<Float> = map { it * other }

operator fun Tensor<Float>.times(other: UByte): Tensor<Float> = map { it * other.toFloat() }

operator fun Tensor<Float>.times(other: Long): Tensor<Float> = map { it * other }

operator fun Tensor<Int>.times(other: Float): Tensor<Float> = map { it * other }

operator fun Tensor<Int>.times(other: Int): Tensor<Int> = map { it * other }

operator fun Tensor<Int>.times(other: UByte): Tensor<Int> = map { it * other.toInt() }

operator fun Tensor<Int>.times(other: Long): Tensor<Long> = map { it * other }

operator fun Tensor<UByte>.times(other: Float): Tensor<Float> = map { it.toFloat() * other }

operator fun Tensor<UByte>.times(other: Int): Tensor<Int> = map { it.toInt() * other }

operator fun Tensor<UByte>.times(other: UByte): Tensor<UByte> = map { (it * other).toUByte() }

operator fun Tensor<UByte>.times(other: Long): Tensor<Long> = map { it.toLong() * other }

operator fun Tensor<Long>.times(other: Float): Tensor<Float> = map { it * other }

operator fun Tensor<Long>.times(other: Int): Tensor<Long> = map { it * other }

operator fun Tensor<Long>.times(other: UByte): Tensor<Long> = map { it * other.toLong() }

operator fun Tensor<Long>.times(other: Long): Tensor<Long> = map { it * other }
