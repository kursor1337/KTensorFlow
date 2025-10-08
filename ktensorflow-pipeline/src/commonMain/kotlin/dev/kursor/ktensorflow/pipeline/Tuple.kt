package dev.kursor.ktensorflow.pipeline

import dev.kursor.ktensorflow.Tensor

sealed interface Tuple {

    data object Zero : Tuple

    data class One<T1>(
        val first: T1
    ) : Tuple

    data class Two<T1, T2>(
        val first: T1,
        val second: T2
    ) : Tuple

    data class Three<T1, T2, T3>(
        val first: T1,
        val second: T2,
        val third: T3
    ) : Tuple

    data class Four<T1, T2, T3, T4>(
        val first: T1,
        val second: T2,
        val third: T3,
        val fourth: T4
    ) : Tuple


    data class Five<T1, T2, T3, T4, T5>(
        val first: T1,
        val second: T2,
        val third: T3,
        val fourth: T4,
        val fifth: T5
    ) : Tuple


    data class Six<T1, T2, T3, T4, T5, T6>(
        val first: T1,
        val second: T2,
        val third: T3,
        val fourth: T4,
        val fifth: T5,
        val sixth: T6
    ) : Tuple

    data class Seven<T1, T2, T3, T4, T5, T6, T7>(
        val first: T1,
        val second: T2,
        val third: T3,
        val fourth: T4,
        val fifth: T5,
        val sixth: T6,
        val seventh: T7
    ) : Tuple

    data class Eight<T1, T2, T3, T4, T5, T6, T7, T8>(
        val first: T1,
        val second: T2,
        val third: T3,
        val fourth: T4,
        val fifth: T5,
        val sixth: T6,
        val seventh: T7,
        val eighth: T8
    ) : Tuple

    data class Nine<T1, T2, T3, T4, T5, T6, T7, T8, T9>(
        val first: T1,
        val second: T2,
        val third: T3,
        val fourth: T4,
        val fifth: T5,
        val sixth: T6,
        val seventh: T7,
        val eighth: T8,
        val ninth: T9
    ) : Tuple

    data class Ten<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10>(
        val first: T1,
        val second: T2,
        val third: T3,
        val fourth: T4,
        val fifth: T5,
        val sixth: T6,
        val seventh: T7,
        val eighth: T8,
        val ninth: T9,
        val tenth: T10
    ) : Tuple
}

fun tuple() = Tuple.Zero

fun <T1> tuple(first: T1) = Tuple.One(first)

fun <T1, T2> tuple(
    first: T1,
    second: T2
) = Tuple.Two(first, second)

fun <T1, T2, T3> tuple(
    first: T1,
    second: T2,
    third: T3
) = Tuple.Three(first, second, third)

fun <T1, T2, T3, T4> tuple(
    first: T1,
    second: T2,
    third: T3,
    fourth: T4
) = Tuple.Four(first, second, third, fourth)

fun <T1, T2, T3, T4, T5> tuple(
    first: T1,
    second: T2,
    third: T3,
    fourth: T4,
    fifth: T5
) = Tuple.Five(first, second, third, fourth, fifth)

fun <T1, T2, T3, T4, T5, T6> tuple(
    first: T1,
    second: T2,
    third: T3,
    fourth: T4,
    fifth: T5,
    sixth: T6
) = Tuple.Six(first, second, third, fourth, fifth, sixth)

fun <T1, T2, T3, T4, T5, T6, T7> tuple(
    first: T1,
    second: T2,
    third: T3,
    fourth: T4,
    fifth: T5,
    sixth: T6,
    seventh: T7
) = Tuple.Seven(first, second, third, fourth, fifth, sixth, seventh)

fun <T1, T2, T3, T4, T5, T6, T7, T8> tuple(
    first: T1,
    second: T2,
    third: T3,
    fourth: T4,
    fifth: T5,
    sixth: T6,
    seventh: T7,
    eighth: T8
) = Tuple.Eight(first, second, third, fourth, fifth, sixth, seventh, eighth)

fun <T1, T2, T3, T4, T5, T6, T7, T8, T9> tuple(
    first: T1,
    second: T2,
    third: T3,
    fourth: T4,
    fifth: T5,
    sixth: T6,
    seventh: T7,
    eighth: T8,
    ninth: T9
) = Tuple.Nine(first, second, third, fourth, fifth, sixth, seventh, eighth, ninth)

fun <T1, T2, T3, T4, T5, T6, T7, T8, T9, T10> tuple(
    first: T1,
    second: T2,
    third: T3,
    fourth: T4,
    fifth: T5,
    sixth: T6,
    seventh: T7,
    eighth: T8,
    ninth: T9,
    tenth: T10
) = Tuple.Ten(first, second, third, fourth, fifth, sixth, seventh, eighth, ninth, tenth)
