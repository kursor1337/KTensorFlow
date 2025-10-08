package dev.kursor.ktensorflow.pipeline

import dev.kursor.ktensorflow.ExperimentalKTensorFlowApi

@ExperimentalKTensorFlowApi
sealed interface Tuple {

    @ExperimentalKTensorFlowApi
    data object Zero : Tuple

    @ExperimentalKTensorFlowApi
    data class One<T1>(
        val first: T1
    ) : Tuple

    @ExperimentalKTensorFlowApi
    data class Two<T1, T2>(
        val first: T1,
        val second: T2
    ) : Tuple

    @ExperimentalKTensorFlowApi
    data class Three<T1, T2, T3>(
        val first: T1,
        val second: T2,
        val third: T3
    ) : Tuple

    @ExperimentalKTensorFlowApi
    data class Four<T1, T2, T3, T4>(
        val first: T1,
        val second: T2,
        val third: T3,
        val fourth: T4
    ) : Tuple

    @ExperimentalKTensorFlowApi
    data class Five<T1, T2, T3, T4, T5>(
        val first: T1,
        val second: T2,
        val third: T3,
        val fourth: T4,
        val fifth: T5
    ) : Tuple

    @ExperimentalKTensorFlowApi
    data class Six<T1, T2, T3, T4, T5, T6>(
        val first: T1,
        val second: T2,
        val third: T3,
        val fourth: T4,
        val fifth: T5,
        val sixth: T6
    ) : Tuple

    @ExperimentalKTensorFlowApi
    data class Seven<T1, T2, T3, T4, T5, T6, T7>(
        val first: T1,
        val second: T2,
        val third: T3,
        val fourth: T4,
        val fifth: T5,
        val sixth: T6,
        val seventh: T7
    ) : Tuple

    @ExperimentalKTensorFlowApi
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

    @ExperimentalKTensorFlowApi
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

    @ExperimentalKTensorFlowApi
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

@ExperimentalKTensorFlowApi
fun tuple() = Tuple.Zero

@ExperimentalKTensorFlowApi
fun <T1> tuple(first: T1) = Tuple.One(first)

@ExperimentalKTensorFlowApi
fun <T1, T2> tuple(
    first: T1,
    second: T2
) = Tuple.Two(first, second)

@ExperimentalKTensorFlowApi
fun <T1, T2, T3> tuple(
    first: T1,
    second: T2,
    third: T3
) = Tuple.Three(first, second, third)

@ExperimentalKTensorFlowApi
fun <T1, T2, T3, T4> tuple(
    first: T1,
    second: T2,
    third: T3,
    fourth: T4
) = Tuple.Four(first, second, third, fourth)

@ExperimentalKTensorFlowApi
fun <T1, T2, T3, T4, T5> tuple(
    first: T1,
    second: T2,
    third: T3,
    fourth: T4,
    fifth: T5
) = Tuple.Five(first, second, third, fourth, fifth)

@ExperimentalKTensorFlowApi
fun <T1, T2, T3, T4, T5, T6> tuple(
    first: T1,
    second: T2,
    third: T3,
    fourth: T4,
    fifth: T5,
    sixth: T6
) = Tuple.Six(first, second, third, fourth, fifth, sixth)

@ExperimentalKTensorFlowApi
fun <T1, T2, T3, T4, T5, T6, T7> tuple(
    first: T1,
    second: T2,
    third: T3,
    fourth: T4,
    fifth: T5,
    sixth: T6,
    seventh: T7
) = Tuple.Seven(first, second, third, fourth, fifth, sixth, seventh)

@ExperimentalKTensorFlowApi
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

@ExperimentalKTensorFlowApi
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

@ExperimentalKTensorFlowApi
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
