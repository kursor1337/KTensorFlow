package dev.kursor.ktensorflow.pipeline.builder

import dev.kursor.ktensorflow.ExperimentalKTensorFlowApi
import dev.kursor.ktensorflow.Interpreter
import dev.kursor.ktensorflow.Tensor
import dev.kursor.ktensorflow.pipeline.stage.Stage
import dev.kursor.ktensorflow.pipeline.Tuple
import dev.kursor.ktensorflow.pipeline.stage.then

@ExperimentalKTensorFlowApi
class InferencePipelineBuilder<Input : Tuple> internal constructor(
    internal val inputStage: Stage<Input, List<Tensor>>,
    internal val interpreter: Interpreter
)

@ExperimentalKTensorFlowApi
fun <Input : Tuple, Output : Tuple> InputPipelineBuilder<Input, Output>.inference(
    interpreter: Interpreter
): InferencePipelineBuilder<Input> {
    val inputStage = this.build().then { it.toTensorList() }
    return InferencePipelineBuilder(inputStage, interpreter)
}

@ExperimentalKTensorFlowApi
internal fun Tuple.toTensorList(): List<Tensor> = when (this) {
    is Tuple.Zero -> emptyList()
    is Tuple.One<*> -> listOf(
        this.first
    )

    is Tuple.Two<*, *> -> listOf(
        this.first,
        this.second
    )

    is Tuple.Three<*, *, *> -> listOf(
        this.first,
        this.second,
        this.third
    )

    is Tuple.Four<*, *, *, *> -> listOf(
        this.first,
        this.second,
        this.third,
        this.fourth
    )

    is Tuple.Five<*, *, *, *, *> -> listOf(
        this.first,
        this.second,
        this.third,
        this.fourth,
        this.fifth
    )

    is Tuple.Six<*, *, *, *, *, *> -> listOf(
        this.first,
        this.second,
        this.third,
        this.fourth,
        this.fifth,
        this.sixth
    )

    is Tuple.Seven<*, *, *, *, *, *, *> -> listOf(
        this.first,
        this.second,
        this.third,
        this.fourth,
        this.fifth,
        this.sixth,
        this.seventh
    )

    is Tuple.Eight<*, *, *, *, *, *, *, *> -> listOf(
        this.first,
        this.second,
        this.third,
        this.fourth,
        this.fifth,
        this.sixth,
        this.seventh,
        this.eighth
    )

    is Tuple.Nine<*, *, *, *, *, *, *, *, *> -> listOf(
        this.first,
        this.second,
        this.third,
        this.fourth,
        this.fifth,
        this.sixth,
        this.seventh,
        this.eighth,
        this.ninth
    )

    is Tuple.Ten<*, *, *, *, *, *, *, *, *, *> -> listOf(
        this.first,
        this.second,
        this.third,
        this.fourth,
        this.fifth,
        this.sixth,
        this.seventh,
        this.eighth,
        this.ninth,
        this.tenth
    )
} as List<Tensor>