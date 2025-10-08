package dev.kursor.ktensorflow.pipeline.stage

import dev.kursor.ktensorflow.ExperimentalKTensorFlowApi

internal class ChainedStage<in Input, Intermediate, out Output>(
    private val inner: Stage<Input, Intermediate>,
    private val outer: Stage<Intermediate, Output>
) : Stage<Input, Output> {
    override fun run(input: Input): Output {
        val intermediate = inner.run(input)
        return outer.run(intermediate)
    }
}

@ExperimentalKTensorFlowApi
fun <Input, Intermediate, Output> Stage<Input, Intermediate>.then(
    other: Stage<Intermediate, Output>
): Stage<Input, Output> {
    return ChainedStage(this, other)
}