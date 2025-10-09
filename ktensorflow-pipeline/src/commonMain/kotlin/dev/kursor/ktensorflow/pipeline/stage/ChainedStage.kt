package dev.kursor.ktensorflow.pipeline.stage

import dev.kursor.ktensorflow.ExperimentalKTensorFlowApi

@ExperimentalKTensorFlowApi
internal class ChainedStage<in Input, Intermediate, out Output>(
    private val inner: Stage<Input, Intermediate>,
    private val outer: Stage<Intermediate, Output>
) : Stage<Input, Output> {
    override fun run(input: Input): Output {
        val intermediate = inner.run(input)
        return outer.run(intermediate)
    }
}

/**
 * Chains two stages together.
 *
 * @param other The stage to chain after this stage.
 * @return A new stage that chains this stage and the other stage.
 */
@ExperimentalKTensorFlowApi
fun <Input, Intermediate, Output> Stage<Input, Intermediate>.then(
    other: Stage<Intermediate, Output>
): Stage<Input, Output> {
    return ChainedStage(this, other)
}