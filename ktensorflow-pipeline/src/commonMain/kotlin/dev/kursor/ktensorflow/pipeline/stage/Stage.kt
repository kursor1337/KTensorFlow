package dev.kursor.ktensorflow.pipeline.stage

import dev.kursor.ktensorflow.ExperimentalKTensorFlowApi

/**
 * Stage is a class that represents a stage of a pipeline.
 * It takes an input and produces an output.
 */
@ExperimentalKTensorFlowApi
fun interface Stage<in Input, out Output> {
    fun run(input: Input): Output

    companion object Companion
}
