package dev.kursor.ktensorflow.pipeline.stage

import dev.kursor.ktensorflow.ExperimentalKTensorFlowApi

@ExperimentalKTensorFlowApi
fun interface Stage<in Input, out Output> {
    fun run(input: Input): Output

    companion object Companion
}
