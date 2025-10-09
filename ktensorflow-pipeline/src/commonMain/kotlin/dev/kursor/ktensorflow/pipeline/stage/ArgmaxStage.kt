package dev.kursor.ktensorflow.pipeline.stage

import dev.kursor.ktensorflow.ExperimentalKTensorFlowApi

@ExperimentalKTensorFlowApi
class ArgmaxStage : Stage<FloatArray, Int> {
    override fun run(input: FloatArray): Int {
        return input.withIndex().maxBy { it.value }.index
    }
}

@ExperimentalKTensorFlowApi
fun <T> Stage<T, FloatArray>.argmax() = this.then(ArgmaxStage())

@ExperimentalKTensorFlowApi
class ValuedArgmaxStage : Stage<FloatArray, Pair<Int, Float>> {
    override fun run(input: FloatArray): Pair<Int, Float> {
        return input.withIndex().maxBy { it.value }.let { it.index to it.value }
    }
}

@ExperimentalKTensorFlowApi
fun <T> Stage<T, FloatArray>.valuedArgmax() = this.then(ValuedArgmaxStage())
