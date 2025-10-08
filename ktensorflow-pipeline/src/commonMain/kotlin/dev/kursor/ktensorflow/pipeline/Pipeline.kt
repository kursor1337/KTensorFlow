package dev.kursor.ktensorflow.pipeline

import dev.kursor.ktensorflow.ExperimentalKTensorFlowApi
import dev.kursor.ktensorflow.pipeline.stage.Stage

/**
 * Pipeline is a class that represents a pipeline of stages.
 */
@ExperimentalKTensorFlowApi
class Pipeline<in Input, out Output>(
    internal val stage: Stage<Input, Output>
) : Stage<Input, Output> by stage {
    companion object
}

/**
 * Creates a linear pipeline, meaning, that is has single input and single output.
 */
@ExperimentalKTensorFlowApi
fun <T> Pipeline.Companion.linear() = Stage<T>()