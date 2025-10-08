package dev.kursor.ktensorflow.pipeline

import dev.kursor.ktensorflow.pipeline.stage.Stage

class Pipeline<in Input, out Output>(
    internal val stage: Stage<Input, Output>
) : Stage<Input, Output> by stage {
    companion object
}

fun <T> Pipeline.Companion.linear() = Stage<T>()