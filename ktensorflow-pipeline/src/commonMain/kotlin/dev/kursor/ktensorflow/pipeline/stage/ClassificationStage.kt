package dev.kursor.ktensorflow.pipeline.stage

import dev.kursor.ktensorflow.ExperimentalKTensorFlowApi

@ExperimentalKTensorFlowApi
class ClassificationStage<T>(private val classes: List<T>) : Stage<Int, T> {
    override fun run(input: Int): T {
        return classes[input]
    }
}

@ExperimentalKTensorFlowApi
fun <Input, T> Stage<Input, Int>.classify(classes: List<T>) =
    this.then(ClassificationStage(classes))