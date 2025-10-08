package dev.kursor.ktensorflow.pipeline.stage

import dev.kursor.ktensorflow.ExperimentalKTensorFlowApi

@ExperimentalKTensorFlowApi
class FilterStage<T>(private val predicate: (T) -> Boolean) : Stage<List<T>, List<T>> {
    override fun run(input: List<T>): List<T> {
        return input.filter(predicate)
    }
}

@ExperimentalKTensorFlowApi
fun <Input, T> Stage<Input, List<T>>.filter(predicate: (T) -> Boolean) = this.then(FilterStage(predicate))

@ExperimentalKTensorFlowApi
fun <Input, T> Stage<Input, List<T>>.filterNot(predicate: (T) -> Boolean) = this.then(FilterStage { !predicate(it) })

@ExperimentalKTensorFlowApi
class IndexedFilterStage<T>(private val predicate: (Int, T) -> Boolean) : Stage<List<T>, List<T>> {
    override fun run(input: List<T>): List<T> {
        return input.filterIndexed(predicate)
    }
}

@ExperimentalKTensorFlowApi
fun <Input, T> Stage<Input, List<T>>.filterIndexed(predicate: (Int, T) -> Boolean) = this.then(IndexedFilterStage(predicate))