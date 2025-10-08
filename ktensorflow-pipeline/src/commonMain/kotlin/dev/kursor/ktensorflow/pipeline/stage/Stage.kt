package dev.kursor.ktensorflow.pipeline.stage

fun interface Stage<in Input, out Output> {
    fun run(input: Input): Output

    companion object Companion
}
