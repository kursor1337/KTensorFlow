package dev.kursor.ktensorflow.pipeline

fun interface Stage<in Input : Any, out Output : Any> {
    fun run(input: Input): Output

    companion object Companion
}

internal class CombinedStage<in Input : Any, Intermediate : Any, out Output : Any>(
    private val inner: Stage<Input, Intermediate>,
    private val outer: Stage<Intermediate, Output>
) : Stage<Input, Output> {
    override fun run(input: Input): Output {
        val intermediate = inner.run(input)
        return outer.run(intermediate)
    }
}

fun <Input : Any, Intermediate : Any, Output : Any> Stage<Input, Intermediate>.then(
    other: Stage<Intermediate, Output>
): Stage<Input, Output> {
    return CombinedStage(this, other)
}

class EmptyStage<Input : Any> : Stage<Input, Input> {
    override fun run(input: Input): Input {
        return input
    }

}

fun <Input : Any> Stage.Companion.empty(): Stage<Input, Input> {
    return EmptyStage()
}
