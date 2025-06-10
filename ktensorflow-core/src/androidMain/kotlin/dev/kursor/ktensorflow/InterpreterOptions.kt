package dev.kursor.ktensorflow

import org.tensorflow.lite.Interpreter

actual data class InterpreterOptions(
    val tflOptions: Interpreter.Options
)

/**
 * Builder for [InterpreterOptions].
 * Allows to specify native TensorFlow options and delegates.
 */
fun InterpreterOptions(builder: Interpreter.Options.() -> Unit): InterpreterOptions {
    return InterpreterOptions(Interpreter.Options().apply(builder))
}

actual fun InterpreterOptions(): InterpreterOptions {
    return InterpreterOptions(Interpreter.Options())
}

actual fun InterpreterOptions(
    numThreads: Int,
    useXNNPACK: Boolean,
    delegates: List<Delegate>
): InterpreterOptions {
    return InterpreterOptions {
        setNumThreads(numThreads)
        setUseXNNPACK(useXNNPACK)
        setDelegates(delegates)
    }
}

/**
 * Adds [delegates] to the [Interpreter.Options] builder.
 * Only the first delegate from [delegates] that is available will be added.
 */
fun Interpreter.Options.setDelegates(delegates: List<Delegate>): Interpreter.Options {
    delegates
        .firstOrNull { it.isAvailable }
        ?.let { addDelegate(it.tflDelegate) }
    return this
}