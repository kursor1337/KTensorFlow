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
 * Adds every available delegate from [delegates] to the [Interpreter.Options] builder, in list
 * order, exactly as on iOS. TensorFlow Lite applies them one after another: each delegate takes
 * the operations it supports from what the previous ones left, and the rest runs on the CPU.
 *
 * Unavailable delegates are skipped without touching their native part.
 */
fun Interpreter.Options.setDelegates(delegates: List<Delegate>): Interpreter.Options {
    delegates
        .filter { it.isAvailable }
        .mapNotNull { it.tflDelegate }
        .forEach { addDelegate(it) }
    return this
}