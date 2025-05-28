package dev.kursor.ktensorflow.api

/**
 * Options for [Interpreter].
 * @param numThreads Number of threads to use for inference.
 * @param useXNNPACK Whether to use XNNPACK for optimized inference.
 * @param hardware [Hardware] to use for inference.
 */
expect class InterpreterOptions

expect fun InterpreterOptions(): InterpreterOptions

expect fun InterpreterOptions(
    numThreads: Int,
    useXNNPACK: Boolean,
    delegates: List<Delegate> = emptyList()
): InterpreterOptions
