package dev.kursor.ktensorflow

/**
 * Options for [Interpreter].
 * @param numThreads Number of threads to use for inference.
 * @param useXNNPACK Whether to use XNNPACK for optimized inference.
 * @param hardware [Hardware] to use for inference.
 */
expect class InterpreterOptions

/**
 * Creates [InterpreterOptions] with default values.
 */
expect fun InterpreterOptions(): InterpreterOptions

/**
 * Creates [InterpreterOptions] with custom values.
 * @param numThreads Number of threads to use for inference. Used only on CPU.
 * @param useXNNPACK Whether to use XNNPACK for optimized inference on CPU.
 * @param delegates List of [Delegate]s to use for inference.
 */
expect fun InterpreterOptions(
    numThreads: Int,
    useXNNPACK: Boolean,
    delegates: List<Delegate> = emptyList()
): InterpreterOptions
