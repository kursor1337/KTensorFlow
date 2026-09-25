package dev.kursor.ktensorflow

/**
 * Options for [Interpreter].
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
 * @param delegates [Delegate]s to use for inference. Every available delegate is applied in
 * list order: each takes the operations it supports from what the previous ones left, and the
 * rest runs on the CPU. Unavailable delegates are skipped. Whether a delegate can actually be
 * applied to the model is known only when the [Interpreter] is created - a delegate that cannot
 * be applied makes that call fail with [TensorFlowException].
 */
expect fun InterpreterOptions(
    numThreads: Int,
    useXNNPACK: Boolean,
    delegates: List<Delegate> = emptyList()
): InterpreterOptions
