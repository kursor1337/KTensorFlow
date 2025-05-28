package dev.kursor.ktensorflow.api

import cocoapods.TensorFlowLiteObjC.TFLDelegate
import cocoapods.TensorFlowLiteObjC.TFLInterpreterOptions

actual data class InterpreterOptions(
    val tflOptions: TFLInterpreterOptions,
    val tflDelegates: List<TFLDelegate>
)

/**
 * Creates [InterpreterOptions] with custom [TFLInterpreterOptions] and [TFLDelegate].
 * Allows to specify native TensorFlow options and delegates.
 */
fun InterpreterOptions(
    delegates: List<TFLDelegate>,
    builder: TFLInterpreterOptions.() -> Unit
): InterpreterOptions {
    return InterpreterOptions(
        tflOptions = TFLInterpreterOptions().apply(builder),
        tflDelegates = delegates
    )
}

actual fun InterpreterOptions(): InterpreterOptions {
    return InterpreterOptions(
        tflOptions = TFLInterpreterOptions(),
        tflDelegates = emptyList()
    )
}

actual fun InterpreterOptions(
    numThreads: Int,
    useXNNPACK: Boolean,
    delegates: List<Delegate>
): InterpreterOptions {
    return InterpreterOptions(
        tflOptions = TFLInterpreterOptions().apply {
            setNumberOfThreads(numThreads.toULong())
            setUseXNNPACK(useXNNPACK)
        },
        tflDelegates = delegates.mapNotNull { it.tflDelegate }
    )
}
