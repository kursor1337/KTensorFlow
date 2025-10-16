package dev.kursor.ktensorflow.npu

import cocoapods.TensorFlowLiteObjC.TFLCoreMLDelegateOptions

actual data class NpuDelegateOptions(
    val tflOptions: TFLCoreMLDelegateOptions
)

/**
 * Creates [NpuDelegateOptions] with the specified [builder].
 * Allows to specify native TensorFlow CoreML delegate options.
 */
fun NpuDelegateOptions(builder: TFLCoreMLDelegateOptions.() -> Unit): NpuDelegateOptions {
    return NpuDelegateOptions(TFLCoreMLDelegateOptions().apply(builder))
}

actual fun NpuDelegateOptions(): NpuDelegateOptions {
    return NpuDelegateOptions(TFLCoreMLDelegateOptions())
}

actual fun NpuDelegateOptions(
    maxDelegatedPartitions: Int
): NpuDelegateOptions {
    return NpuDelegateOptions {
        setMaxDelegatedPartitions(maxDelegatedPartitions.toULong())
    }
}