package dev.kursor.ktensorflow.gpu

import cocoapods.TensorFlowLiteObjC.TFLMetalDelegateOptions

actual data class GpuDelegateOptions(
    val tflOptions: TFLMetalDelegateOptions
)

/**
 * Creates [GpuDelegateOptions] with the specified [builder].
 * Allows to specify native TensorFlow Metal delegate options.
 */
fun GpuDelegateOptions(builder: TFLMetalDelegateOptions.() -> Unit): GpuDelegateOptions {
    return GpuDelegateOptions(TFLMetalDelegateOptions().apply(builder))
}

actual fun GpuDelegateOptions(): GpuDelegateOptions {
    return GpuDelegateOptions(TFLMetalDelegateOptions())
}

actual fun GpuDelegateOptions(
    precisionLossAllowed: Boolean,
    quantizationEnabled: Boolean
): GpuDelegateOptions {
    return GpuDelegateOptions {
        setPrecisionLossAllowed(precisionLossAllowed)
        setQuantizationEnabled(quantizationEnabled)
    }
}