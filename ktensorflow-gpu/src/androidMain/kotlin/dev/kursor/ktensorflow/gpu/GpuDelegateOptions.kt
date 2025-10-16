package dev.kursor.ktensorflow.gpu

import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegateFactory

actual class GpuDelegateOptions(
    val tflOptions: GpuDelegateFactory.Options
)

/**
 * Creates [GpuDelegateOptions] with the specified [builder].
 * Allows to specify native TensorFlow GPU delegate options
 */
fun GpuDelegateOptions(builder: GpuDelegateFactory.Options.() -> Unit): GpuDelegateOptions {
    return GpuDelegateOptions(GpuDelegateFactory.Options().apply(builder))
}

actual fun GpuDelegateOptions(): GpuDelegateOptions {
    return GpuDelegateOptions(CompatibilityList().bestOptionsForThisDevice)
}

actual fun GpuDelegateOptions(
    precisionLossAllowed: Boolean,
    quantizationEnabled: Boolean
): GpuDelegateOptions {
    return GpuDelegateOptions {
        setPrecisionLossAllowed(precisionLossAllowed)
        setQuantizedModelsAllowed(quantizationEnabled)
    }
}