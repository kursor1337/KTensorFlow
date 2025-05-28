package dev.kursor.ktensorflow.api.gpu

import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegateFactory

actual class GpuDelegateOptions(
    val tflOptions: GpuDelegateFactory.Options
)

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