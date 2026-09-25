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
    // Опции - обычный Java-объект и не зависят от списка, а сам список держит нативную память
    return GpuDelegateOptions(CompatibilityList().use { it.bestOptionsForThisDevice })
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