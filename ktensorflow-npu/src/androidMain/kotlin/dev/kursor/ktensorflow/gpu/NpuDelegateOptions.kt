package dev.kursor.ktensorflow.gpu

import org.tensorflow.lite.nnapi.NnApiDelegate

actual class NpuDelegateOptions(
    val tflOptions: NnApiDelegate.Options
)

/**
 * Creates [NpuDelegateOptions] with the specified [builder].
 * Allows to specify native TensorFlow NNAPI delegate options
 */
fun NpuDelegateOptions(builder: NnApiDelegate.Options.() -> Unit): NpuDelegateOptions {
    return NpuDelegateOptions(NnApiDelegate.Options().apply(builder))
}

actual fun NpuDelegateOptions(): NpuDelegateOptions {
    return NpuDelegateOptions(NnApiDelegate.Options())
}

actual fun NpuDelegateOptions(
    maxDelegatedPartitions: Int
): NpuDelegateOptions {
    return NpuDelegateOptions {
        setMaxNumberOfDelegatedPartitions(maxDelegatedPartitions)
    }
}