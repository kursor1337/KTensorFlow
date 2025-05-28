package dev.kursor.ktensorflow.api.gpu

import dev.kursor.ktensorflow.api.Delegate
import dev.kursor.ktensorflow.impl.AndroidGpuDelegate

actual fun GpuDelegate(options: GpuDelegateOptions): Delegate {
    return AndroidGpuDelegate(options.tflOptions)
}