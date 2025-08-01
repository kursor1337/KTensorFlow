package dev.kursor.ktensorflow.gpu

import dev.kursor.ktensorflow.api.Delegate
import dev.kursor.ktensorflow.gpu.AndroidGpuDelegate

actual fun GpuDelegate(options: GpuDelegateOptions): Delegate {
    return AndroidGpuDelegate(options.tflOptions)
}