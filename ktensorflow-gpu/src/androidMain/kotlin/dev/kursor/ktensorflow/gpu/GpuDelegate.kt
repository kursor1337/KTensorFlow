package dev.kursor.ktensorflow.gpu

import dev.kursor.ktensorflow.Delegate

actual fun GpuDelegate(options: GpuDelegateOptions): Delegate {
    return AndroidGpuDelegate(options.tflOptions)
}