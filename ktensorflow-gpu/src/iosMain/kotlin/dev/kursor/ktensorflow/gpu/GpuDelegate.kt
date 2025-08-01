package dev.kursor.ktensorflow.gpu

import dev.kursor.ktensorflow.api.Delegate
import dev.kursor.ktensorflow.gpu.IosGpuDelegate

actual fun GpuDelegate(options: GpuDelegateOptions): Delegate {
    return IosGpuDelegate(options.tflOptions)
}