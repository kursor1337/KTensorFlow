package dev.kursor.ktensorflow.api.gpu

import dev.kursor.ktensorflow.api.Delegate
import dev.kursor.ktensorflow.impl.IosGpuDelegate

actual fun GpuDelegate(options: GpuDelegateOptions): Delegate {
    return IosGpuDelegate(options.tflOptions)
}