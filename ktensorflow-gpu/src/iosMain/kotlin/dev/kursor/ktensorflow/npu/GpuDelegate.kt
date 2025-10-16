package dev.kursor.ktensorflow.npu

import dev.kursor.ktensorflow.Delegate

actual fun GpuDelegate(options: GpuDelegateOptions): Delegate {
    return IosGpuDelegate(options.tflOptions)
}