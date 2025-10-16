package dev.kursor.ktensorflow.npu

import dev.kursor.ktensorflow.Delegate

actual fun NpuDelegate(options: NpuDelegateOptions): Delegate {
    return IosNpuDelegate(options.tflOptions)
}