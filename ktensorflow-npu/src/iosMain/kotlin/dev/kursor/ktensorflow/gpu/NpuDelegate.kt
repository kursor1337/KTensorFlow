package dev.kursor.ktensorflow.gpu

import dev.kursor.ktensorflow.Delegate

actual fun NpuDelegate(options: NpuDelegateOptions): Delegate {
    return IosNpuDelegate(options.tflOptions)
}