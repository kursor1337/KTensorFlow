package dev.kursor.ktensorflow.api.gpu

import dev.kursor.ktensorflow.api.Delegate

interface GpuDelegate : Delegate

expect fun GpuDelegate(
    options: GpuDelegateOptions = GpuDelegateOptions()
): Delegate
