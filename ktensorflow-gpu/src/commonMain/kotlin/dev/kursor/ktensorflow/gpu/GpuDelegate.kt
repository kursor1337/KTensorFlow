package dev.kursor.ktensorflow.gpu

import dev.kursor.ktensorflow.api.Delegate

/**
 * Delegate to use GPU for inference.
 */
interface GpuDelegate : Delegate

/**
 * Creates [dev.kursor.ktensorflow.gpu.GpuDelegate] with the specified [options].
 */
expect fun GpuDelegate(
    options: GpuDelegateOptions = GpuDelegateOptions()
): Delegate
