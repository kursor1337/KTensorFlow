package dev.kursor.ktensorflow.gpu

import dev.kursor.ktensorflow.Delegate

/**
 * Delegate to use GPU for inference.
 */
interface GpuDelegate : Delegate

/**
 * Creates [GpuDelegate] with the specified [options].
 */
expect fun GpuDelegate(
    options: GpuDelegateOptions = GpuDelegateOptions()
): Delegate
