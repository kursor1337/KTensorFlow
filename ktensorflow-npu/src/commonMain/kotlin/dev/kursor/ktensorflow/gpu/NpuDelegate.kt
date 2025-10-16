package dev.kursor.ktensorflow.gpu

import dev.kursor.ktensorflow.Delegate

/**
 * Delegate to use NPU for inference.
 */
interface NpuDelegate : Delegate

/**
 * Creates [NpuDelegate] with the specified [options].
 */
expect fun NpuDelegate(
    options: NpuDelegateOptions = NpuDelegateOptions()
): Delegate
