package dev.kursor.ktensorflow.npu

import dev.kursor.ktensorflow.Delegate

/**
 * Delegate to use NPU for inference.
 */
interface NpuDelegate : Delegate

/**
 * Creates [NpuDelegate] with the specified [options].
 * Note: NPU delegate on Android uses NNAPI delegate under the hood.
 * NNAPI delegate needs time to compile for underlying hardware (up to 15 seconds on lower-end devices)
 * This can cause ANR's if called on Main Thread.
 * It is highly advised to create Interpreter with NpuDelegate on background threads (Dispatchers.IO)
 */
expect fun NpuDelegate(
    options: NpuDelegateOptions = NpuDelegateOptions()
): Delegate
