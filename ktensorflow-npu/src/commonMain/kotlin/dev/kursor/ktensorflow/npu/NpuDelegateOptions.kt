package dev.kursor.ktensorflow.npu

/**
 * Options for [NpuDelegate].
*/
expect class NpuDelegateOptions

/**
 * Creates default options for [NpuDelegate].
*/
expect fun NpuDelegateOptions(): NpuDelegateOptions

/**
 * Creates options for [NpuDelegate].
 * @param maxDelegatedPartitions The maximum number of partitions to delegate.
*/
expect fun NpuDelegateOptions(
    maxDelegatedPartitions: Int
): NpuDelegateOptions