package dev.kursor.ktensorflow.api.gpu

/**
 * Options for [GpuDelegate].
*/
expect class GpuDelegateOptions

/**
 * Creates default options for [GpuDelegate].
*/
expect fun GpuDelegateOptions(): GpuDelegateOptions

/**
 * Creates options for [GpuDelegate].
 * @param precisionLossAllowed Whether to allow precision loss, e.g. float32 to float16.
 * @param quantizationEnabled Whether to enable use of quantized models.
*/
expect fun GpuDelegateOptions(
    precisionLossAllowed: Boolean,
    quantizationEnabled: Boolean
): GpuDelegateOptions