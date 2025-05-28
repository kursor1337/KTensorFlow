package dev.kursor.ktensorflow.api.gpu

expect class GpuDelegateOptions

expect fun GpuDelegateOptions(): GpuDelegateOptions

expect fun GpuDelegateOptions(
    precisionLossAllowed: Boolean,
    quantizationEnabled: Boolean
): GpuDelegateOptions