package dev.kursor.ktensorflow.impl

import dev.kursor.ktensorflow.api.gpu.GpuDelegate
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate as TFLGpuDelegate
import org.tensorflow.lite.gpu.GpuDelegateFactory as TFLGpuDelegateFactory

internal class AndroidGpuDelegate(
    val options: TFLGpuDelegateFactory.Options
) : GpuDelegate {
    private val compatList = CompatibilityList()

    override val isAvailable: Boolean = compatList.isDelegateSupportedOnThisDevice

    override val tflDelegate: org.tensorflow.lite.Delegate by lazy {
        TFLGpuDelegate(options)
    }
}