package dev.kursor.ktensorflow.npu

import org.tensorflow.lite.Delegate
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegateFactory

internal class AndroidGpuDelegate(
    val options: GpuDelegateFactory.Options
) : GpuDelegate {
    private val compatList = CompatibilityList()

    override val isAvailable: Boolean = compatList.isDelegateSupportedOnThisDevice

    override val tflDelegate: Delegate by lazy {
        org.tensorflow.lite.gpu.GpuDelegate(options)
    }
}