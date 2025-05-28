package dev.kursor.ktensorflow.impl

import cocoapods.TensorFlowLiteObjC.TFLDelegate
import cocoapods.TensorFlowLiteObjC.TFLMetalDelegate
import cocoapods.TensorFlowLiteObjC.TFLMetalDelegateOptions
import dev.kursor.ktensorflow.api.gpu.GpuDelegate

internal class IosGpuDelegate(
    private val tflOptions: TFLMetalDelegateOptions
) : GpuDelegate {

    override val tflDelegate: TFLDelegate? = try {
        TFLMetalDelegate(tflOptions)
    } catch (_: NullPointerException) {
        null
    }
    override val isAvailable: Boolean = tflDelegate != null
}