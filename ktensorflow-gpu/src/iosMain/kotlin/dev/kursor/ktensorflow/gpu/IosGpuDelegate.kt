package dev.kursor.ktensorflow.gpu

import cocoapods.TensorFlowLiteObjC.TFLDelegate
import cocoapods.TensorFlowLiteObjC.TFLMetalDelegate
import cocoapods.TensorFlowLiteObjC.TFLMetalDelegateOptions

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