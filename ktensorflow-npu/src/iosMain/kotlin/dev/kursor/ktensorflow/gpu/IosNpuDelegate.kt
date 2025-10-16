package dev.kursor.ktensorflow.gpu

import cocoapods.TensorFlowLiteObjC.TFLDelegate
import cocoapods.TensorFlowLiteObjC.TFLCoreMLDelegate
import cocoapods.TensorFlowLiteObjC.TFLCoreMLDelegateOptions

internal class IosNpuDelegate(
    private val tflOptions: TFLCoreMLDelegateOptions
) : NpuDelegate {

    override val tflDelegate: TFLDelegate? = try {
        TFLCoreMLDelegate(tflOptions)
    } catch (_: NullPointerException) {
        null
    }
    override val isAvailable: Boolean = tflDelegate != null
}