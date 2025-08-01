package dev.kursor.ktensorflow

import cocoapods.TensorFlowLiteObjC.TFLDelegate

actual interface Delegate {
    actual val isAvailable: Boolean

    val tflDelegate: TFLDelegate?
}