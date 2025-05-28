package dev.kursor.ktensorflow.api

import cocoapods.TensorFlowLiteObjC.TFLDelegate

actual interface Delegate {
    actual val isAvailable: Boolean

    val tflDelegate: TFLDelegate?
}