package dev.kursor.ktensorflow.api

import org.tensorflow.lite.Delegate as TFLDelegate

actual interface Delegate {

    actual val isAvailable: Boolean

    val tflDelegate: TFLDelegate
}
