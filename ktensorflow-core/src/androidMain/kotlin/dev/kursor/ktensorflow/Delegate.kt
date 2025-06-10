package dev.kursor.ktensorflow

import org.tensorflow.lite.Delegate as TFLDelegate

actual interface Delegate {

    actual val isAvailable: Boolean

    val tflDelegate: TFLDelegate
}
