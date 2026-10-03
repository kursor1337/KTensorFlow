package dev.kursor.ktensorflow

import org.tensorflow.lite.Delegate as TFLDelegate

actual interface Delegate : AutoCloseable {

    actual val isAvailable: Boolean

    /**
     * Native TensorFlow Lite delegate, or null if the delegate is not supported on this device.
     *
     * @throws IllegalStateException if the delegate has already been closed.
     */
    val tflDelegate: TFLDelegate?

    actual override fun close()
}
