package dev.kursor.ktensorflow.gpu

import cocoapods.TensorFlowLiteObjC.TFLDelegate
import cocoapods.TensorFlowLiteObjC.TFLMetalDelegate
import cocoapods.TensorFlowLiteObjC.TFLMetalDelegateOptions

internal class IosGpuDelegate(
    tflOptions: TFLMetalDelegateOptions
) : GpuDelegate {

    private var delegate: TFLDelegate? = try {
        TFLMetalDelegate(tflOptions)
    } catch (_: NullPointerException) {
        null
    }
    private var closed = false

    override val isAvailable: Boolean = delegate != null

    // Objective-C объект освобождает ARC, как только на него не остаётся ссылок: close
    // отпускает нашу, а закрытый делегат отклоняется так же, как на Android
    override val tflDelegate: TFLDelegate?
        get() {
            check(!closed) { "GPU delegate has already been closed" }
            return delegate
        }

    override fun close() {
        closed = true
        delegate = null
    }
}
