package dev.kursor.ktensorflow.npu

import cocoapods.TensorFlowLiteObjC.TFLCoreMLDelegate
import cocoapods.TensorFlowLiteObjC.TFLCoreMLDelegateOptions
import cocoapods.TensorFlowLiteObjC.TFLDelegate

internal class IosNpuDelegate(
    tflOptions: TFLCoreMLDelegateOptions
) : NpuDelegate {

    private var delegate: TFLDelegate? = try {
        TFLCoreMLDelegate(tflOptions)
    } catch (_: NullPointerException) {
        null
    }
    private var closed = false

    override val isAvailable: Boolean = delegate != null

    // Objective-C объект освобождает ARC, как только на него не остаётся ссылок: close
    // отпускает нашу, а закрытый делегат отклоняется так же, как на Android
    override val tflDelegate: TFLDelegate?
        get() {
            check(!closed) { "NPU delegate has already been closed" }
            return delegate
        }

    override fun close() {
        closed = true
        delegate = null
    }
}
