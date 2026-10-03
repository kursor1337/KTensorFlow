package dev.kursor.ktensorflow.npu

import android.os.Build
import org.tensorflow.lite.Delegate
import org.tensorflow.lite.nnapi.NnApiDelegate

internal class AndroidNpuDelegate(
    options: NnApiDelegate.Options
) : NpuDelegate {

    private var delegate: NnApiDelegate? =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
            NnApiDelegate(options)
        } else {
            null
        }

    private var closed = false

    override val isAvailable: Boolean = delegate != null

    override val tflDelegate: Delegate?
        @Synchronized get() {
            check(!closed) { "NPU delegate has already been closed" }
            return delegate
        }

    @Synchronized
    override fun close() {
        closed = true
        delegate?.close()
        delegate = null
    }
}
