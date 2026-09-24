package dev.kursor.ktensorflow.npu

import android.os.Build
import org.tensorflow.lite.Delegate
import org.tensorflow.lite.nnapi.NnApiDelegate

internal class AndroidNpuDelegate(
    options: NnApiDelegate.Options
) : NpuDelegate {

    override val tflDelegate: Delegate? =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) NnApiDelegate(options) else null

    override val isAvailable: Boolean = tflDelegate != null
}
