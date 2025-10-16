package dev.kursor.ktensorflow.gpu

import android.os.Build
import org.tensorflow.lite.Delegate
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.nnapi.NnApiDelegate

internal class AndroidNpuDelegate(
    val options: NnApiDelegate.Options
) : NpuDelegate {
    private val compatList = CompatibilityList()

    override val isAvailable: Boolean = Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1 &&
            compatList.isDelegateSupportedOnThisDevice

    override val tflDelegate: Delegate by lazy {
        NnApiDelegate(options)
    }
}