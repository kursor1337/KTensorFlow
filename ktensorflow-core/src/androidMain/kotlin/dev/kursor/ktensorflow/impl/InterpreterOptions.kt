package dev.kursor.ktensorflow.impl

import com.qualcomm.qti.QnnDelegate
import dev.kursor.ktensorflow.api.Hardware
import dev.kursor.ktensorflow.api.InterpreterOptions
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate

internal fun InterpreterOptions.toTensorFlowInterpreterOptions() = Interpreter.Options().apply {
    hardwarePriorities.forEach { hardware ->
        val result = when (hardware) {
            Hardware.CPU -> trySetupCpu()
            Hardware.GPU -> trySetupGpu()
            // Hardware.NPU -> trySetupNpu()
        }
        if (result) {
            return@forEach
        }
    }
}

private fun Interpreter.Options.trySetupCpu(): Boolean {
    setUseXNNPACK(useXNNPACK)
    setNumThreads(numThreads)
    return true
}

private fun Interpreter.Options.trySetupGpu(): Boolean {
    val compatList = CompatibilityList()
    if (compatList.isDelegateSupportedOnThisDevice) {
        addDelegate(GpuDelegate(compatList.bestOptionsForThisDevice))
        return true
    } else {
        return false
    }
}

private fun Interpreter.Options.trySetupNpu(): Boolean {
    val isAvailable = QnnDelegate.Capability.entries.any {
        QnnDelegate.checkCapability(it)
    }
    if (isAvailable) {
        addDelegate(QnnDelegate(QnnDelegate.Options()))
        return true
    } else {
        return false
    }
}
