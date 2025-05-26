package dev.kursor.ktensorflow.impl

import cocoapods.TensorFlowLiteObjC.TFLDelegate
import cocoapods.TensorFlowLiteObjC.TFLInterpreterOptions
import cocoapods.TensorFlowLiteObjC.TFLMetalDelegate
import dev.kursor.ktensorflow.api.Hardware
import dev.kursor.ktensorflow.api.InterpreterOptions

internal fun InterpreterOptions.toTensorFlowInterpreterOptions() = TFLInterpreterOptions().apply {
    setNumberOfThreads(numThreads.toULong())
    setUseXNNPACK(useXNNPACK)
}

internal fun List<Hardware>.toTflDelegate(): List<TFLDelegate> {
    return firstNotNullOfOrNull { hardware ->
        runCatching {
            when (hardware) {
                // Hardware.NPU -> TFLCoreMLDelegate()
                Hardware.GPU -> TFLMetalDelegate()
                Hardware.CPU -> null
            }
        }
            .getOrNull()
    }
        ?.let { listOf(it) }
        ?: emptyList()
}
