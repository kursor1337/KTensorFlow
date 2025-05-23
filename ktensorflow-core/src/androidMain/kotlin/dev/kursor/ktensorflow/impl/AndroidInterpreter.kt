package dev.kursor.ktensorflow.impl

import dev.kursor.ktensorflow.api.Interpreter
import dev.kursor.ktensorflow.api.InterpreterOptions
import dev.kursor.ktensorflow.api.Tensor
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

class AndroidInterpreter(
    filePath: String,
    options: InterpreterOptions
) : Interpreter {

    private val tensorFlowInterpreter = org.tensorflow.lite.Interpreter(
        File(filePath),
        options.toTensorFlowInterpreterOptions(),
    )

    @OptIn(ExperimentalUnsignedTypes::class)
    override fun run(inputs: List<Tensor>, outputs: List<Tensor>) {

        val inputsArray = inputs
            .map { ByteBuffer.wrap(it.data).apply { order(ByteOrder.nativeOrder()) } }
            .toTypedArray()
        val outputsArray = outputs.withIndex().associate {
            it.index to ByteBuffer.wrap(it.value.data).apply { order(ByteOrder.nativeOrder()) }
        }
        tensorFlowInterpreter.runForMultipleInputsOutputs(
            inputsArray,
            outputsArray
        )
    }

    override fun close() {
        tensorFlowInterpreter.close()
    }
}
