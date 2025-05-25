package dev.kursor.ktensorflow.impl

import dev.kursor.ktensorflow.api.Interpreter
import dev.kursor.ktensorflow.api.InterpreterOptions
import dev.kursor.ktensorflow.api.ModelDesc
import dev.kursor.ktensorflow.api.Tensor
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.tensorflow.lite.Interpreter as TFLInterpreter

class AndroidInterpreter(
    modelDesc: ModelDesc,
    options: InterpreterOptions
) : Interpreter {

    private val tensorFlowInterpreter = when (modelDesc) {
        is ModelDesc.ByteBuffer -> TFLInterpreter(
            modelDesc.buffer,
            options.toTensorFlowInterpreterOptions(),
        )
        is ModelDesc.File -> TFLInterpreter(
            modelDesc.file,
            options.toTensorFlowInterpreterOptions(),
        )
    }

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
