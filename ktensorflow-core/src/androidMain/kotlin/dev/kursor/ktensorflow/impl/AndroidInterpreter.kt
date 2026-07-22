package dev.kursor.ktensorflow.impl

import dev.kursor.ktensorflow.Interpreter
import dev.kursor.ktensorflow.InterpreterOptions
import dev.kursor.ktensorflow.ModelDesc
import dev.kursor.ktensorflow.ModelTensorData
import dev.kursor.ktensorflow.ModelMeta
import dev.kursor.ktensorflow.toKTensorFlow
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.tensorflow.lite.Interpreter as TFLInterpreter

internal class AndroidInterpreter(
    modelDesc: ModelDesc,
    options: InterpreterOptions
) : Interpreter {

    private val tensorFlowInterpreter = when (modelDesc) {
        is ModelDesc.ByteBuffer -> TFLInterpreter(
            modelDesc.buffer,
            options.tflOptions,
        )

        is ModelDesc.File -> TFLInterpreter(
            modelDesc.file,
            options.tflOptions,
        )
    }

    override val inputTensorCount: Int
        get() = tensorFlowInterpreter.inputTensorCount

    override val outputTensorCount: Int
        get() = tensorFlowInterpreter.outputTensorCount

    override fun getModelMeta(): ModelMeta {
        val inputs = (0..<inputTensorCount)
            .map(tensorFlowInterpreter::getInputTensor)
            .map {
                ModelTensorData(
                    index = it.index(),
                    name = it.name(),
                    dataType = it.dataType().toKTensorFlow(),
                    shape = it.shape().toList()
                )
            }

        val outputs = (0..<outputTensorCount)
            .map(tensorFlowInterpreter::getOutputTensor)
            .map {
                ModelTensorData(
                    index = it.index(),
                    name = it.name(),
                    dataType = it.dataType().toKTensorFlow(),
                    shape = it.shape().toList()
                )
            }

        return ModelMeta(inputs, outputs)
    }

    override fun resizeInput(index: Int, dims: IntArray) {
        tensorFlowInterpreter.resizeInput(index, dims)
        tensorFlowInterpreter.allocateTensors()
    }

    @OptIn(ExperimentalUnsignedTypes::class)
    override fun run(inputs: List<ByteArray>, outputs: Map<Int, ByteArray>) {
        val inputsArray = inputs
            .map {
                ByteBuffer.wrap(it)
                    .apply { order(ByteOrder.nativeOrder()) }
            }
            .toTypedArray()
        val outputsArray = outputs.mapValues {
            ByteBuffer.wrap(it.value)
                .apply { order(ByteOrder.nativeOrder()) }
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