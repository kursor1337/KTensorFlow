package dev.kursor.ktensorflow.impl

import dev.kursor.ktensorflow.Interpreter
import dev.kursor.ktensorflow.InterpreterOptions
import dev.kursor.ktensorflow.ModelDesc
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

    init {
        tensorFlowInterpreter.allocateTensors()

        val outputCount = tensorFlowInterpreter.outputTensorCount

        println("=== СКАНИРОВАНИЕ ВЫХОДОВ TFLITE НА IOS ===")
        for (i in 0 until outputCount) {
            val tensor = tensorFlowInterpreter.getOutputTensor(i)

            println("Anddroid Index: $i | Name: ${tensor.name()} | Real Byte Size: ${tensor.numBytes()}")
        }
        println("=========================================")
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