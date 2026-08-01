package dev.kursor.ktensorflow.impl

import dev.kursor.ktensorflow.Interpreter
import dev.kursor.ktensorflow.InterpreterOptions
import dev.kursor.ktensorflow.ModelDesc
import dev.kursor.ktensorflow.ModelMeta
import dev.kursor.ktensorflow.ModelTensorData
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
        val rawInputs = (0 until inputTensorCount).map { i ->
            i to tensorFlowInterpreter.getInputTensor(i)
        }
        val rawOutputs = (0 until outputTensorCount).map { i ->
            i to tensorFlowInterpreter.getOutputTensor(i)
        }

        val signatureKeys = tensorFlowInterpreter.signatureKeys
        val defaultSignature = signatureKeys.firstOrNull()

        if (defaultSignature != null) {
            val inputs = tensorFlowInterpreter
                .getSignatureInputs(defaultSignature)
                .map { sigName ->
                    val sigTensor = tensorFlowInterpreter
                        .getInputTensorFromSignature(
                            sigName,
                            defaultSignature
                        )

                    val index = rawInputs
                        .first { it.second.name() == sigTensor.name() }
                        .first

                    ModelTensorData(
                        index = index,
                        name = sigName,
                        internalName = sigTensor.name(),
                        dataType = sigTensor.dataType().toKTensorFlow(),
                        shape = sigTensor.shape().toList()
                    )
                }

            val outputs = tensorFlowInterpreter
                .getSignatureOutputs(defaultSignature)
                .map { sigName ->
                    val sigTensor = tensorFlowInterpreter
                        .getOutputTensorFromSignature(
                            sigName,
                            defaultSignature
                        )

                    val index = rawOutputs
                        .first { it.second.name() == sigTensor.name() }
                        .first

                    ModelTensorData(
                        index = index,
                        name = sigName,
                        internalName = sigTensor.name(),
                        dataType = sigTensor.dataType().toKTensorFlow(),
                        shape = sigTensor.shape().toList()
                    )
                }

            return ModelMeta(inputs, outputs)
        }

        return ModelMeta(
            inputData = rawInputs.map { (index, tensor) ->
                ModelTensorData(
                    index = index,
                    name = tensor.name(),
                    internalName = tensor.name(),
                    dataType = tensor.dataType().toKTensorFlow(),
                    shape = tensor.shape().toList()
                )
            },
            outputData = rawOutputs.map { (index, tensor) ->
                ModelTensorData(
                    index = index,
                    name = tensor.name(),
                    internalName = tensor.name(),
                    dataType = tensor.dataType().toKTensorFlow(),
                    shape = tensor.shape().toList()
                )
            }
        )
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