package dev.kursor.ktensorflow.impl

import cocoapods.TensorFlowLiteObjC.TFLInterpreter
import cocoapods.TensorFlowLiteObjC.TFLTensor
import dev.kursor.ktensorflow.Interpreter
import dev.kursor.ktensorflow.InterpreterOptions
import dev.kursor.ktensorflow.ModelDesc
import dev.kursor.ktensorflow.ModelMeta
import dev.kursor.ktensorflow.ModelTensorData
import dev.kursor.ktensorflow.toKTensorFlow
import kotlin.math.min

// private val on options because it is required to keep references so that they are not
// garbage collected since kotlin gc does not know if objects are passed to obj-c
internal class IosInterpreter(
    modelDesc: ModelDesc,
    private val options: InterpreterOptions
) : Interpreter {

    private val tflInterpreter: TFLInterpreter = checkError { errPtr ->
        when (modelDesc) {
            is ModelDesc.PathInBundle -> {
                TFLInterpreter(
                    modelPath = modelDesc.pathInBundle,
                    options = options.tflOptions,
                    error = errPtr,
                    delegates = options.tflDelegates
                )
            }
        }
    }

    init {
        checkError { errPtr ->
            tflInterpreter.allocateTensorsWithError(errPtr)
        }
    }

    override val inputTensorCount: Int
        get() = tflInterpreter.inputTensorCount.toInt()

    override val outputTensorCount: Int
        get() = tflInterpreter.outputTensorCount.toInt()

    override fun getModelMeta(): ModelMeta {
        val rawInputs = (0 until inputTensorCount).map { i ->
            i to getInputTensor(i)
        }
        val rawOutputs = (0 until outputTensorCount).map { i ->
            i to getOutputTensor(i)
        }

        @Suppress("UNCHECKED_CAST")
        val signatureKeys = tflInterpreter.signatureKeys() as? List<String>
        val defaultSignature = signatureKeys?.firstOrNull()

        if (defaultSignature != null) {
            val runner = checkError { err ->
                tflInterpreter.signatureRunnerWithKey(defaultSignature, err)
            }

            @Suppress("UNCHECKED_CAST")
            val sigInputs = runner.inputs() as? List<String> ?: emptyList()
            val inputs = sigInputs.map { sigName ->
                val sigTensor = checkError { err ->
                    runner.inputTensorWithName(sigName, err)
                }
                val internalName = sigTensor.name()
                val index = rawInputs.first { it.second.name() == internalName }.first

                ModelTensorData(
                    index = index,
                    name = sigName,
                    internalName = sigTensor.name(),
                    dataType = sigTensor.dataType().toKTensorFlow(),
                    shape = sigTensor.shape().map { (it as Number).toInt() }
                )
            }

            @Suppress("UNCHECKED_CAST")
            val sigOutputs = runner.outputs() as? List<String> ?: emptyList()
            val outputs = sigOutputs.map { sigName ->
                val sigTensor = checkError { err ->
                    runner.outputTensorWithName(sigName, err)
                }
                val internalName = sigTensor.name()
                val index = rawOutputs.first { it.second.name() == internalName }.first

                ModelTensorData(
                    index = index,
                    name = sigName,
                    internalName = sigTensor.name(),
                    dataType = sigTensor.dataType().toKTensorFlow(),
                    shape = sigTensor.shape().map { (it as Number).toInt() }
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
                    shape = tensor.shape().map { (it as Number).toInt() }
                )
            },
            outputData = rawOutputs.map { (index, tensor) ->
                ModelTensorData(
                    index = index,
                    name = tensor.name(),
                    internalName = tensor.name(),
                    dataType = tensor.dataType().toKTensorFlow(),
                    shape = tensor.shape().map { (it as Number).toInt() }
                )
            }
        )
    }

    private fun getInputTensor(index: Int): TFLTensor {
        return checkError { errPtr ->
            tflInterpreter.inputTensorAtIndex(index.toULong(), errPtr)
        }
    }

    private fun getOutputTensor(index: Int): TFLTensor {
        return checkError { errPtr ->
            tflInterpreter.outputTensorAtIndex(index.toULong(), errPtr)
        }
    }

    override fun resizeInput(index: Int, dims: IntArray) {
        checkError { errPtr ->
            tflInterpreter.resizeInputTensorAtIndex(
                index.toULong(),
                dims.toList(),
                errPtr
            )
        }
        checkError { errPtr ->
            tflInterpreter.allocateTensorsWithError(errPtr)
        }
    }

    override fun run(
        inputs: List<ByteArray>,
        outputs: Map<Int, ByteArray>
    ) {
        if (inputs.size > tflInterpreter.inputTensorCount().toInt()) {
            throw IllegalArgumentException("Wrong inputs dimension.")
        }

        inputs.forEachIndexed { index, input ->
            val inputTensor = getInputTensor(index)
            val data = input.toNSData()
            checkError { errPtr ->
                inputTensor.copyData(data, errPtr)
            }
            Unit
        }
        checkError { errPtr ->
            tflInterpreter.invokeWithError(errPtr)
        }

        for (entry in outputs) {
            val (i, byteArray) = entry
            val outputTensor = getOutputTensor(i)

            val array = checkError { errPtr ->
                outputTensor.dataWithError(errPtr)
            }
                .toByteArray()

            array.copyInto(
                destination = byteArray,
                endIndex = min(array.size, byteArray.size)
            )
        }
    }

    override fun close() {
        // do nothing
    }
}

fun TFLTensor.shape(): List<Int> {
    return checkError { errPtr ->
        this.shapeWithError(errPtr)
    }.map { (it as Number).toInt() }
}