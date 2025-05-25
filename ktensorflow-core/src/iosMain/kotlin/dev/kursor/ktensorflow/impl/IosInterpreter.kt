package dev.kursor.ktensorflow.impl

import dev.kursor.ktensorflow.api.Interpreter
import dev.kursor.ktensorflow.api.InterpreterOptions
import dev.kursor.ktensorflow.api.Tensor
import cocoapods.TensorFlowLiteObjC.TFLInterpreter
import cocoapods.TensorFlowLiteObjC.TFLTensor
import dev.kursor.ktensorflow.api.ModelDesc

class IosInterpreter(
    modelDesc: ModelDesc,
    options: InterpreterOptions
) : Interpreter {

    // it is required to keep references so they are not garbage collected
    // since kotlin gc does not know if objects are passed to obj-c
    private val tflDelegate = options.hardwarePriorities.toTflDelegate()
    private val tflOptions = options.toTensorFlowInterpreterOptions()

    private val tflInterpreter: TFLInterpreter = checkError { errPtr ->
        when (modelDesc) {
            is ModelDesc.PathInBundle -> {
                TFLInterpreter(
                    modelPath = modelDesc.pathInBundle,
                    options = tflOptions,
                    error = errPtr,
                    delegates = tflDelegate
                )
            }
        }
    }

    init {
        checkError { errPtr ->
            tflInterpreter.allocateTensorsWithError(errPtr)
        }
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

    override fun run(
        inputs: List<Tensor>,
        outputs: List<Tensor>
    ) {
        if (inputs.size > tflInterpreter.inputTensorCount().toInt()) {
            throw IllegalArgumentException("Wrong inputs dimension.")
        }

        inputs.forEachIndexed { index, input ->
            val inputTensor = getInputTensor(index)
            val data = input.data.toNSData()
            checkError { errPtr ->
                inputTensor.copyData(data, errPtr)
            }
            Unit
        }
        checkError { errPtr ->
            tflInterpreter.invokeWithError(errPtr)
        }

        for (i in 0 until tflInterpreter.outputTensorCount().toInt()) {
            val outputTensor = getOutputTensor(i)

            val array = checkError { errPtr ->
                outputTensor.dataWithError(errPtr)
            }
                .toByteArray()

            array.copyInto(destination = outputs[i].data)
        }
    }

    override fun close() {
        // do nothing
    }
}

