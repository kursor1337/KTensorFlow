package dev.kursor.ktensorflow.pipeline

import dev.kursor.ktensorflow.Interpreter
import dev.kursor.ktensorflow.Tensor
import dev.kursor.ktensorflow.TensorDataType
import dev.kursor.ktensorflow.TensorShape

class PipelineInput<in Input : Any>(
    val index: Int,
    val stage: Stage<Input, Tensor>
)

class PipelineOutput<out Output : Any>(
    val index: Int,
    val dataType: TensorDataType,
    val shape: TensorShape,
    val stage: Stage<Tensor, Output>
)

class Pipeline(
    inputs: List<PipelineInput<Any>>,
    outputs: List<PipelineOutput<Any>>,
    private val interpreter: Interpreter
) {

    val inputs = inputs.sortedBy { it.index }
    val outputs = outputs.sortedBy { it.index }

    fun run(inputs: List<Any>): List<Any> {
        val modelInputs = this.inputs
            .zip(inputs)
            .map { (pipelineInput, data) ->
                pipelineInput.stage.run(data)
            }

        val modelOutputs = outputs
            .map {
                Tensor(
                    dataType = it.dataType,
                    shape = it.shape
                )
            }


        interpreter.run(modelInputs, modelOutputs)

        return modelOutputs
            .zip(outputs)
            .map { (tensor, output) ->
                output.stage.run(tensor)
            }
    }
}