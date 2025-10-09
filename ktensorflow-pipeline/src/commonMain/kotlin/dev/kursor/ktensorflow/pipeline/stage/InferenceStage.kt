package dev.kursor.ktensorflow.pipeline.stage

import dev.kursor.ktensorflow.ExperimentalKTensorFlowApi
import dev.kursor.ktensorflow.Interpreter
import dev.kursor.ktensorflow.Tensor
import dev.kursor.ktensorflow.TensorDataType
import dev.kursor.ktensorflow.TensorShape

/**
 * Represents an output data of the inference stage.
 *
 * @property index The index of the output tensor.
 * @property dataType The data type of the output tensor.
 * @property shape The shape of the output tensor.
 */
@ExperimentalKTensorFlowApi
data class InferenceOutputData(
    val index: Int,
    val dataType: TensorDataType,
    val shape: TensorShape
)

/**
 * Represents a single input and single output inference stage.
 */
@ExperimentalKTensorFlowApi
class SingleInferenceStage(
    private val interpreter: Interpreter,
    private val output: InferenceOutputData
) : Stage<Tensor, Tensor> {
    override fun run(input: Tensor): Tensor {
        val outputTensor = mapOf(output.index to output.toTensor())
        interpreter.run(listOf(input), outputTensor)
        return outputTensor[output.index]!!
    }
}

/**
 * Represents a multiple input and multiple output inference stage.
 */
@ExperimentalKTensorFlowApi
class MultiInferenceStage(
    private val interpreter: Interpreter,
    private val outputs: List<InferenceOutputData>
) : Stage<List<Tensor>, Map<Int, Tensor>> {

    override fun run(input: List<Tensor>): Map<Int, Tensor> {
        val outputTensors = outputs.associate {
            it.index to it.toTensor()
        }
        interpreter.run(input, outputTensors)
        return outputTensors
    }
}

/**
 * Adds a [SingleInferenceStage] to a [Stage].
 *
 * @param interpreter The interpreter to use for inference.
 * @param dataType The data type of the output tensor.
 * @param shape The shape of the output tensor.
 * @param index The index of the output tensor.
 */
@ExperimentalKTensorFlowApi
fun <Input> Stage<Input, Tensor>.inference(
    interpreter: Interpreter,
    dataType: TensorDataType,
    shape: TensorShape,
    index: Int = 0
): Stage<Input, Tensor> = this.then(
    SingleInferenceStage(
        interpreter,
        InferenceOutputData(
            index,
            dataType,
            shape
        )
    )
)

/**
 * Adds a [MultiInferenceStage] to a [Stage].
 *
 * @param interpreter The interpreter to use for inference.
 * @param outputs The list of output data.
 */
@ExperimentalKTensorFlowApi
fun <Input> Stage<Input, List<Tensor>>.inference(
    interpreter: Interpreter,
    outputs: List<InferenceOutputData>
): Stage<Input, Map<Int, Tensor>> = this.then(
    MultiInferenceStage(interpreter, outputs)
)

@ExperimentalKTensorFlowApi
internal fun InferenceOutputData.toTensor() = Tensor(dataType, shape)