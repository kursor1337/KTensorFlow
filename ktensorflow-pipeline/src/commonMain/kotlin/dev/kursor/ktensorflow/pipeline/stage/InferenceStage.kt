package dev.kursor.ktensorflow.pipeline.stage

import dev.kursor.ktensorflow.ExperimentalKTensorFlowApi
import dev.kursor.ktensorflow.Interpreter
import dev.kursor.ktensorflow.tensor.Tensor
import dev.kursor.ktensorflow.tensor.TensorDataType
import dev.kursor.ktensorflow.tensor.TensorShape
import dev.kursor.ktensorflow.tensor.run
import kotlin.reflect.KClass

/**
 * Represents an output data of the inference stage.
 *
 * @property index The index of the output tensor.
 * @property dataType The data type of the output tensor.
 * @property shape The shape of the output tensor.
 */
@ExperimentalKTensorFlowApi
data class InferenceOutputData<T : Any>(
    val index: Int,
    val dataType: TensorDataType<T>,
    val shape: TensorShape
)

/**
 * Represents a single input and single output inference stage.
 */
@ExperimentalKTensorFlowApi
class SingleInferenceStage<T : Any>(
    private val interpreter: Interpreter,
    private val output: InferenceOutputData<T>
) : Stage<Tensor<*>, Tensor<T>> {
    override fun run(input: Tensor<*>): Tensor<T> {
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
    private val outputs: List<InferenceOutputData<*>>
) : Stage<List<Tensor<*>>, Map<Int, Tensor<*>>> {

    override fun run(input: List<Tensor<*>>): Map<Int, Tensor<*>> {
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
fun <Input, T : Any> Stage<Input, Tensor<T>>.inference(
    interpreter: Interpreter,
    dataType: TensorDataType<T>,
    shape: TensorShape,
    index: Int = 0
): Stage<Input, Tensor<T>> = this.then(
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
fun <Input> Stage<Input, List<Tensor<*>>>.inference(
    interpreter: Interpreter,
    outputs: List<InferenceOutputData<*>>
): Stage<Input, Map<Int, Tensor<*>>> = this.then(
    MultiInferenceStage(interpreter, outputs)
)

@ExperimentalKTensorFlowApi
internal fun <T : Any> InferenceOutputData<T>.toTensor() = Tensor(dataType, shape)