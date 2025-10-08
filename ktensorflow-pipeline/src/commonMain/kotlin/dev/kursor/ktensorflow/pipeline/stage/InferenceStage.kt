package dev.kursor.ktensorflow.pipeline.stage

import dev.kursor.ktensorflow.ExperimentalKTensorFlowApi
import dev.kursor.ktensorflow.Interpreter
import dev.kursor.ktensorflow.Tensor
import dev.kursor.ktensorflow.TensorDataType
import dev.kursor.ktensorflow.TensorShape

@ExperimentalKTensorFlowApi
data class InferenceOutputData(
    val index: Int,
    val dataType: TensorDataType,
    val shape: TensorShape
)

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

@ExperimentalKTensorFlowApi
fun <Input> Stage<Input, List<Tensor>>.inference(
    interpreter: Interpreter,
    outputs: List<InferenceOutputData>
): Stage<Input, Map<Int, Tensor>> = this.then(
    MultiInferenceStage(interpreter, outputs)
)

@ExperimentalKTensorFlowApi
internal fun InferenceOutputData.toTensor() = Tensor(dataType, shape)