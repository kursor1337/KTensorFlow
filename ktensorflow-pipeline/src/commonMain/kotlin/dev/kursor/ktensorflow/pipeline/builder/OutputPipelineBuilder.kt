package dev.kursor.ktensorflow.pipeline.builder

import dev.kursor.ktensorflow.ExperimentalKTensorFlowApi
import dev.kursor.ktensorflow.Interpreter
import dev.kursor.ktensorflow.tensor.Tensor
import dev.kursor.ktensorflow.tensor.TensorShape
import dev.kursor.ktensorflow.pipeline.Pipeline
import dev.kursor.ktensorflow.pipeline.Tuple
import dev.kursor.ktensorflow.pipeline.stage.CombinedStage
import dev.kursor.ktensorflow.pipeline.stage.InferenceOutputData
import dev.kursor.ktensorflow.pipeline.stage.MultiInferenceStage
import dev.kursor.ktensorflow.pipeline.stage.Stage
import dev.kursor.ktensorflow.pipeline.stage.then
import dev.kursor.ktensorflow.tensor.TensorDataType
import kotlin.jvm.JvmName

/**
 * A builder that represents a complete pipeline that can be built into [Pipeline]
 */
@Suppress("UNCHECKED_CAST")
@ExperimentalKTensorFlowApi
class OutputPipelineBuilder<Input : Tuple, OutputInput : Tuple, Output : Tuple> internal constructor(
    internal val inputStage: Stage<Input, List<Tensor<*>>>,
    internal val interpreter: Interpreter,
    internal val outputData: List<PipelineOutputData<Any?, *>> = listOf()
) {

    /**
     * Builds the pipeline.
     */
    fun build(): Pipeline<Input, Output> {
        val inferenceStage = MultiInferenceStage(
            interpreter,
            outputData.map {
                InferenceOutputData(
                    index = it.index,
                    dataType = it.dataType,
                    shape = it.shape
                )
            }
        )

        val outputStage: Stage<OutputInput, Output> = outputData
            .map { it.stage }
            .let { CombinedStage(it as List<Stage<Any?, Any?>>) }

        return inputStage
            .then(inferenceStage)
            .then { it.toTuple<OutputInput>() }
            .then(outputStage)
            .let(::Pipeline)
    }
}

/**
 * Adds an output with postprocessing stage to the pipeline.
 *
 * @param index The index of the output tensor.
 * @param dataType The data type of the output tensor.
 * @param shape The shape of the output tensor.
 * @param postprocessing The postprocessing stage to apply to the output tensor.
 * @return An [OutputPipelineBuilder] with the added output.
 */
@ExperimentalKTensorFlowApi
fun <SO, Input : Tuple, T : Any> InferencePipelineBuilder<Input>.output(
    index: Int = 0,
    dataType: TensorDataType<T>,
    shape: TensorShape,
    postprocessing: Stage<Tensor<T>, SO>
): OutputPipelineBuilder<Input, Tuple.One<Tensor<T>>, Tuple.One<SO>> {
    return OutputPipelineBuilder(
        inputStage = inputStage,
        interpreter = interpreter,
        outputData = listOf(PipelineOutputData(index, dataType, shape, postprocessing))
    )
}

/**
 * Adds an output with postprocessing stage to the pipeline.
 *
 * @param index The index of the output tensor.
 * @param dataType The data type of the output tensor.
 * @param shape The shape of the output tensor.
 * @param postprocessing The postprocessing stage to apply to the output tensor.
 * @return An [OutputPipelineBuilder] with the added output.
 */
@ExperimentalKTensorFlowApi
@JvmName("output2")
fun <SO, Input : Tuple, O1, T1, T : Any> OutputPipelineBuilder<Input, Tuple.One<T1>, Tuple.One<O1>>.output(
    index: Int,
    dataType: TensorDataType<T>,
    shape: TensorShape,
    postprocessing: Stage<Tensor<T>, SO>
): OutputPipelineBuilder<Input, Tuple.Two<T1, Tensor<T>>, Tuple.Two<O1, SO>> {
    return OutputPipelineBuilder(
        inputStage = inputStage,
        interpreter = interpreter,
        outputData = outputData + PipelineOutputData(index, dataType, shape, postprocessing)
    )
}

/**
 * Adds an output with postprocessing stage to the pipeline.
 *
 * @param index The index of the output tensor.
 * @param dataType The data type of the output tensor.
 * @param shape The shape of the output tensor.
 * @param postprocessing The postprocessing stage to apply to the output tensor.
 * @return An [OutputPipelineBuilder] with the added output.
 */
@ExperimentalKTensorFlowApi
@JvmName("output3")
fun <SO, Input : Tuple, O1, O2, T1, T2, T : Any> OutputPipelineBuilder<Input, Tuple.Two<T1, T2>, Tuple.Two<O1, O2>>.output(
    index: Int,
    dataType: TensorDataType<T>,
    shape: TensorShape,
    postprocessing: Stage<Tensor<T>, SO>
): OutputPipelineBuilder<Input, Tuple.Three<T1, T2, Tensor<T>>, Tuple.Three<O1, O2, SO>> {
    return OutputPipelineBuilder(
        inputStage = inputStage,
        interpreter = interpreter,
        outputData = outputData + PipelineOutputData(index, dataType, shape, postprocessing)
    )
}

/**
 * Adds an output with postprocessing stage to the pipeline.
 *
 * @param index The index of the output tensor.
 * @param dataType The data type of the output tensor.
 * @param shape The shape of the output tensor.
 * @param postprocessing The postprocessing stage to apply to the output tensor.
 * @return An [OutputPipelineBuilder] with the added output.
 */
@ExperimentalKTensorFlowApi
@JvmName("output4")
fun <SO, Input : Tuple, O1, O2, O3, T1, T2, T3, T : Any> OutputPipelineBuilder<Input, Tuple.Three<T1, T2, T3>, Tuple.Three<O1, O2, O3>>.output(
    index: Int,
    dataType: TensorDataType<T>,
    shape: TensorShape,
    postprocessing: Stage<Tensor<T>, SO>
): OutputPipelineBuilder<Input, Tuple.Four<T1, T2, T3, Tensor<T>>, Tuple.Four<O1, O2, O3, SO>> {
    return OutputPipelineBuilder(
        inputStage = inputStage,
        interpreter = interpreter,
        outputData = outputData + PipelineOutputData(index, dataType, shape, postprocessing)
    )
}

/**
 * Adds an output with postprocessing stage to the pipeline.
 *
 * @param index The index of the output tensor.
 * @param dataType The data type of the output tensor.
 * @param shape The shape of the output tensor.
 * @param postprocessing The postprocessing stage to apply to the output tensor.
 * @return An [OutputPipelineBuilder] with the added output.
 */
@ExperimentalKTensorFlowApi
@JvmName("output5")
fun <SO, Input : Tuple, O1, O2, O3, O4, T1, T2, T3, T4, T : Any> OutputPipelineBuilder<Input, Tuple.Four<T1, T2, T3, T4>, Tuple.Four<O1, O2, O3, O4>>.output(
    index: Int,
    dataType: TensorDataType<T>,
    shape: TensorShape,
    postprocessing: Stage<Tensor<T>, SO>
): OutputPipelineBuilder<Input, Tuple.Five<T1, T2, T3, T4, Tensor<T>>, Tuple.Five<O1, O2, O3, O4, SO>> {
    return OutputPipelineBuilder(
        inputStage = inputStage,
        interpreter = interpreter,
        outputData = outputData + PipelineOutputData(index, dataType, shape, postprocessing)
    )
}

/**
 * Adds an output with postprocessing stage to the pipeline.
 *
 * @param index The index of the output tensor.
 * @param dataType The data type of the output tensor.
 * @param shape The shape of the output tensor.
 * @param postprocessing The postprocessing stage to apply to the output tensor.
 * @return An [OutputPipelineBuilder] with the added output.
 */
@ExperimentalKTensorFlowApi
@JvmName("output6")
fun <SO, Input : Tuple, O1, O2, O3, O4, O5, T1, T2, T3, T4, T5, T : Any> OutputPipelineBuilder<Input, Tuple.Five<T1, T2, T3, T4, T5>, Tuple.Five<O1, O2, O3, O4, O5>>.output(
    index: Int,
    dataType: TensorDataType<T>,
    shape: TensorShape,
    postprocessing: Stage<Tensor<T>, SO>
): OutputPipelineBuilder<Input, Tuple.Six<T1, T2, T3, T4, T5, Tensor<T>>, Tuple.Six<O1, O2, O3, O4, O5, SO>> {
    return OutputPipelineBuilder(
        inputStage = inputStage,
        interpreter = interpreter,
        outputData = outputData + PipelineOutputData(index, dataType, shape, postprocessing)
    )
}

/**
 * Adds an output with postprocessing stage to the pipeline.
 *
 * @param index The index of the output tensor.
 * @param dataType The data type of the output tensor.
 * @param shape The shape of the output tensor.
 * @param postprocessing The postprocessing stage to apply to the output tensor.
 * @return An [OutputPipelineBuilder] with the added output.
 */
@ExperimentalKTensorFlowApi
@JvmName("output7")
fun <SO, Input : Tuple, O1, O2, O3, O4, O5, O6, T1, T2, T3, T4, T5, T6, T : Any> OutputPipelineBuilder<Input, Tuple.Six<T1, T2, T3, T4, T5, T6>, Tuple.Six<O1, O2, O3, O4, O5, O6>>.output(
    index: Int,
    dataType: TensorDataType<T>,
    shape: TensorShape,
    postprocessing: Stage<Tensor<T>, SO>
): OutputPipelineBuilder<Input, Tuple.Seven<T1, T2, T3, T4, T5, T6, Tensor<T>>, Tuple.Seven<O1, O2, O3, O4, O5, O6, SO>> {
    return OutputPipelineBuilder(
        inputStage = inputStage,
        interpreter = interpreter,
        outputData = outputData + PipelineOutputData(index, dataType, shape, postprocessing)
    )
}

/**
 * Adds an output with postprocessing stage to the pipeline.
 *
 * @param index The index of the output tensor.
 * @param dataType The data type of the output tensor.
 * @param shape The shape of the output tensor.
 * @param postprocessing The postprocessing stage to apply to the output tensor.
 * @return An [OutputPipelineBuilder] with the added output.
 */
@ExperimentalKTensorFlowApi
@JvmName("output8")
fun <SO, Input : Tuple, O1, O2, O3, O4, O5, O6, O7, T1, T2, T3, T4, T5, T6, T7, T : Any> OutputPipelineBuilder<Input, Tuple.Seven<T1, T2, T3, T4, T5, T6, T7>, Tuple.Seven<O1, O2, O3, O4, O5, O6, O7>>.output(
    index: Int,
    dataType: TensorDataType<T>,
    shape: TensorShape,
    postprocessing: Stage<Tensor<T>, SO>
): OutputPipelineBuilder<Input, Tuple.Eight<T1, T2, T3, T4, T5, T6, T7, Tensor<T>>, Tuple.Eight<O1, O2, O3, O4, O5, O6, O7, SO>> {
    return OutputPipelineBuilder(
        inputStage = inputStage,
        interpreter = interpreter,
        outputData = outputData + PipelineOutputData(index, dataType, shape, postprocessing)
    )
}

/**
 * Adds an output with postprocessing stage to the pipeline.
 *
 * @param index The index of the output tensor.
 * @param dataType The data type of the output tensor.
 * @param shape The shape of the output tensor.
 * @param postprocessing The postprocessing stage to apply to the output tensor.
 * @return An [OutputPipelineBuilder] with the added output.
 */
@ExperimentalKTensorFlowApi
@JvmName("output9")
fun <SO, Input : Tuple, O1, O2, O3, O4, O5, O6, O7, O8, T1, T2, T3, T4, T5, T6, T7, T8, T : Any> OutputPipelineBuilder<Input, Tuple.Eight<T1, T2, T3, T4, T5, T6, T7, T8>, Tuple.Eight<O1, O2, O3, O4, O5, O6, O7, O8>>.output(
    index: Int,
    dataType: TensorDataType<T>,
    shape: TensorShape,
    postprocessing: Stage<Tensor<T>, SO>
): OutputPipelineBuilder<Input, Tuple.Nine<T1, T2, T3, T4, T5, T6, T7, T8, Tensor<T>>, Tuple.Nine<O1, O2, O3, O4, O5, O6, O7, O8, SO>> {
    return OutputPipelineBuilder(
        inputStage = inputStage,
        interpreter = interpreter,
        outputData = outputData + PipelineOutputData(index, dataType, shape, postprocessing)
    )
}

/**
 * Adds an output with postprocessing stage to the pipeline.
 *
 * @param index The index of the output tensor.
 * @param dataType The data type of the output tensor.
 * @param shape The shape of the output tensor.
 * @param postprocessing The postprocessing stage to apply to the output tensor.
 * @return An [OutputPipelineBuilder] with the added output.
 */
@ExperimentalKTensorFlowApi
@JvmName("output10")
fun <SO, Input : Tuple, O1, O2, O3, O4, O5, O6, O7, O8, O9, T1, T2, T3, T4, T5, T6, T7, T8, T9, T : Any> OutputPipelineBuilder<Input, Tuple.Nine<T1, T2, T3, T4, T5, T6, T7, T8, T9>, Tuple.Nine<O1, O2, O3, O4, O5, O6, O7, O8, O9>>.output(
    index: Int,
    dataType: TensorDataType<T>,
    shape: TensorShape,
    postprocessing: Stage<Tensor<T>, SO>
): OutputPipelineBuilder<Input, Tuple.Ten<T1, T2, T3, T4, T5, T6, T7, T8, T9, Tensor<T>>, Tuple.Ten<O1, O2, O3, O4, O5, O6, O7, O8, O9, SO>> {
    return OutputPipelineBuilder(
        inputStage = inputStage,
        interpreter = interpreter,
        outputData = outputData + PipelineOutputData(index, dataType, shape, postprocessing)
    )
}

@ExperimentalKTensorFlowApi
internal data class PipelineOutputData<Output, T : Any>(
    val index: Int,
    val dataType: TensorDataType<T>,
    val shape: TensorShape,
    val stage: Stage<Tensor<T>, Output>
)

@Suppress("UNCHECKED_CAST")
@ExperimentalKTensorFlowApi
internal fun <T : Tuple> Map<Int, Tensor<*>>.toTuple(): T {
    val list = this.toList().sortedBy { it.first }.map { it.second }
    return when (list.size) {
        0 -> Tuple.Zero
        1 -> Tuple.One(list[0])
        2 -> Tuple.Two(list[0], list[1])
        3 -> Tuple.Three(list[0], list[1], list[2])
        4 -> Tuple.Four(list[0], list[1], list[2], list[3])
        5 -> Tuple.Five(list[0], list[1], list[2], list[3], list[4])
        6 -> Tuple.Six(list[0], list[1], list[2], list[3], list[4], list[5])
        7 -> Tuple.Seven(list[0], list[1], list[2], list[3], list[4], list[5], list[6])
        8 -> Tuple.Eight(list[0], list[1], list[2], list[3], list[4], list[5], list[6], list[7])
        9 -> Tuple.Nine(list[0], list[1], list[2], list[3], list[4], list[5], list[6], list[7], list[8])
        10 -> Tuple.Ten(list[0], list[1], list[2], list[3], list[4], list[5], list[6], list[7], list[8], list[9])
        else -> error("Unsupported tuple size: ${list.size}")
    } as T
}