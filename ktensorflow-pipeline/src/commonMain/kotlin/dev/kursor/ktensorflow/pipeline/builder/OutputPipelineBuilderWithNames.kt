package dev.kursor.ktensorflow.pipeline.builder

import dev.kursor.ktensorflow.ExperimentalKTensorFlowApi
import dev.kursor.ktensorflow.pipeline.Tuple
import dev.kursor.ktensorflow.pipeline.stage.Stage
import dev.kursor.ktensorflow.tensor.Tensor
import dev.kursor.ktensorflow.tensor.TensorDataType
import dev.kursor.ktensorflow.tensor.TensorShape
import kotlin.jvm.JvmName

/**
 * Adds an output with postprocessing stage to the pipeline using the output tensor name.
 *
 * @param name The name of the output tensor.
 * @param dataType The data type of the output tensor.
 * @param shape The shape of the output tensor.
 * @param postprocessing The postprocessing stage to apply to the output tensor.
 * @return An [OutputPipelineBuilder] with the added output.
 */
@ExperimentalKTensorFlowApi
fun <SO, Input : Tuple, T : Any> InferencePipelineBuilder<Input>.output(
    name: String,
    dataType: TensorDataType<T>,
    shape: TensorShape,
    postprocessing: Stage<Tensor<T>, SO>
): OutputPipelineBuilder<Input, Tuple.One<Tensor<T>>, Tuple.One<SO>> {
    val index = interpreter.getModelMeta().outputsByName[name]?.index
        ?: error("Output tensor with name '$name' not found in the model.")
    return output(index, dataType, shape, postprocessing)
}

/**
 * Adds an output with postprocessing stage to the pipeline using the output tensor name.
 *
 * @param name The name of the output tensor.
 * @param dataType The data type of the output tensor.
 * @param shape The shape of the output tensor.
 * @param postprocessing The postprocessing stage to apply to the output tensor.
 * @return An [OutputPipelineBuilder] with the added output.
 */
@ExperimentalKTensorFlowApi
@JvmName("outputName2")
fun <SO, Input : Tuple, O1, T1, T : Any> OutputPipelineBuilder<Input, Tuple.One<T1>, Tuple.One<O1>>.output(
    name: String,
    dataType: TensorDataType<T>,
    shape: TensorShape,
    postprocessing: Stage<Tensor<T>, SO>
): OutputPipelineBuilder<Input, Tuple.Two<T1, Tensor<T>>, Tuple.Two<O1, SO>> {
    val index = interpreter.getModelMeta().outputsByName[name]?.index
        ?: error("Output tensor with name '$name' not found in the model.")
    return output(index, dataType, shape, postprocessing)
}

/**
 * Adds an output with postprocessing stage to the pipeline using the output tensor name.
 *
 * @param name The name of the output tensor.
 * @param dataType The data type of the output tensor.
 * @param shape The shape of the output tensor.
 * @param postprocessing The postprocessing stage to apply to the output tensor.
 * @return An [OutputPipelineBuilder] with the added output.
 */
@ExperimentalKTensorFlowApi
@JvmName("outputName3")
fun <SO, Input : Tuple, O1, O2, T1, T2, T : Any> OutputPipelineBuilder<Input, Tuple.Two<T1, T2>, Tuple.Two<O1, O2>>.output(
    name: String,
    dataType: TensorDataType<T>,
    shape: TensorShape,
    postprocessing: Stage<Tensor<T>, SO>
): OutputPipelineBuilder<Input, Tuple.Three<T1, T2, Tensor<T>>, Tuple.Three<O1, O2, SO>> {
    val index = interpreter.getModelMeta().outputsByName[name]?.index
        ?: error("Output tensor with name '$name' not found in the model.")
    return output(index, dataType, shape, postprocessing)
}

/**
 * Adds an output with postprocessing stage to the pipeline using the output tensor name.
 *
 * @param name The name of the output tensor.
 * @param dataType The data type of the output tensor.
 * @param shape The shape of the output tensor.
 * @param postprocessing The postprocessing stage to apply to the output tensor.
 * @return An [OutputPipelineBuilder] with the added output.
 */
@ExperimentalKTensorFlowApi
@JvmName("outputName4")
fun <SO, Input : Tuple, O1, O2, O3, T1, T2, T3, T : Any> OutputPipelineBuilder<Input, Tuple.Three<T1, T2, T3>, Tuple.Three<O1, O2, O3>>.output(
    name: String,
    dataType: TensorDataType<T>,
    shape: TensorShape,
    postprocessing: Stage<Tensor<T>, SO>
): OutputPipelineBuilder<Input, Tuple.Four<T1, T2, T3, Tensor<T>>, Tuple.Four<O1, O2, O3, SO>> {
    val index = interpreter.getModelMeta().outputsByName[name]?.index
        ?: error("Output tensor with name '$name' not found in the model.")
    return output(index, dataType, shape, postprocessing)
}

/**
 * Adds an output with postprocessing stage to the pipeline using the output tensor name.
 *
 * @param name The name of the output tensor.
 * @param dataType The data type of the output tensor.
 * @param shape The shape of the output tensor.
 * @param postprocessing The postprocessing stage to apply to the output tensor.
 * @return An [OutputPipelineBuilder] with the added output.
 */
@ExperimentalKTensorFlowApi
@JvmName("outputName5")
fun <SO, Input : Tuple, O1, O2, O3, O4, T1, T2, T3, T4, T : Any> OutputPipelineBuilder<Input, Tuple.Four<T1, T2, T3, T4>, Tuple.Four<O1, O2, O3, O4>>.output(
    name: String,
    dataType: TensorDataType<T>,
    shape: TensorShape,
    postprocessing: Stage<Tensor<T>, SO>
): OutputPipelineBuilder<Input, Tuple.Five<T1, T2, T3, T4, Tensor<T>>, Tuple.Five<O1, O2, O3, O4, SO>> {
    val index = interpreter.getModelMeta().outputsByName[name]?.index
        ?: error("Output tensor with name '$name' not found in the model.")
    return output(index, dataType, shape, postprocessing)
}

/**
 * Adds an output with postprocessing stage to the pipeline using the output tensor name.
 *
 * @param name The name of the output tensor.
 * @param dataType The data type of the output tensor.
 * @param shape The shape of the output tensor.
 * @param postprocessing The postprocessing stage to apply to the output tensor.
 * @return An [OutputPipelineBuilder] with the added output.
 */
@ExperimentalKTensorFlowApi
@JvmName("outputName6")
fun <SO, Input : Tuple, O1, O2, O3, O4, O5, T1, T2, T3, T4, T5, T : Any> OutputPipelineBuilder<Input, Tuple.Five<T1, T2, T3, T4, T5>, Tuple.Five<O1, O2, O3, O4, O5>>.output(
    name: String,
    dataType: TensorDataType<T>,
    shape: TensorShape,
    postprocessing: Stage<Tensor<T>, SO>
): OutputPipelineBuilder<Input, Tuple.Six<T1, T2, T3, T4, T5, Tensor<T>>, Tuple.Six<O1, O2, O3, O4, O5, SO>> {
    val index = interpreter.getModelMeta().outputsByName[name]?.index
        ?: error("Output tensor with name '$name' not found in the model.")
    return output(index, dataType, shape, postprocessing)
}

/**
 * Adds an output with postprocessing stage to the pipeline using the output tensor name.
 *
 * @param name The name of the output tensor.
 * @param dataType The data type of the output tensor.
 * @param shape The shape of the output tensor.
 * @param postprocessing The postprocessing stage to apply to the output tensor.
 * @return An [OutputPipelineBuilder] with the added output.
 */
@ExperimentalKTensorFlowApi
@JvmName("outputName7")
fun <SO, Input : Tuple, O1, O2, O3, O4, O5, O6, T1, T2, T3, T4, T5, T6, T : Any> OutputPipelineBuilder<Input, Tuple.Six<T1, T2, T3, T4, T5, T6>, Tuple.Six<O1, O2, O3, O4, O5, O6>>.output(
    name: String,
    dataType: TensorDataType<T>,
    shape: TensorShape,
    postprocessing: Stage<Tensor<T>, SO>
): OutputPipelineBuilder<Input, Tuple.Seven<T1, T2, T3, T4, T5, T6, Tensor<T>>, Tuple.Seven<O1, O2, O3, O4, O5, O6, SO>> {
    val index = interpreter.getModelMeta().outputsByName[name]?.index
        ?: error("Output tensor with name '$name' not found in the model.")
    return output(index, dataType, shape, postprocessing)
}

/**
 * Adds an output with postprocessing stage to the pipeline using the output tensor name.
 *
 * @param name The name of the output tensor.
 * @param dataType The data type of the output tensor.
 * @param shape The shape of the output tensor.
 * @param postprocessing The postprocessing stage to apply to the output tensor.
 * @return An [OutputPipelineBuilder] with the added output.
 */
@ExperimentalKTensorFlowApi
@JvmName("outputName8")
fun <SO, Input : Tuple, O1, O2, O3, O4, O5, O6, O7, T1, T2, T3, T4, T5, T6, T7, T : Any> OutputPipelineBuilder<Input, Tuple.Seven<T1, T2, T3, T4, T5, T6, T7>, Tuple.Seven<O1, O2, O3, O4, O5, O6, O7>>.output(
    name: String,
    dataType: TensorDataType<T>,
    shape: TensorShape,
    postprocessing: Stage<Tensor<T>, SO>
): OutputPipelineBuilder<Input, Tuple.Eight<T1, T2, T3, T4, T5, T6, T7, Tensor<T>>, Tuple.Eight<O1, O2, O3, O4, O5, O6, O7, SO>> {
    val index = interpreter.getModelMeta().outputsByName[name]?.index
        ?: error("Output tensor with name '$name' not found in the model.")
    return output(index, dataType, shape, postprocessing)
}

/**
 * Adds an output with postprocessing stage to the pipeline using the output tensor name.
 *
 * @param name The name of the output tensor.
 * @param dataType The data type of the output tensor.
 * @param shape The shape of the output tensor.
 * @param postprocessing The postprocessing stage to apply to the output tensor.
 * @return An [OutputPipelineBuilder] with the added output.
 */
@ExperimentalKTensorFlowApi
@JvmName("outputName9")
fun <SO, Input : Tuple, O1, O2, O3, O4, O5, O6, O7, O8, T1, T2, T3, T4, T5, T6, T7, T8, T : Any> OutputPipelineBuilder<Input, Tuple.Eight<T1, T2, T3, T4, T5, T6, T7, T8>, Tuple.Eight<O1, O2, O3, O4, O5, O6, O7, O8>>.output(
    name: String,
    dataType: TensorDataType<T>,
    shape: TensorShape,
    postprocessing: Stage<Tensor<T>, SO>
): OutputPipelineBuilder<Input, Tuple.Nine<T1, T2, T3, T4, T5, T6, T7, T8, Tensor<T>>, Tuple.Nine<O1, O2, O3, O4, O5, O6, O7, O8, SO>> {
    val index = interpreter.getModelMeta().outputsByName[name]?.index
        ?: error("Output tensor with name '$name' not found in the model.")
    return output(index, dataType, shape, postprocessing)
}

/**
 * Adds an output with postprocessing stage to the pipeline using the output tensor name.
 *
 * @param name The name of the output tensor.
 * @param dataType The data type of the output tensor.
 * @param shape The shape of the output tensor.
 * @param postprocessing The postprocessing stage to apply to the output tensor.
 * @return An [OutputPipelineBuilder] with the added output.
 */
@ExperimentalKTensorFlowApi
@JvmName("outputName10")
fun <SO, Input : Tuple, O1, O2, O3, O4, O5, O6, O7, O8, O9, T1, T2, T3, T4, T5, T6, T7, T8, T9, T : Any> OutputPipelineBuilder<Input, Tuple.Nine<T1, T2, T3, T4, T5, T6, T7, T8, T9>, Tuple.Nine<O1, O2, O3, O4, O5, O6, O7, O8, O9>>.output(
    name: String,
    dataType: TensorDataType<T>,
    shape: TensorShape,
    postprocessing: Stage<Tensor<T>, SO>
): OutputPipelineBuilder<Input, Tuple.Ten<T1, T2, T3, T4, T5, T6, T7, T8, T9, Tensor<T>>, Tuple.Ten<O1, O2, O3, O4, O5, O6, O7, O8, O9, SO>> {
    val index = interpreter.getModelMeta().outputsByName[name]?.index
        ?: error("Output tensor with name '$name' not found in the model.")
    return output(index, dataType, shape, postprocessing)
}