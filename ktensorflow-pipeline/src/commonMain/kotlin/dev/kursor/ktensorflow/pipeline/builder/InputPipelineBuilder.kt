@file:Suppress("UNCHECKED_CAST")

package dev.kursor.ktensorflow.pipeline.builder

import dev.kursor.ktensorflow.ExperimentalKTensorFlowApi
import dev.kursor.ktensorflow.tensor.Tensor
import dev.kursor.ktensorflow.pipeline.Pipeline
import dev.kursor.ktensorflow.pipeline.stage.CombinedStage
import dev.kursor.ktensorflow.pipeline.stage.Stage
import dev.kursor.ktensorflow.pipeline.Tuple
import dev.kursor.ktensorflow.pipeline.Tuple.Eight
import dev.kursor.ktensorflow.pipeline.Tuple.Five
import dev.kursor.ktensorflow.pipeline.Tuple.Four
import dev.kursor.ktensorflow.pipeline.Tuple.Nine
import dev.kursor.ktensorflow.pipeline.Tuple.One
import dev.kursor.ktensorflow.pipeline.Tuple.Seven
import dev.kursor.ktensorflow.pipeline.Tuple.Six
import dev.kursor.ktensorflow.pipeline.Tuple.Ten
import dev.kursor.ktensorflow.pipeline.Tuple.Three
import dev.kursor.ktensorflow.pipeline.Tuple.Two
import kotlin.jvm.JvmName

/**
 * A builder that represents a pipeline with preprocessing stages.
 */
@ExperimentalKTensorFlowApi
class InputPipelineBuilder<Input : Tuple, Output : Tuple> internal constructor(
    internal val inputStages: List<Stage<Any?, Tensor<*>>> = listOf()
) {
    internal fun build(): Stage<Input, Output> {
        return CombinedStage(inputStages)
    }
}

/**
 * Creates an [InputPipelineBuilder] with a single input.
 *
 * @param preprocessing The preprocessing stage to apply to the input.
 * @return An [InputPipelineBuilder] with the added input.
 */
@ExperimentalKTensorFlowApi
fun <SI> Pipeline.Companion.input(preprocessing: Stage<SI, Tensor<*>>): InputPipelineBuilder<One<SI>, One<Tensor<*>>> {
    return InputPipelineBuilder<One<SI>, One<Tensor<*>>>(listOf(preprocessing) as List<Stage<Any?, Tensor<*>>>)
}

/**
 * Adds an input with a preprocessing stage to the pipeline.
 *
 * @param preprocessing The preprocessing stage to apply to the input.
 * @return An [InputPipelineBuilder] with the added input.
 */
@ExperimentalKTensorFlowApi
@JvmName("input2")
fun <SI, T1> InputPipelineBuilder<One<T1>, One<Tensor<*>>>.input(
    preprocessing: Stage<SI, Tensor<*>>
): InputPipelineBuilder<Two<T1, SI>, Two<T1, SI>> {
    return InputPipelineBuilder(
        (inputStages + preprocessing) as List<Stage<Any?, Tensor<*>>>
    )
}

/**
 * Adds an input with a preprocessing stage to the pipeline.
 *
 * @param preprocessing The preprocessing stage to apply to the input.
 * @return An [InputPipelineBuilder] with the added input.
 */
@ExperimentalKTensorFlowApi
@JvmName("input3")
fun <SI, I1, I2> InputPipelineBuilder<Two<I1, I2>, Two<Tensor<*>, Tensor<*>>>.input(
    preprocessing: Stage<SI, Tensor<*>>
): InputPipelineBuilder<Three<I1, I2, SI>, Three<Tensor<*>, Tensor<*>, Tensor<*>>> {
    return InputPipelineBuilder(
        (inputStages + preprocessing) as List<Stage<Any?, Tensor<*>>>
    )
}

/**
 * Adds an input with a preprocessing stage to the pipeline.
 *
 * @param preprocessing The preprocessing stage to apply to the input.
 * @return An [InputPipelineBuilder] with the added input.
 */
@ExperimentalKTensorFlowApi
@JvmName("input4")
fun <SI, I1, I2, I3> InputPipelineBuilder<Three<I1, I2, I3>, Three<Tensor<*>, Tensor<*>, Tensor<*>>>.input(
    preprocessing: Stage<SI, Tensor<*>>
): InputPipelineBuilder<Four<I1, I2, I3, SI>, Four<Tensor<*>, Tensor<*>, Tensor<*>, Tensor<*>>> {
    return InputPipelineBuilder(
        (inputStages + preprocessing) as List<Stage<Any?, Tensor<*>>>
    )
}

/**
 * Adds an input with a preprocessing stage to the pipeline.
 *
 * @param preprocessing The preprocessing stage to apply to the input.
 * @return An [InputPipelineBuilder] with the added input.
 */
@ExperimentalKTensorFlowApi
@JvmName("input5")
fun <SI, I1, I2, I3, I4> InputPipelineBuilder<Four<I1, I2, I3, I4>, Four<Tensor<*>, Tensor<*>, Tensor<*>, Tensor<*>>>.input(
    preprocessing: Stage<SI, Tensor<*>>
): InputPipelineBuilder<Five<I1, I2, I3, I4, SI>, Five<Tensor<*>, Tensor<*>, Tensor<*>, Tensor<*>, Tensor<*>>> {
    return InputPipelineBuilder(
        (inputStages + preprocessing) as List<Stage<Any?, Tensor<*>>>
    )
}

/**
 * Adds an input with a preprocessing stage to the pipeline.
 *
 * @param preprocessing The preprocessing stage to apply to the input.
 * @return An [InputPipelineBuilder] with the added input.
 */
@ExperimentalKTensorFlowApi
@JvmName("input6")
fun <SI, I1, I2, I3, I4, I5> InputPipelineBuilder<Five<I1, I2, I3, I4, I5>, Five<Tensor<*>, Tensor<*>, Tensor<*>, Tensor<*>, Tensor<*>>>.input(
    preprocessing: Stage<SI, Tensor<*>>
): InputPipelineBuilder<Six<I1, I2, I3, I4, I5, SI>, Six<Tensor<*>, Tensor<*>, Tensor<*>, Tensor<*>, Tensor<*>, Tensor<*>>> {
    return InputPipelineBuilder(
        (inputStages + preprocessing) as List<Stage<Any?, Tensor<*>>>
    )
}

/**
 * Adds an input with a preprocessing stage to the pipeline.
 *
 * @param preprocessing The preprocessing stage to apply to the input.
 * @return An [InputPipelineBuilder] with the added input.
 */
@ExperimentalKTensorFlowApi
@JvmName("input7")
fun <SI, I1, I2, I3, I4, I5, I6> InputPipelineBuilder<Six<I1, I2, I3, I4, I5, I6>, Six<Tensor<*>, Tensor<*>, Tensor<*>, Tensor<*>, Tensor<*>, Tensor<*>>>.input(
    preprocessing: Stage<SI, Tensor<*>>
): InputPipelineBuilder<Seven<I1, I2, I3, I4, I5, I6, SI>, Seven<Tensor<*>, Tensor<*>, Tensor<*>, Tensor<*>, Tensor<*>, Tensor<*>, Tensor<*>>> {
    return InputPipelineBuilder(
        (inputStages + preprocessing) as List<Stage<Any?, Tensor<*>>>
    )
}

/**
 * Adds an input with a preprocessing stage to the pipeline.
 *
 * @param preprocessing The preprocessing stage to apply to the input.
 * @return An [InputPipelineBuilder] with the added input.
 */
@ExperimentalKTensorFlowApi
@JvmName("input8")
fun <SI, I1, I2, I3, I4, I5, I6, I7> InputPipelineBuilder<Seven<I1, I2, I3, I4, I5, I6, I7>, Seven<Tensor<*>, Tensor<*>, Tensor<*>, Tensor<*>, Tensor<*>, Tensor<*>, Tensor<*>>>.input(
    preprocessing: Stage<SI, Tensor<*>>
): InputPipelineBuilder<Eight<I1, I2, I3, I4, I5, I6, I7, SI>, Eight<Tensor<*>, Tensor<*>, Tensor<*>, Tensor<*>, Tensor<*>, Tensor<*>, Tensor<*>, Tensor<*>>> {
    return InputPipelineBuilder(
        (inputStages + preprocessing) as List<Stage<Any?, Tensor<*>>>
    )
}

/**
 * Adds an input with a preprocessing stage to the pipeline.
 *
 * @param preprocessing The preprocessing stage to apply to the input.
 * @return An [InputPipelineBuilder] with the added input.
 */
@ExperimentalKTensorFlowApi
@JvmName("input9")
fun <SI, I1, I2, I3, I4, I5, I6, I7, I8> InputPipelineBuilder<Eight<I1, I2, I3, I4, I5, I6, I7, I8>, Eight<Tensor<*>, Tensor<*>, Tensor<*>, Tensor<*>, Tensor<*>, Tensor<*>, Tensor<*>, Tensor<*>>>.input(
    preprocessing: Stage<SI, Tensor<*>>
): InputPipelineBuilder<Nine<I1, I2, I3, I4, I5, I6, I7, I8, SI>, Nine<Tensor<*>, Tensor<*>, Tensor<*>, Tensor<*>, Tensor<*>, Tensor<*>, Tensor<*>, Tensor<*>, Tensor<*>>> {
    return InputPipelineBuilder(
        (inputStages + preprocessing) as List<Stage<Any?, Tensor<*>>>
    )
}
/**
 * Adds an input with a preprocessing stage to the pipeline.
 *
 * @param preprocessing The preprocessing stage to apply to the input.
 * @return An [InputPipelineBuilder] with the added input.
 */
@ExperimentalKTensorFlowApi
@JvmName("input10")
fun <SI, I1, I2, I3, I4, I5, I6, I7, I8, I9> InputPipelineBuilder<Nine<I1, I2, I3, I4, I5, I6, I7, I8, I9>, Nine<Tensor<*>, Tensor<*>, Tensor<*>, Tensor<*>, Tensor<*>, Tensor<*>, Tensor<*>, Tensor<*>, Tensor<*>>>.input(
    preprocessing: Stage<SI, Tensor<*>>
): InputPipelineBuilder<Ten<I1, I2, I3, I4, I5, I6, I7, I8, I9, SI>, Ten<Tensor<*>, Tensor<*>, Tensor<*>, Tensor<*>, Tensor<*>, Tensor<*>, Tensor<*>, Tensor<*>, Tensor<*>, Tensor<*>>> {
    return InputPipelineBuilder(
        (inputStages + preprocessing) as List<Stage<Any?, Tensor<*>>>
    )
}