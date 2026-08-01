package dev.kursor.ktensorflow.tensor

import dev.kursor.ktensorflow.Interpreter
import kotlin.jvm.JvmName

/**
 * Runs model inference for multiple inputs and outputs.
 * Result of the inference will be written to the output [Tensor]s, which should be
 * allocated beforehand and passed to this method.
 * WARNING: This function is not thread-safe. You should not call it from multiple threads.
 * @param inputs List of input [Tensor]s.
 * @param outputs Map of output [Tensor]s, key is index of the output [Tensor]
 */
fun Interpreter.run(
    inputs: List<PhysicalTensor<*>>,
    outputs: Map<Int, PhysicalTensor<*>>
) = run(
    inputs = inputs.map { it.data },
    outputs = outputs.mapValues { it.value.data }
)

/**
 * Runs model inference for multiple inputs and outputs.
 * Result of the inference will be written to the output [Tensor]s, which should be
 * allocated beforehand and passed to this method.
 * WARNING: This function is not thread-safe. You should not call it from multiple threads.
 * @param inputs List of input [Tensor]s.
 * @param outputs Map of output [Tensor]s, key is index of the output [Tensor]
 */
@JvmName("runWithNames")
fun Interpreter.run(
    inputs: List<PhysicalTensor<*>>,
    outputs: Map<String, PhysicalTensor<*>>
) {
    val modelMeta = getModelMeta()
    val outputMap = outputs.mapKeys { modelMeta.outputsByName[it.key]?.index ?: 0 }
    run(
        inputs = inputs,
        outputs = outputMap
    )
}

/**
 * Runs model inference for single input and output.
 * Result of the inference will be written to the output [Tensor], which should be
 * allocated beforehand and passed to this method.
 * WARNING: This function is not thread-safe. You should not call it from multiple threads.
 * @param input input [Tensor].
 * @param output output [Tensor]
 */
fun Interpreter.run(
    input: PhysicalTensor<*>,
    output: PhysicalTensor<*>
) = run(listOf(input), mapOf(0 to output))