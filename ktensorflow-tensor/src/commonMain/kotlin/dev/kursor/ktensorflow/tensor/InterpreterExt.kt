package dev.kursor.ktensorflow.tensor

import dev.kursor.ktensorflow.Interpreter

/**
 * Runs model inference for multiple inputs and outputs.
 * Result of the inference will be written to the output [Tensor]s, which should be
 * allocated beforehand and passed to this method.
 * @param inputs List of input [Tensor]s.
 * @param outputs Map of output [Tensor]s, key is index of the output [Tensor]
 */
fun Interpreter.run(
    inputs: List<Tensor<*>>,
    outputs: Map<Int, Tensor<*>>
) = run(
    inputs = inputs.map { it.data },
    outputs = outputs.mapValues { it.value.data }
)

/**
 * Runs model inference for multiple inputs and outputs.
 * Result of the inference will be written to the output [Tensor]s, which should be
 * allocated beforehand and passed to this method.
 * @param inputs List of input [Tensor]s.
 * @param outputs Map of output [Tensor]s, key is index of the output [Tensor]
 */
fun Interpreter.run(
    input: Tensor<*>,
    output: Tensor<*>
) = run(listOf(input), mapOf(0 to output))