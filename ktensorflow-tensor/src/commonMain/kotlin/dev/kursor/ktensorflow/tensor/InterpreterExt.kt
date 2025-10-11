package dev.kursor.ktensorflow.tensor

import dev.kursor.ktensorflow.Interpreter

fun Interpreter.run(
    inputs: List<Tensor>,
    outputs: Map<Int, Tensor>
) = run(
    inputs = inputs.map { it.data },
    outputs = outputs.mapValues { it.value.data }
)

fun Interpreter.run(
    input: Tensor,
    output: Tensor
) = run(listOf(input), mapOf(0 to output))