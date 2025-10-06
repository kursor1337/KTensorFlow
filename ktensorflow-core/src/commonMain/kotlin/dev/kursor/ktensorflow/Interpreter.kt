package dev.kursor.ktensorflow

/**
 * Interpreter to run inference on a model.
 */
interface Interpreter {

    /**
     * Runs model inference for multiple inputs and outputs.
     * Result of the inference will be written to the output [Tensor]s, which should be
     * allocated beforehand and passed to this method.
     * @param inputs List of input [Tensor]s.
     * @param outputs Map of output [Tensor]s, key is Tensor index..
     */
    fun run(
        inputs: List<Tensor>,
        outputs: Map<Int, Tensor>
    )

    /**
     * Release resources associated with the [Interpreter].
     */
    fun close()
}

/**
 * Creates a new [Interpreter] for the given [ModelDesc] and [InterpreterOptions].
 */
expect fun Interpreter(
    modelDesc: ModelDesc,
    options: InterpreterOptions,
): Interpreter

/**
 * Runs model inference for single input and output.
 * Result of the inference will be written to the output [Tensor], which should be
 * allocated beforehand and passed to this method.
 * @param input Input [Tensor].
 * @param output Output [Tensor].
 */
fun Interpreter.run(
    input: Tensor,
    output: Tensor
): Tensor {
    run(listOf(input), mapOf(0 to output))
    return output
}