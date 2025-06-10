package dev.kursor.ktensorflow

/**
 * Interpreter to run inference on a model.
 */
interface Interpreter {

    /**
     * Runs model inference.
     * Result of the inference will be written to the output [Tensor]s, which should be
     * allocated beforehand and passed to this method.
     * @param inputs List of input [Tensor]s.
     * @param outputs List of output [Tensor]s.
     */
    fun run(
        inputs: List<Tensor>,
        outputs: List<Tensor>
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
 * Runs model inference with a single input and output [Tensor].
 * This is a convenience method that wraps the [run] method.
 * @param input Input [Tensor].
 * @param output Output [Tensor].
 */
fun Interpreter.run(
    input: Tensor,
    output: Tensor
) = run(listOf(input), listOf(output))
