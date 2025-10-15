package dev.kursor.ktensorflow

/**
 * Interpreter to run inference on a model.
 */
interface Interpreter {

    /**
     * Runs model inference for multiple inputs and outputs.
     * Result of the inference will be written to the output [ByteArray]s, which should be
     * allocated beforehand and passed to this method.
     * @param inputs List of input [ByteArray]s.
     * @param outputs Map of output [ByteArray]s, key is output index..
     */
    fun run(
        inputs: List<ByteArray>,
        outputs: Map<Int, ByteArray>
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
 * Result of the inference will be written to the output [ByteArray], which should be
 * allocated beforehand and passed to this method.
 * @param input Input [ByteArray].
 * @param output Output [ByteArray].
 */
fun Interpreter.run(
    input: ByteArray,
    output: ByteArray
) = run(listOf(input), mapOf(0 to output))