package dev.kursor.ktensorflow.api

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

expect fun Interpreter(
    modelDesc: ModelDesc,
    options: InterpreterOptions,
): Interpreter
