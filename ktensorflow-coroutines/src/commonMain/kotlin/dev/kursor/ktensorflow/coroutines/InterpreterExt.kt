package dev.kursor.ktensorflow.coroutines

import dev.kursor.ktensorflow.Interpreter
import dev.kursor.ktensorflow.run
import dev.kursor.ktensorflow.tensor.PhysicalTensor
import dev.kursor.ktensorflow.tensor.run
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Suspends the current coroutine and runs model inference for multiple inputs and outputs
 * on a background thread ([Dispatchers.Default]).
 *
 * This is a safe, non-blocking alternative to [Interpreter.run] that ensures heavy
 * CPU computations do not block the calling thread (e.g., the Main/UI thread).
 * Results of the inference will be written to the provided output [ByteArray]s, which
 * should be allocated beforehand.
 *
 * @param inputs List of input [ByteArray]s.
 * @param outputs Map of output [ByteArray]s, where the key is the output tensor index.
 */
suspend fun Interpreter.runSuspend(
    inputs: List<ByteArray>,
    outputs: Map<Int, ByteArray>
) = withContext(Dispatchers.Default) {
    run(inputs, outputs)
}

/**
 * Suspends the current coroutine and runs model inference for multiple inputs and outputs
 * on a background thread ([Dispatchers.Default]).
 *
 * This is a safe, non-blocking alternative to [Interpreter.run] that ensures heavy
 * CPU computations do not block the calling thread. Results of the inference will be
 * written to the provided output [ByteArray]s, which should be allocated beforehand.
 *
 * @param inputs List of input [ByteArray]s.
 * @param outputs Map of output [ByteArray]s, where the key is the output tensor signature name.
 */
suspend fun Interpreter.runSuspend(
    inputs: List<ByteArray>,
    outputs: Map<String, ByteArray>
) = withContext(Dispatchers.Default) {
    run(inputs, outputs)
}

/**
 * Suspends the current coroutine and runs model inference for a single input and output
 * on a background thread ([Dispatchers.Default]).
 *
 * This is a safe, non-blocking alternative to [Interpreter.run] that ensures heavy
 * CPU computations do not block the calling thread. The result of the inference will be
 * written to the output [ByteArray], which should be allocated beforehand.
 *
 * @param input Input [ByteArray].
 * @param output Output [ByteArray].
 */
suspend fun Interpreter.runSuspend(
    input: ByteArray,
    output: ByteArray
) = withContext(Dispatchers.Default) {
    run(input, output)
}

/**
 * Suspends the current coroutine and runs model inference for multiple inputs and outputs
 * using [PhysicalTensor]s on a background thread ([Dispatchers.Default]).
 *
 * This is a safe, non-blocking alternative to [Interpreter.run] that ensures heavy
 * CPU computations do not block the calling thread. Results of the inference will be
 * written directly into the provided output [PhysicalTensor]s.
 *
 * @param inputs List of input [PhysicalTensor]s.
 * @param outputs Map of output [PhysicalTensor]s, where the key is the output tensor index.
 */
suspend fun Interpreter.runSuspend(
    inputs: List<PhysicalTensor<*>>,
    outputs: Map<Int, PhysicalTensor<*>>
) = withContext(Dispatchers.Default) {
    run(inputs, outputs)
}

/**
 * Suspends the current coroutine and runs model inference for multiple inputs and outputs
 * using [PhysicalTensor]s on a background thread ([Dispatchers.Default]).
 *
 * This is a safe, non-blocking alternative to [Interpreter.run] that ensures heavy
 * CPU computations do not block the calling thread. Results of the inference will be
 * written directly into the provided output [PhysicalTensor]s.
 *
 * @param inputs List of input [PhysicalTensor]s.
 * @param outputs Map of output [PhysicalTensor]s, where the key is the output tensor signature name.
 */
suspend fun Interpreter.runSuspend(
    inputs: List<PhysicalTensor<*>>,
    outputs: Map<String, PhysicalTensor<*>>
) = withContext(Dispatchers.Default) {
    run(inputs, outputs)
}

/**
 * Suspends the current coroutine and runs model inference for a single input and output
 * using [PhysicalTensor]s on a background thread ([Dispatchers.Default]).
 *
 * This is a safe, non-blocking alternative to [Interpreter.run] that ensures heavy
 * CPU computations do not block the calling thread. The result of the inference will be
 * written directly into the output [PhysicalTensor].
 *
 * @param input Input [PhysicalTensor].
 * @param output Output [PhysicalTensor].
 */
suspend fun Interpreter.runSuspend(
    input: PhysicalTensor<*>,
    output: PhysicalTensor<*>
) = withContext(Dispatchers.Default) {
    run(input, output)
}