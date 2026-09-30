package dev.kursor.ktensorflow.coroutines

import dev.kursor.ktensorflow.Interpreter
import dev.kursor.ktensorflow.InterpreterOptions
import dev.kursor.ktensorflow.ModelDesc
import dev.kursor.ktensorflow.TensorFlowException
import dev.kursor.ktensorflow.run
import dev.kursor.ktensorflow.tensor.PhysicalTensor
import dev.kursor.ktensorflow.tensor.run
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.withContext
import kotlin.jvm.JvmName

/**
 * Creates an [Interpreter] ready to be run through coroutines: it is created on the thread this
 * module runs inference on.
 *
 * The interpreter applies its delegates while it is created, and the GPU delegate on its OpenGL
 * backend, which Android uses on devices without OpenCL, must then run inference on that same
 * thread. Creating the interpreter here and running it through [runSuspend], `processFlow` or a
 * pipeline's `runSuspend` keeps both on one thread. An interpreter without such a delegate, or
 * one used on iOS, can be created anywhere.
 *
 * The returned object is a plain [Interpreter]; the name describes how it was created, like
 * `SupervisorJob()` in kotlinx.coroutines returns a plain `Job`.
 *
 * @param modelDesc The model to load.
 * @param options Options of the interpreter, including its delegates.
 * @param dispatcher Dispatcher the interpreter is created on. By default it is the shared
 *   [InferenceDispatcher] all functions of this module use by default. A dispatcher of your own,
 *   such as an [InferenceDispatcher] per interpreter, lets several interpreters run in parallel;
 *   with the GPU delegate on Android it must be single-threaded, as [InferenceDispatcher] is, and
 *   the same one must be passed to every run of this interpreter.
 * @throws TensorFlowException if the model cannot be loaded or a delegate cannot be applied.
 */
// Имя с большой буквы при другом возвращаемом типе - сознательно, по образцу SupervisorJob():
// оно называет вид интерпретатора, а не отдельный тип
@Suppress("FunctionNaming")
suspend fun SuspendInterpreter(
    modelDesc: ModelDesc,
    options: InterpreterOptions,
    dispatcher: CoroutineDispatcher = SharedInferenceDispatcher
): Interpreter = withContext(dispatcher) {
    Interpreter(modelDesc, options)
}

/**
 * Suspends the current coroutine and runs model inference for multiple inputs and outputs
 * on a background thread.
 *
 * By default inference runs on the shared inference dispatcher, where calls from the whole
 * process queue as suspended coroutines instead of blocking threads; pass [dispatcher] to run
 * interpreters in parallel.
 *
 * This is a safe, non-blocking alternative to [Interpreter.run] that ensures heavy
 * CPU computations do not block the calling thread (e.g., the Main/UI thread).
 * Results of the inference will be written to the provided output [ByteArray]s, which
 * should be allocated beforehand.
 *
 * @param inputs List of input [ByteArray]s.
 * @param outputs Map of output [ByteArray]s, where the key is the output tensor index.
 * @param dispatcher Dispatcher inference runs on. By default it is one shared
 *   [InferenceDispatcher], which runs one inference at a time for the whole process. Pass an
 *   [InferenceDispatcher] per interpreter to run several models in parallel.
 */
suspend fun Interpreter.runSuspend(
    inputs: List<ByteArray>,
    outputs: Map<Int, ByteArray>,
    dispatcher: CoroutineDispatcher = SharedInferenceDispatcher
) = withContext(dispatcher) {
    run(inputs, outputs)
}

/**
 * Suspends the current coroutine and runs model inference for multiple inputs and outputs
 * on a background thread.
 *
 * By default inference runs on the shared inference dispatcher, where calls from the whole
 * process queue as suspended coroutines instead of blocking threads; pass [dispatcher] to run
 * interpreters in parallel.
 *
 * This is a safe, non-blocking alternative to [Interpreter.run] that ensures heavy
 * CPU computations do not block the calling thread. Results of the inference will be
 * written to the provided output [ByteArray]s, which should be allocated beforehand.
 *
 * @param inputs List of input [ByteArray]s.
 * @param outputs Map of output [ByteArray]s, where the key is the output tensor signature name.
 * @param dispatcher Dispatcher inference runs on. By default it is one shared
 *   [InferenceDispatcher], which runs one inference at a time for the whole process. Pass an
 *   [InferenceDispatcher] per interpreter to run several models in parallel.
 */
@JvmName("runSuspendWithNames")
suspend fun Interpreter.runSuspend(
    inputs: List<ByteArray>,
    outputs: Map<String, ByteArray>,
    dispatcher: CoroutineDispatcher = SharedInferenceDispatcher
) = withContext(dispatcher) {
    run(inputs, outputs)
}

/**
 * Suspends the current coroutine and runs model inference for a single input and output
 * on a background thread.
 *
 * By default inference runs on the shared inference dispatcher, where calls from the whole
 * process queue as suspended coroutines instead of blocking threads; pass [dispatcher] to run
 * interpreters in parallel.
 *
 * This is a safe, non-blocking alternative to [Interpreter.run] that ensures heavy
 * CPU computations do not block the calling thread. The result of the inference will be
 * written to the output [ByteArray], which should be allocated beforehand.
 *
 * @param input Input [ByteArray].
 * @param output Output [ByteArray].
 * @param dispatcher Dispatcher inference runs on. By default it is one shared
 *   [InferenceDispatcher], which runs one inference at a time for the whole process. Pass an
 *   [InferenceDispatcher] per interpreter to run several models in parallel.
 */
suspend fun Interpreter.runSuspend(
    input: ByteArray,
    output: ByteArray,
    dispatcher: CoroutineDispatcher = SharedInferenceDispatcher
) = withContext(dispatcher) {
    run(input, output)
}

/**
 * Suspends the current coroutine and runs model inference for multiple inputs and outputs
 * using [PhysicalTensor]s on a background thread.
 *
 * By default inference runs on the shared inference dispatcher, where calls from the whole
 * process queue as suspended coroutines instead of blocking threads; pass [dispatcher] to run
 * interpreters in parallel.
 *
 * This is a safe, non-blocking alternative to [Interpreter.run] that ensures heavy
 * CPU computations do not block the calling thread. Results of the inference will be
 * written directly into the provided output [PhysicalTensor]s.
 *
 * @param inputs List of input [PhysicalTensor]s.
 * @param outputs Map of output [PhysicalTensor]s, where the key is the output tensor index.
 * @param dispatcher Dispatcher inference runs on. By default it is one shared
 *   [InferenceDispatcher], which runs one inference at a time for the whole process. Pass an
 *   [InferenceDispatcher] per interpreter to run several models in parallel.
 */
@JvmName("runSuspendTensors")
suspend fun Interpreter.runSuspend(
    inputs: List<PhysicalTensor<*>>,
    outputs: Map<Int, PhysicalTensor<*>>,
    dispatcher: CoroutineDispatcher = SharedInferenceDispatcher
) = withContext(dispatcher) {
    run(inputs, outputs)
}

/**
 * Suspends the current coroutine and runs model inference for multiple inputs and outputs
 * using [PhysicalTensor]s on a background thread.
 *
 * By default inference runs on the shared inference dispatcher, where calls from the whole
 * process queue as suspended coroutines instead of blocking threads; pass [dispatcher] to run
 * interpreters in parallel.
 *
 * This is a safe, non-blocking alternative to [Interpreter.run] that ensures heavy
 * CPU computations do not block the calling thread. Results of the inference will be
 * written directly into the provided output [PhysicalTensor]s.
 *
 * @param inputs List of input [PhysicalTensor]s.
 * @param outputs Map of output [PhysicalTensor]s, where the key is the output tensor signature name.
 * @param dispatcher Dispatcher inference runs on. By default it is one shared
 *   [InferenceDispatcher], which runs one inference at a time for the whole process. Pass an
 *   [InferenceDispatcher] per interpreter to run several models in parallel.
 */
@JvmName("runSuspendTensorsWithNames")
suspend fun Interpreter.runSuspend(
    inputs: List<PhysicalTensor<*>>,
    outputs: Map<String, PhysicalTensor<*>>,
    dispatcher: CoroutineDispatcher = SharedInferenceDispatcher
) = withContext(dispatcher) {
    run(inputs, outputs)
}

/**
 * Suspends the current coroutine and runs model inference for a single input and output
 * using [PhysicalTensor]s on a background thread.
 *
 * By default inference runs on the shared inference dispatcher, where calls from the whole
 * process queue as suspended coroutines instead of blocking threads; pass [dispatcher] to run
 * interpreters in parallel.
 *
 * This is a safe, non-blocking alternative to [Interpreter.run] that ensures heavy
 * CPU computations do not block the calling thread. The result of the inference will be
 * written directly into the output [PhysicalTensor].
 *
 * @param input Input [PhysicalTensor].
 * @param output Output [PhysicalTensor].
 * @param dispatcher Dispatcher inference runs on. By default it is one shared
 *   [InferenceDispatcher], which runs one inference at a time for the whole process. Pass an
 *   [InferenceDispatcher] per interpreter to run several models in parallel.
 */
suspend fun Interpreter.runSuspend(
    input: PhysicalTensor<*>,
    output: PhysicalTensor<*>,
    dispatcher: CoroutineDispatcher = SharedInferenceDispatcher
) = withContext(dispatcher) {
    run(input, output)
}