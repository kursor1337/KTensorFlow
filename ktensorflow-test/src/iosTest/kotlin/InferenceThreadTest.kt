import dev.kursor.ktensorflow.InterpreterOptions
import dev.kursor.ktensorflow.coroutines.InferenceDispatcher
import dev.kursor.ktensorflow.coroutines.SuspendInterpreter
import dev.kursor.ktensorflow.coroutines.runSuspend
import dev.kursor.ktensorflow.tensor.Tensor
import dev.kursor.ktensorflow.tensor.TensorShape
import kotlinx.coroutines.runBlocking
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * На iOS делегаты к потоку не привязаны, и инференс через модуль корутин лишь выполняется по
 * одному (это проверяет CoroutinesModuleTests). Здесь - что SuspendInterpreter даёт рабочий
 * интерпретатор и на этой платформе.
 */
class InferenceThreadTest {

    @Test
    fun suspendInterpreterGivesAWorkingInterpreter() = runBlocking {
        val interpreter = SuspendInterpreter(loadModel("mnist", "tflite"), InterpreterOptions())
        val output = Tensor<Float>(shape = TensorShape(10)).toPhysical()

        interpreter.runSuspend(Tensor<Float>(shape = TensorShape(28, 28)).toPhysical(), output)

        assertTrue((0 until 10).all { output.getFlat(it).isFinite() }, "the model must have produced its outputs")
        interpreter.close()
    }

    @Test
    fun aClosedInferenceDispatcherIsRejectedLikeOnAndroid() = runBlocking {
        // Раньше close() на iOS ничего не делал, и закрытый диспетчер продолжал работать
        val dispatcher = InferenceDispatcher().apply { close() }
        val interpreter = SuspendInterpreter(loadModel("mnist", "tflite"), InterpreterOptions())
        val input = Tensor<Float>(shape = TensorShape(28, 28)).toPhysical()
        val output = Tensor<Float>(shape = TensorShape(10)).toPhysical()

        assertClosedDispatcherError {
            SuspendInterpreter(loadModel("mnist", "tflite"), InterpreterOptions(), dispatcher)
        }
        assertClosedDispatcherError { interpreter.runSuspend(input, output, dispatcher) }
        interpreter.close()
    }

    // На JVM CancellationException - подкласс IllegalStateException, поэтому одного типа мало:
    // старая тихая отмена ("The task was rejected") тоже прошла бы такую проверку
    private suspend fun assertClosedDispatcherError(block: suspend () -> Unit) {
        val error = runCatching { block() }.exceptionOrNull()
        assertEquals("InferenceDispatcher has already been closed", error?.message, "$error")
    }
}
