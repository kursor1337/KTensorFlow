import dev.kursor.ktensorflow.InterpreterOptions
import dev.kursor.ktensorflow.coroutines.SuspendInterpreter
import dev.kursor.ktensorflow.coroutines.runSuspend
import dev.kursor.ktensorflow.tensor.Tensor
import dev.kursor.ktensorflow.tensor.TensorShape
import kotlinx.coroutines.runBlocking
import kotlin.test.Test
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
}
