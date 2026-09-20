import dev.kursor.ktensorflow.Interpreter
import dev.kursor.ktensorflow.InterpreterOptions
import dev.kursor.ktensorflow.ModelDesc
import dev.kursor.ktensorflow.TensorFlowException
import dev.kursor.ktensorflow.tensor.Tensor
import dev.kursor.ktensorflow.tensor.TensorShape
import dev.kursor.ktensorflow.tensor.run
import kotlin.test.Test
import kotlin.test.assertFailsWith

/**
 * Отказ рантайма обязан приходить наружу как [TensorFlowException] на обеих платформах -
 * см. одноимённый тест в androidDeviceTest.
 */
class TensorFlowExceptionTest {

    @Test
    fun loadingAMissingModelThrowsTensorFlowException() {
        assertFailsWith<TensorFlowException> {
            Interpreter(
                ModelDesc.PathInBundle("there-is-no-such-model.tflite"),
                InterpreterOptions()
            )
        }
    }

    @Test
    fun runningInferenceWithAMismatchedShapeThrowsTensorFlowException() {
        val interpreter = createInterpreter("mnist", "tflite", null)

        val wrongInput = Tensor<Float>(shape = TensorShape(3, 3)).toPhysical()
        val output = Tensor<Float>(shape = TensorShape(10)).toPhysical()

        assertFailsWith<TensorFlowException> {
            interpreter.run(wrongInput, output)
        }

        interpreter.close()
    }
}
