import dev.kursor.ktensorflow.Interpreter
import dev.kursor.ktensorflow.InterpreterOptions
import dev.kursor.ktensorflow.ModelDesc
import dev.kursor.ktensorflow.TensorFlowException
import dev.kursor.ktensorflow.tensor.Tensor
import dev.kursor.ktensorflow.tensor.TensorShape
import dev.kursor.ktensorflow.tensor.run
import kotlin.test.Test
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

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

    // --- одинаковые ожидания на обеих платформах ---

    @Test
    fun runRejectsMissingInputs() {
        // iOS раньше запускал модель на входах от предыдущего вызова, Android падал
        val interpreter = createInterpreter("mnist", "tflite", null)

        assertFailsWith<TensorFlowException> { interpreter.run(emptyList(), mapOf(0 to ByteArray(40))) }

        interpreter.close()
    }

    @Test
    fun runRejectsAnOutputBufferShorterThanTheTensor() {
        // iOS раньше молча обрезал результат, Android падал
        val interpreter = createInterpreter("mnist", "tflite", null)

        assertFailsWith<TensorFlowException> {
            interpreter.run(listOf(ByteArray(28 * 28 * 4)), mapOf(0 to ByteArray(20)))
        }

        interpreter.close()
    }

    @Test
    fun runAcceptsAnOutputBufferLongerThanTheTensor() {
        val interpreter = createInterpreter("mnist", "tflite", null)
        val output = ByteArray(80) { 7 }

        interpreter.run(listOf(ByteArray(28 * 28 * 4)), mapOf(0 to output))

        assertTrue(output.drop(40).all { it == 7.toByte() }, "the tail past the tensor stays untouched")
        interpreter.close()
    }
}
