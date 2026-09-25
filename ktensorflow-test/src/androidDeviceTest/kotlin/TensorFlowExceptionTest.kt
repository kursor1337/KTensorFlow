import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import dev.kursor.ktensorflow.Interpreter
import dev.kursor.ktensorflow.InterpreterOptions
import dev.kursor.ktensorflow.ModelDesc
import dev.kursor.ktensorflow.TensorFlowException
import dev.kursor.ktensorflow.tensor.Tensor
import dev.kursor.ktensorflow.tensor.TensorShape
import dev.kursor.ktensorflow.tensor.run
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Отказ рантайма обязан приходить наружу как [TensorFlowException] на обеих платформах.
 * На iOS он и так приходил NSError'ом, а Java-API TensorFlow Lite бросал свои
 * IllegalArgumentException/IllegalStateException, и поймать это из общего кода было нечем.
 */
@RunWith(AndroidJUnit4::class)
class TensorFlowExceptionTest {

    private val context = InstrumentationRegistry.getInstrumentation().targetContext

    private fun brokenModel(): ModelDesc {
        val garbage = ByteArray(256) { it.toByte() }
        val buffer = ByteBuffer
            .allocateDirect(garbage.size)
            .order(ByteOrder.nativeOrder())
            .put(garbage)
            .apply { rewind() }
        return ModelDesc.ByteBuffer(buffer)
    }

    @Test
    fun loadingACorruptModelThrowsTensorFlowException() {
        val failure = assertFailsWith<TensorFlowException> {
            Interpreter(brokenModel(), InterpreterOptions())
        }

        assertNotNull("the original platform failure must be kept as the cause", failure.cause)
        assertTrue(
            "the message should say what failed, was: ${failure.message}",
            failure.message.orEmpty().contains("load the model")
        )
    }

    @Test
    fun runningInferenceWithAMismatchedShapeThrowsTensorFlowException() {
        val interpreter = createInterpreter(context, "mnist.tflite", null)

        // Модель ждёт 28x28, подаём заведомо не тот размер
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
        val interpreter = createInterpreter(context, "mnist.tflite", null)

        assertFailsWith<TensorFlowException> { interpreter.run(emptyList(), mapOf(0 to ByteArray(40))) }

        interpreter.close()
    }

    @Test
    fun runRejectsAnOutputBufferShorterThanTheTensor() {
        // iOS раньше молча обрезал результат, Android падал
        val interpreter = createInterpreter(context, "mnist.tflite", null)

        assertFailsWith<TensorFlowException> {
            interpreter.run(listOf(ByteArray(28 * 28 * 4)), mapOf(0 to ByteArray(20)))
        }

        interpreter.close()
    }

    @Test
    fun runAcceptsAnOutputBufferLongerThanTheTensor() {
        val interpreter = createInterpreter(context, "mnist.tflite", null)
        val output = ByteArray(80) { 7 }

        interpreter.run(listOf(ByteArray(28 * 28 * 4)), mapOf(0 to output))

        assertTrue(output.drop(40).all { it == 7.toByte() }, "the tail past the tensor stays untouched")
        interpreter.close()
    }
}
