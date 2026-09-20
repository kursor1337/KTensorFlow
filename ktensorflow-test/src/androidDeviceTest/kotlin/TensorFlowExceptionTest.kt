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
}
