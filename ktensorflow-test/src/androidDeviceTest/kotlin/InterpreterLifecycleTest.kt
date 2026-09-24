import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import dev.kursor.ktensorflow.TensorFlowException
import dev.kursor.ktensorflow.tensor.Tensor
import dev.kursor.ktensorflow.tensor.TensorShape
import dev.kursor.ktensorflow.tensor.run
import org.junit.Test
import org.junit.runner.RunWith
import java.util.concurrent.atomic.AtomicReference
import kotlin.concurrent.thread
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Жизненный цикл интерпретатора. Интерпретатор TensorFlow Lite не допускает одновременных
 * вызовов, а его Java-API их не синхронизирует: close() из другого потока во время инференса
 * освобождал нативный объект прямо под выполняющимся run и ронял процесс SIGSEGV'ом. Это рядовой
 * сценарий: ViewModel.onCleared() закрывает интерпретатор, пока runSuspend ещё работает.
 */
@RunWith(AndroidJUnit4::class)
class InterpreterLifecycleTest {

    private val context = InstrumentationRegistry.getInstrumentation().targetContext

    private fun input() = Tensor<Float>(shape = TensorShape(28, 28)).toPhysical()

    private fun output() = Tensor<Float>(shape = TensorShape(10)).toPhysical()

    @Test
    fun closingWhileAnotherThreadRunsInferenceFailsCleanlyInsteadOfCrashing() {
        repeat(20) { attempt ->
            val interpreter = createInterpreter(context, "mnist.tflite", null)
            val failure = AtomicReference<Throwable?>()
            val worker = thread {
                val input = input()
                val output = output()
                try {
                    while (true) interpreter.run(input, output)
                } catch (t: Throwable) {
                    failure.set(t)
                }
            }

            Thread.sleep((attempt % 10 + 1).toLong())
            interpreter.close()
            worker.join()

            assertTrue(failure.get() is TensorFlowException, "attempt $attempt: ${failure.get()}")
        }
    }

    @Test
    fun everyCallAfterCloseFailsWithTensorFlowException() {
        val interpreter = createInterpreter(context, "mnist.tflite", null)

        interpreter.close()

        assertFailsWith<TensorFlowException> { interpreter.run(input(), output()) }
        assertFailsWith<TensorFlowException> { interpreter.getModelMeta() }
        assertFailsWith<TensorFlowException> { interpreter.inputTensorCount }
    }

    @Test
    fun closingTwiceIsSafe() {
        val interpreter = createInterpreter(context, "mnist.tflite", null)

        interpreter.close()
        interpreter.close()
    }

    @Test
    fun modelMetaReflectsAResizedInput() {
        // Метаданные кэшируются, и resizeInput обязан этот кэш сбрасывать
        val interpreter = createInterpreter(context, "mnist.tflite", null)
        val before = interpreter.getModelMeta().inputData[0].shape
        val resized = before.toMutableList().also { it[0] = it[0] * 2 }

        interpreter.resizeInput(0, resized.toIntArray())

        assertEquals(resized, interpreter.getModelMeta().inputData[0].shape)
        interpreter.close()
    }
}
