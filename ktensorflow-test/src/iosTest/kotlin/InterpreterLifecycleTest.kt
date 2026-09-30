import dev.kursor.ktensorflow.TensorFlowException
import dev.kursor.ktensorflow.tensor.Tensor
import dev.kursor.ktensorflow.tensor.TensorShape
import dev.kursor.ktensorflow.tensor.run
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Жизненный цикл интерпретатора - см. одноимённый тест в androidDeviceTest. Раньше close() на iOS
 * ничего не делал, и run после него продолжал работать, в отличие от Android.
 */
class InterpreterLifecycleTest {

    private fun input() = Tensor<Float>(shape = TensorShape(28, 28)).toPhysical()

    private fun output() = Tensor<Float>(shape = TensorShape(10)).toPhysical()

    @Test
    fun everyCallAfterCloseFailsWithTensorFlowException() {
        val interpreter = createInterpreter("mnist", "tflite", null)

        interpreter.close()

        assertFailsWith<TensorFlowException> { interpreter.run(input(), output()) }
        assertFailsWith<TensorFlowException> { interpreter.getModelMeta() }
        assertFailsWith<TensorFlowException> { interpreter.inputTensorCount }
    }

    @Test
    fun closingTwiceIsSafe() {
        val interpreter = createInterpreter("mnist", "tflite", null)

        interpreter.close()
        interpreter.close()
    }

    @Test
    fun closingWhileAnotherThreadRunsInferenceFailsCleanly() = runBlocking {
        repeat(20) { attempt ->
            val interpreter = createInterpreter("mnist", "tflite", null)
            var failure: Throwable? = null
            val worker = launch(Dispatchers.Default) {
                val input = input()
                val output = output()
                try {
                    while (true) interpreter.run(input, output)
                } catch (t: TensorFlowException) {
                    failure = t
                }
            }

            delay((attempt % 10 + 1).toLong())
            interpreter.close()
            worker.join()

            assertTrue(failure is TensorFlowException, "attempt $attempt: $failure")
        }
    }

    @Test
    fun modelMetaReflectsAResizedInput() {
        val interpreter = createInterpreter("mnist", "tflite", null)
        val before = interpreter.getModelMeta().inputData[0].shape
        val resized = before.toMutableList().also { it[0] = it[0] * 2 }

        interpreter.resizeInput(0, resized.toIntArray())

        assertEquals(resized, interpreter.getModelMeta().inputData[0].shape)
        interpreter.close()
    }

    @Test
    fun aRejectedResizeLeavesTheInterpreterWorking() {
        // Раньше iOS молча принимал форму, с которой выход модели получал нулевую размерность,
        // и падал уже на следующем run или getModelMeta
        val interpreter = createInterpreter("mnist", "tflite", null)
        val original = interpreter.getModelMeta().inputData[0].shape

        for (dims in rejectedShapes) {
            assertFailsWith<TensorFlowException>(dims.contentToString()) { interpreter.resizeInput(0, dims) }
            assertEquals(original, interpreter.getModelMeta().inputData[0].shape, dims.contentToString())
            interpreter.run(input(), output())
        }

        val resized = original.toMutableList().also { it[0] = 2 }
        interpreter.resizeInput(0, resized.toIntArray())
        assertEquals(resized, interpreter.getModelMeta().inputData[0].shape)
        interpreter.close()
    }

    private val rejectedShapes = listOf(
        intArrayOf(),
        intArrayOf(0, 28, 28),
        intArrayOf(-1, 28, 28),
        intArrayOf(28),
        intArrayOf(1, 5, 5)
    )
}
