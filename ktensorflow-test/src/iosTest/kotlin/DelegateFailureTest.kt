import cocoapods.TensorFlowLiteObjC.TFLCDelegate
import cocoapods.TensorFlowLiteObjC.TFLDelegate
import dev.kursor.ktensorflow.Delegate
import dev.kursor.ktensorflow.Interpreter
import dev.kursor.ktensorflow.InterpreterOptions
import dev.kursor.ktensorflow.TensorFlowException
import kotlinx.cinterop.COpaquePointer
import kotlinx.cinterop.COpaquePointerVar
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.allocArray
import kotlinx.cinterop.free
import kotlinx.cinterop.nativeHeap
import kotlinx.cinterop.set
import kotlinx.cinterop.staticCFunction
import kotlin.test.Test
import kotlin.test.assertFailsWith
import kotlin.test.assertIs
import kotlin.test.assertTrue

/**
 * Делегат, доступный на устройстве, может не примениться к модели (сломанный драйвер, нехватка
 * памяти). Раньше такой отказ уходил наружу общим "Failed to create the interpreter" без намёка
 * на делегат. Настоящие делегаты на симуляторе либо применяются, либо недоступны, поэтому отказ
 * моделируется нативной структурой TfLiteDelegate, у которой Prepare всегда возвращает ошибку.
 */
@OptIn(ExperimentalForeignApi::class)
class DelegateFailureTest {

    private class FailingTflDelegate : TFLDelegate() {
        // TfLiteDelegate: data_, Prepare, CopyFromBufferHandle, CopyToBufferHandle,
        // FreeBufferHandle, flags, opaque_delegate_builder
        val native = nativeHeap.allocArray<COpaquePointerVar>(7).also { fields ->
            for (i in 0 until 7) fields[i] = null
            // kTfLiteError
            fields[1] = staticCFunction { _: COpaquePointer?, _: COpaquePointer? -> 1 }
        }

        override fun cDelegate(): TFLCDelegate = native
    }

    private class FailingDelegate : Delegate {
        private val delegate = FailingTflDelegate()

        override val isAvailable = true
        override val tflDelegate: TFLDelegate = delegate

        override fun close() = nativeHeap.free(delegate.native)
    }

    @Test
    fun aDelegateThatCannotBeAppliedFailsWithAMessageThatNamesTheDelegate() {
        val delegate = FailingDelegate()

        val failure = assertFailsWith<TensorFlowException> {
            Interpreter(
                loadModel("mnist", "tflite"),
                InterpreterOptions(numThreads = 1, useXNNPACK = false, delegates = listOf(delegate))
            )
        }

        assertTrue(
            failure.message.orEmpty().startsWith("Failed to create the interpreter: a delegate could not be applied"),
            "was: ${failure.message}"
        )
        assertIs<TensorFlowException>(failure.cause, "the platform error must be kept as the cause")
        delegate.close()
    }
}
