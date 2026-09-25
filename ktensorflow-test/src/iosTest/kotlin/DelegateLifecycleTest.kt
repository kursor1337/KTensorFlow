import dev.kursor.ktensorflow.InterpreterOptions
import dev.kursor.ktensorflow.gpu.GpuDelegate
import dev.kursor.ktensorflow.npu.NpuDelegate
import kotlinx.cinterop.ExperimentalForeignApi
import kotlin.test.Test
import kotlin.test.assertFailsWith

/**
 * Контракт закрытия делегатов тот же, что на Android: повторный close безопасен, а закрытый
 * делегат явно отказывается участвовать в новом интерпретаторе.
 */
@OptIn(ExperimentalForeignApi::class)
class DelegateLifecycleTest {

    @Test
    fun closingADelegateTwiceIsSafe() {
        listOf(GpuDelegate(), NpuDelegate()).forEach { delegate ->
            delegate.close()
            delegate.close()
        }
    }

    @Test
    fun aClosedDelegateRefusesToGiveOutItsNativeDelegate() {
        listOf(GpuDelegate(), NpuDelegate()).forEach { delegate ->
            delegate.close()
            assertFailsWith<IllegalStateException> { delegate.tflDelegate }
        }
    }

    @Test
    fun anAvailableClosedDelegateCannotBeUsedForANewInterpreter() {
        listOf(GpuDelegate(), NpuDelegate()).filter { it.isAvailable }.forEach { delegate ->
            delegate.close()
            assertFailsWith<IllegalStateException> {
                InterpreterOptions(numThreads = 1, useXNNPACK = false, delegates = listOf(delegate))
            }
        }
    }
}
