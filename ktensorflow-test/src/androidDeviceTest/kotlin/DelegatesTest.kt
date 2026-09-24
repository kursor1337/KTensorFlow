import android.os.Build
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import dev.kursor.ktensorflow.Delegate
import dev.kursor.ktensorflow.Interpreter
import dev.kursor.ktensorflow.InterpreterOptions
import dev.kursor.ktensorflow.TensorFlowException
import dev.kursor.ktensorflow.gpu.GpuDelegate
import dev.kursor.ktensorflow.gpu.GpuDelegateOptions
import dev.kursor.ktensorflow.npu.NpuDelegate
import dev.kursor.ktensorflow.npu.NpuDelegateOptions
import dev.kursor.ktensorflow.setDelegates
import dev.kursor.ktensorflow.tensor.Tensor
import dev.kursor.ktensorflow.tensor.TensorShape
import dev.kursor.ktensorflow.tensor.div
import dev.kursor.ktensorflow.tensor.run
import dev.kursor.ktensorflow.tensor.toArray
import dev.kursor.ktensorflow.tensor.toFloatTensor
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue
import org.tensorflow.lite.Interpreter as TFLInterpreter

/**
 * Делегаты GPU и NNAPI: сборка опций, проверка доступности и - если устройство делегат
 * поддерживает - совпадение результата с обычным CPU-инференсом.
 *
 * Builder-перегрузки опций здесь не декорация: они принимают receiver'ом тип из
 * TensorFlow Lite, и сам факт компиляции этого файла проверяет, что этот тип доступен
 * потребителю модуля транзитивно.
 */
@RunWith(AndroidJUnit4::class)
@OptIn(ExperimentalUnsignedTypes::class)
class DelegatesTest {

    private val context = InstrumentationRegistry.getInstrumentation().targetContext

    private fun predictions(delegate: Delegate?): List<Int> {
        val interpreter = createInterpreter(context, "mnist.tflite", delegate)
        val data = loadDataset(context, "mnist.csv").take(16)

        val result = data.map { (_, image) ->
            val input = (Tensor<UByte>(image).toFloatTensor() / 255f).toPhysical()
            val output = Tensor<Float>(shape = TensorShape(10))
            interpreter.run(input, output)
            output.toArray<FloatArray>().withIndex().maxBy { it.value }.index
        }

        interpreter.close()
        return result
    }

    @Test
    fun gpuDelegateOptionsAcceptBothTheFlagsAndTheNativeBuilder() {
        val fromFlags = GpuDelegateOptions(precisionLossAllowed = true, quantizationEnabled = true)
        val fromBuilder = GpuDelegateOptions {
            setPrecisionLossAllowed(true)
            setQuantizedModelsAllowed(true)
        }

        assertEquals(true, fromFlags.tflOptions.isPrecisionLossAllowed)
        assertEquals(true, fromBuilder.tflOptions.isPrecisionLossAllowed)
        assertEquals(true, fromBuilder.tflOptions.areQuantizedModelsAllowed())
    }

    @Test
    fun npuDelegateOptionsAcceptBothThePartitionCountAndTheNativeBuilder() {
        val fromCount = NpuDelegateOptions(maxDelegatedPartitions = 3)
        val fromBuilder = NpuDelegateOptions { setMaxNumberOfDelegatedPartitions(5) }

        assertEquals(3, fromCount.tflOptions.maxNumberOfDelegatedPartitions)
        assertEquals(5, fromBuilder.tflOptions.maxNumberOfDelegatedPartitions)
    }

    @Test
    fun gpuDelegateReportsAvailabilityWithoutThrowing() {
        val delegate = GpuDelegate()

        // Доступность зависит от устройства; важно, что запрос не падает
        println("GPU delegate available: ${delegate.isAvailable}")
        assertTrue(delegate.isAvailable || !delegate.isAvailable)
    }

    @Test
    fun npuDelegateReportsAvailabilityWithoutThrowing() {
        val delegate = NpuDelegate()

        println("NPU delegate available: ${delegate.isAvailable}")
        assertTrue(delegate.isAvailable || !delegate.isAvailable)
    }

    @Test
    fun gpuDelegateProducesTheSamePredictionsAsTheCpu() {
        val delegate = GpuDelegate()
        if (!delegate.isAvailable) {
            println("GPU delegate is not supported on this device, skipping the parity check")
            return
        }

        assertEquals(predictions(null), predictions(delegate))
    }

    @Test
    fun npuDelegateProducesTheSamePredictionsAsTheCpu() {
        val delegate = NpuDelegate()
        if (!delegate.isAvailable) {
            println("NPU delegate is not supported on this device, skipping the parity check")
            return
        }

        assertEquals(predictions(null), predictions(delegate))
    }

    // --- где на самом деле отказывает NNAPI ---

    @Test
    fun npuDelegateThatCannotBeAppliedFailsAtInterpreterCreation() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O_MR1) {
            println("NNAPI requires Android 8.1+, skipping")
            return
        }

        // Конструктор NnApiDelegate ускоритель не проверяет, поэтому делегат считается доступным.
        // Отказ проявляется только когда интерпретатор применяет делегат к графу модели.
        val delegate = NpuDelegate(NpuDelegateOptions { setAcceleratorName("does-not-exist") })
        assertTrue(delegate.isAvailable, "construction alone must not probe the accelerator")

        val failure = assertFailsWith<TensorFlowException> {
            Interpreter(
                loadModel(context, "mnist.tflite"),
                InterpreterOptions(numThreads = 1, useXNNPACK = false, delegates = listOf(delegate))
            )
        }

        assertTrue(
            failure.cause?.message.orEmpty().contains("Failed to apply delegate"),
            "the platform failure must be kept as the cause, was: ${failure.cause}"
        )
    }

    // --- как список делегатов попадает в интерпретатор ---
    // Фейковые делегаты: addDelegate только складывает их в список опций, нативный код
    // не вызывается, поэтому семантику можно проверить на любом устройстве.

    private class FakeTflDelegate(val name: String) : org.tensorflow.lite.Delegate {
        override fun getNativeHandle(): Long = 0L
    }

    private class FakeDelegate(
        override val isAvailable: Boolean,
        private val delegate: org.tensorflow.lite.Delegate?
    ) : Delegate {
        var tflDelegateReads = 0
            private set

        override val tflDelegate: org.tensorflow.lite.Delegate?
            get() {
                tflDelegateReads++
                return delegate
            }
    }

    @Test
    fun everyAvailableDelegateIsPassedInListOrder() {
        val first = FakeTflDelegate("first")
        val second = FakeTflDelegate("second")

        val options = TFLInterpreter.Options().setDelegates(
            listOf(
                FakeDelegate(isAvailable = true, delegate = first),
                FakeDelegate(isAvailable = false, delegate = FakeTflDelegate("unavailable")),
                FakeDelegate(isAvailable = true, delegate = second)
            )
        )

        assertEquals(listOf(first, second), options.delegates)
    }

    @Test
    fun unavailableDelegatesAreSkippedWithoutTouchingTheirNativePart() {
        // У GPU-делегата обращение к tflDelegate сразу создаёт нативный делегат - ровно то,
        // от чего защищает проверка доступности. Поэтому до tflDelegate недоступного
        // делегата дело доходить не должно вовсе.
        val unavailable = FakeDelegate(isAvailable = false, delegate = FakeTflDelegate("gpu"))

        val options = TFLInterpreter.Options().setDelegates(listOf(unavailable))

        assertEquals(0, unavailable.tflDelegateReads, "tflDelegate of an unavailable delegate was read")
        assertTrue(options.delegates.isEmpty())
    }

    @Test
    fun anAvailableDelegateWithoutANativeDelegateIsSkipped() {
        val options = TFLInterpreter.Options().setDelegates(
            listOf(FakeDelegate(isAvailable = true, delegate = null))
        )

        assertTrue(options.delegates.isEmpty())
    }
}
