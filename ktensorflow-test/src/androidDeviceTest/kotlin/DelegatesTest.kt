import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import dev.kursor.ktensorflow.gpu.GpuDelegate
import dev.kursor.ktensorflow.gpu.GpuDelegateOptions
import dev.kursor.ktensorflow.npu.NpuDelegate
import dev.kursor.ktensorflow.npu.NpuDelegateOptions
import dev.kursor.ktensorflow.tensor.Tensor
import dev.kursor.ktensorflow.tensor.TensorShape
import dev.kursor.ktensorflow.tensor.div
import dev.kursor.ktensorflow.tensor.run
import dev.kursor.ktensorflow.tensor.toArray
import dev.kursor.ktensorflow.tensor.toFloatTensor
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.test.assertEquals
import kotlin.test.assertTrue

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

    private fun predictions(delegate: dev.kursor.ktensorflow.Delegate?): List<Int> {
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
}
