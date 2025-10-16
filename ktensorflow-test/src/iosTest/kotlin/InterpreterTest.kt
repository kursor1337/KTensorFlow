import dev.kursor.ktensorflow.Delegate
import dev.kursor.ktensorflow.gpu.GpuDelegate
import dev.kursor.ktensorflow.npu.NpuDelegate
import dev.kursor.ktensorflow.tensor.Tensor
import dev.kursor.ktensorflow.tensor.TensorShape
import dev.kursor.ktensorflow.tensor.div
import dev.kursor.ktensorflow.tensor.run
import dev.kursor.ktensorflow.tensor.toArray
import dev.kursor.ktensorflow.tensor.toFloatTensor
import kotlin.test.Test
import kotlin.test.assertTrue

class InterpreterTest {

    @Test
    fun testWithCpu() {
        test(null)
    }

    @Test
    fun testWithGpu() {
        test(GpuDelegate())
    }

    @Test
    fun testWithNpu() {
        test(NpuDelegate())
    }

    @OptIn(ExperimentalUnsignedTypes::class)
    fun test(delegate: Delegate?) {
        val interpreter = createInterpreter(
            modelFileName = "mnist",
            modelFileExtension = "tflite",
            delegate = delegate
        )

        val data = loadDataset("mnist", "csv")

        var accuratePredictions = 0

        data.forEachIndexed { i, pair ->
            val (label, image) = pair

            val input = Tensor<UByte>(image).toFloatTensor() / 255f

            val output = Tensor<Float>(
                shape = TensorShape(10)
            )
            interpreter.run(input, output)
            val result = output
                .toArray<FloatArray>()
                .withIndex()
                .maxBy { it.value }
                .index

            if (result == label.toInt()) {
                accuratePredictions++
            }
            println("test $i: result = $result")
        }
        val accuracy = accuratePredictions.toDouble() / data.size
        println("accuracy: $accuracy")
        assertTrue(accuracy > 0.9)
    }
}
