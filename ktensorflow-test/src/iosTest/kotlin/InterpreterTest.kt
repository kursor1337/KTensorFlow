import dev.kursor.ktensorflow.tensor.Tensor
import dev.kursor.ktensorflow.tensor.TensorShape
import dev.kursor.ktensorflow.tensor.run
import dev.kursor.ktensorflow.tensor.toArray
import kotlin.math.sign
import kotlin.test.Test
import kotlin.test.assertTrue

class InterpreterTest {

    @OptIn(ExperimentalUnsignedTypes::class)
    @Test
    fun testWithCpu() {
        val interpreter = createInterpreter("mnist", "tflite")

        val data = loadDataset("mnist", "csv")

        var accuratePredictions = 0

        data.forEachIndexed { i, pair ->
            val (label, image) = pair

            val input = Tensor<UByte>(image)
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
