import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import dev.kursor.ktensorflow.Interpreter
import dev.kursor.ktensorflow.InterpreterOptions
import dev.kursor.ktensorflow.ModelDesc
import dev.kursor.ktensorflow.tensor.Tensor
import dev.kursor.ktensorflow.tensor.TensorShape
import dev.kursor.ktensorflow.tensor.div
import dev.kursor.ktensorflow.tensor.run
import dev.kursor.ktensorflow.tensor.toArray
import dev.kursor.ktensorflow.tensor.toFloatTensor
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.FileInputStream
import java.nio.channels.FileChannel
import kotlin.random.Random

@RunWith(AndroidJUnit4::class)
class InterpreterTest {

    private val context = InstrumentationRegistry.getInstrumentation().targetContext

    @OptIn(ExperimentalUnsignedTypes::class)
    @Test
    fun testWithCpu() {
        val interpreter = createInterpreter(context, "mnist.tflite")

        val data = loadDataset(context, "mnist.csv")

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
