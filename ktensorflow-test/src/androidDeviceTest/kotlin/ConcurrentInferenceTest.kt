import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import dev.kursor.ktensorflow.coroutines.runSuspend
import dev.kursor.ktensorflow.tensor.Tensor
import dev.kursor.ktensorflow.tensor.TensorShape
import dev.kursor.ktensorflow.tensor.div
import dev.kursor.ktensorflow.tensor.toArray
import dev.kursor.ktensorflow.tensor.toFloatTensor
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Interpreter.run не потокобезопасен, а runSuspend уводит его на пул потоков. Здесь несколько
 * инференсов на ОДНОМ интерпретаторе запускаются одновременно: без сериализации нативный
 * интерпретатор вошёл бы в гонку и вернул мусор или упал.
 */
@RunWith(AndroidJUnit4::class)
@OptIn(ExperimentalUnsignedTypes::class)
class ConcurrentInferenceTest {

    private val context = InstrumentationRegistry.getInstrumentation().targetContext

    @Test
    fun concurrentRunSuspendOnASingleInterpreterStaysCorrect() = runBlocking {
        val interpreter = createInterpreter(context, "mnist.tflite", null)
        val data = loadDataset(context, "mnist.csv").take(24)

        val predictions = data.map { (label, image) ->
            async {
                val input = (Tensor<UByte>(image).toFloatTensor() / 255f).toPhysical()
                val output = Tensor<Float>(shape = TensorShape(10))

                interpreter.runSuspend(input, output)

                label.toInt() to output.toArray<FloatArray>().withIndex().maxBy { it.value }.index
            }
        }.awaitAll()

        val accurate = predictions.count { (label, predicted) -> label == predicted }
        val accuracy = accurate.toDouble() / predictions.size
        println("concurrent accuracy: $accuracy")
        assertTrue("concurrent inference produced wrong results, accuracy=$accuracy", accuracy > 0.9)
    }
}
