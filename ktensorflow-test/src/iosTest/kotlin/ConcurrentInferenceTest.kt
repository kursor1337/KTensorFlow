import dev.kursor.ktensorflow.coroutines.runSuspend
import dev.kursor.ktensorflow.tensor.Tensor
import dev.kursor.ktensorflow.tensor.TensorShape
import dev.kursor.ktensorflow.tensor.div
import dev.kursor.ktensorflow.tensor.toArray
import dev.kursor.ktensorflow.tensor.toFloatTensor
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.runBlocking
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Interpreter.run не потокобезопасен, а runSuspend уводит его на пул потоков. Здесь несколько
 * инференсов на ОДНОМ интерпретаторе запускаются одновременно: без сериализации нативный
 * интерпретатор вошёл бы в гонку и вернул мусор или упал.
 */
@OptIn(ExperimentalUnsignedTypes::class)
class ConcurrentInferenceTest {

    @Test
    fun concurrentRunSuspendOnASingleInterpreterStaysCorrect() = runBlocking {
        val interpreter = createInterpreter("mnist", "tflite", null)
        val data = loadDataset("mnist", "csv").take(24)

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
        assertTrue(accuracy > 0.9, "concurrent inference produced wrong results, accuracy=$accuracy")
    }
}
