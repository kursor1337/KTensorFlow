import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import dev.kursor.ktensorflow.Delegate
import dev.kursor.ktensorflow.Interpreter
import dev.kursor.ktensorflow.InterpreterOptions
import dev.kursor.ktensorflow.ModelDesc
import dev.kursor.ktensorflow.Tensor
import dev.kursor.ktensorflow.TensorDataType
import dev.kursor.ktensorflow.TensorShape
import dev.kursor.ktensorflow.typedData
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.FileInputStream
import java.nio.channels.FileChannel
import kotlin.random.Random

@RunWith(AndroidJUnit4::class)
class InterpreterTest {

    private val context = InstrumentationRegistry.getInstrumentation().targetContext

    @Test
    fun testWithCpu() {
        testWithHardware(emptyList())
        assertTrue(true)
    }

    fun testWithHardware(delegates: List<Delegate>) {
        val fileDescriptor = context.assets.openFd("mnist.tflite")
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val byteBuffer = fileChannel.map(
            FileChannel.MapMode.READ_ONLY,
            fileDescriptor.startOffset,
            fileDescriptor.declaredLength
        )
        val modelDesc = ModelDesc.ByteBuffer(byteBuffer)
        val options = InterpreterOptions(
            numThreads = 4,
            useXNNPACK = true,
            delegates = delegates
        )
        val interpreter = Interpreter(modelDesc, options)

        for (i in 0 until 100_000) {
            val inputArray = Array(28) {
                FloatArray(28) {
                    Random.nextFloat()
                }
            }
            val input = Tensor(inputArray)
            val output = Tensor(
                shape = TensorShape(10),
                dataType = TensorDataType.Float32
            )
            interpreter.run(listOf(input), listOf(output))
            val result = output
                .typedData<FloatArray>()
                .withIndex()
                .maxBy { it.value }
                .index
            println("test $i: result = $result")
        }
    }
}
