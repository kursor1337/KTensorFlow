@file:OptIn(ExperimentalUnsignedTypes::class)

import android.content.Context
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import dev.kursor.ktensorflow.Interpreter
import dev.kursor.ktensorflow.InterpreterOptions
import dev.kursor.ktensorflow.ModelDesc
import dev.kursor.ktensorflow.Tensor
import dev.kursor.ktensorflow.TensorDataType
import dev.kursor.ktensorflow.TensorShape
import dev.kursor.ktensorflow.pipeline.Pipeline
import dev.kursor.ktensorflow.pipeline.Tuple
import dev.kursor.ktensorflow.pipeline.builder.inference
import dev.kursor.ktensorflow.pipeline.builder.input
import dev.kursor.ktensorflow.pipeline.builder.output
import dev.kursor.ktensorflow.pipeline.linear
import dev.kursor.ktensorflow.pipeline.stage.Stage
import dev.kursor.ktensorflow.pipeline.stage.inference
import dev.kursor.ktensorflow.pipeline.stage.then
import dev.kursor.ktensorflow.pipeline.tuple
import dev.kursor.ktensorflow.typedData
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.FileInputStream
import java.nio.channels.FileChannel

@RunWith(AndroidJUnit4::class)
class PipelineTest {

    private val context = InstrumentationRegistry.getInstrumentation().targetContext

    fun loadModel(context: Context, fileName: String): ModelDesc {
        val fileDescriptor = context.assets.openFd(fileName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val byteBuffer = fileChannel.map(
            FileChannel.MapMode.READ_ONLY,
            fileDescriptor.startOffset,
            fileDescriptor.declaredLength
        )
        return ModelDesc.ByteBuffer(byteBuffer)
    }

    fun loadDataset(context: Context, fileName: String): List<Pair<Byte, Array<UByteArray>>> {
        val dataset = context
            .assets
            .open(fileName)

        val csvDataFrame = CsvDataFrame(dataset)

        return csvDataFrame.extractImages()
    }

    fun createInterpreter(context: Context, modelFileName: String): Interpreter {
        val modelDesc = loadModel(context, modelFileName)

        val options = InterpreterOptions(
            numThreads = 4,
            useXNNPACK = true
        )

        return Interpreter(modelDesc, options)
    }

    @Test
    fun testSimple() {
        val interpreter = createInterpreter(context, "mnist.tflite")
        val pipeline = Pipeline.linear<Array<UByteArray>>()
            .floatify()
            .normalize()
            .tensorize()
            .inference(
                interpreter = interpreter,
                index = 0,
                dataType = TensorDataType.Float32,
                shape = TensorShape(10)
            )
            .argmax()
            .classify(listOf("0", "1", "2", "3", "4", "5", "6", "7", "8", "9"))

        test { pipeline.run(it) }
    }

    @Test
    fun testComplex() {
        val interpreter = createInterpreter(context, "mnist.tflite")
        val pipeline = Pipeline
            .input(
                Stage<Array<UByteArray>>()
                    .floatify()
                    .normalize()
                    .tensorize()
            )
            .inference(interpreter)
            .output(
                index = 0,
                dataType = TensorDataType.Float32,
                shape = TensorShape(10),
                Stage<Tensor>()
                    .argmax()
                    .classify(listOf("0", "1", "2", "3", "4", "5", "6", "7", "8", "9"))
            )
            .build()

        test { pipeline.run(tuple(it)).first }
    }

    private fun test(pipelineRun: (Array<UByteArray>) -> String) {

        val data = loadDataset(context, "mnist.csv")

        var accuratePredictions = 0

        data.forEachIndexed { index, pair ->
            val (label, image) = pair
            val result = pipelineRun(image)
            println("test $index: prediction = $result, label = $label")
            if (result == label.toString()) {
                accuratePredictions++
            }
        }
        val accuracy = accuratePredictions.toDouble() / data.size
        println("accuracy: $accuracy")
        assertTrue(accuracy > 0.9)
    }
}

fun CsvDataFrame.extractImages(): List<Pair<Byte, Array<UByteArray>>> =
    this.map {
        val label = it["label"].toByte()
        val image = Array(28) { i ->
            UByteArray(28) { j ->
                it["${i + 1}x${j + 1}"].toUByte()
            }
        }
        label to image
    }

fun <I, O : Any> Stage<I, O>.tensorize(): Stage<I, Tensor> = this.then { Tensor(it) }

fun <T> Stage<T, Array<UByteArray>>.floatify() = this.then {
    Array(it.size) { i ->
        FloatArray(it.size) { j ->
            it[i][j].toFloat()
        }
    }
}

fun <T> Stage<T, Array<FloatArray>>.normalize() = this.then {
    val maxValue = it.maxBy { it.max() }.max()
    Array(it.size) { i ->
        FloatArray(it[i].size) { j ->
            it[i][j] / maxValue
        }
    }
}


fun <T> Stage<T, Tensor>.argmax() = this.then {
    it.typedData<FloatArray>().withIndex().maxBy { it.value }.index
}

fun <T, C> Stage<T, Int>.classify(classes: List<C>) = this.then { classes[it] }