@file:OptIn(ExperimentalUnsignedTypes::class)

import dev.kursor.ktensorflow.ExperimentalKTensorFlowApi
import dev.kursor.ktensorflow.pipeline.Pipeline
import dev.kursor.ktensorflow.pipeline.builder.inference
import dev.kursor.ktensorflow.pipeline.builder.input
import dev.kursor.ktensorflow.pipeline.builder.output
import dev.kursor.ktensorflow.pipeline.linear
import dev.kursor.ktensorflow.pipeline.stage.Stage
import dev.kursor.ktensorflow.pipeline.stage.inference
import dev.kursor.ktensorflow.pipeline.stage.then
import dev.kursor.ktensorflow.pipeline.tuple
import dev.kursor.ktensorflow.tensor.Tensor
import dev.kursor.ktensorflow.tensor.TensorDataType
import dev.kursor.ktensorflow.tensor.TensorShape
import dev.kursor.ktensorflow.tensor.argmax
import dev.kursor.ktensorflow.tensor.normalize
import dev.kursor.ktensorflow.tensor.toArray
import dev.kursor.ktensorflow.tensor.toFloatTensor
import floatify
import kotlin.test.Test
import kotlin.test.assertTrue

@OptIn(ExperimentalKTensorFlowApi::class)
class PipelineTest {

    @Test
    fun testSimple() {
        val interpreter = createInterpreter("mnist", "tflite")
        val pipeline = Pipeline.linear<Array<UByteArray>>()
            .tensorize()
            .floatify()
            .normalize()
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
        val interpreter = createInterpreter("mnist", "tflite")
        val pipeline = Pipeline
            .input(
                Stage<Array<UByteArray>>()
                    .tensorize()
                    .floatify()
                    .normalize()
            )
            .inference(interpreter)
            .output(
                index = 0,
                dataType = TensorDataType.Float32,
                shape = TensorShape(10),
                Stage<Tensor<Float>>()
                    .argmax()
                    .classify(listOf("0", "1", "2", "3", "4", "5", "6", "7", "8", "9"))
            )
            .build()

        test { pipeline.run(tuple(it)).first }
    }

    private fun test(pipelineRun: (Array<UByteArray>) -> String) {

        val data = loadDataset("mnist", "csv")

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

@OptIn(ExperimentalKTensorFlowApi::class)
fun <I, O : Any> Stage<I, O>.tensorize(): Stage<I, Tensor<UByte>> = this.then { Tensor<UByte>(it) }

@OptIn(ExperimentalKTensorFlowApi::class)
fun <T> Stage<T, Tensor<UByte>>.floatify() = this.then {
    it.toFloatTensor()
}

@OptIn(ExperimentalKTensorFlowApi::class)
fun <T> Stage<T, Tensor<Float>>.normalize() = this.then {
    it.normalize()
}


@OptIn(ExperimentalKTensorFlowApi::class)
fun <T> Stage<T, Tensor<Float>>.argmax() = this.then {
    it.argmax()[0]
}

@OptIn(ExperimentalKTensorFlowApi::class)
fun <T, C> Stage<T, Int>.classify(classes: List<C>) = this.then { classes[it] }