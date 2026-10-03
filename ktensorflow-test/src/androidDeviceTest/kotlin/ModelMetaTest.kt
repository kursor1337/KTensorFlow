import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import dev.kursor.ktensorflow.ExperimentalKTensorFlowApi
import dev.kursor.ktensorflow.pipeline.Pipeline
import dev.kursor.ktensorflow.pipeline.builder.inference
import dev.kursor.ktensorflow.pipeline.builder.input
import dev.kursor.ktensorflow.pipeline.builder.output
import dev.kursor.ktensorflow.pipeline.stage.Stage
import dev.kursor.ktensorflow.pipeline.tuple
import dev.kursor.ktensorflow.tensor.Tensor
import dev.kursor.ktensorflow.tensor.TensorDataType
import dev.kursor.ktensorflow.tensor.TensorShape
import dev.kursor.ktensorflow.tensor.div
import dev.kursor.ktensorflow.tensor.run
import dev.kursor.ktensorflow.tensor.toFloatTensor
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.test.assertFailsWith

/**
 * Заголовочная фича 2.0: интерпретатор извлекает метаданные модели (SignatureDef),
 * что позволяет адресовать выходы по человекочитаемому имени вместо индекса.
 * Проверяется на реальном устройстве через реальный интерпретатор.
 */
@RunWith(AndroidJUnit4::class)
@OptIn(ExperimentalUnsignedTypes::class, ExperimentalKTensorFlowApi::class)
class ModelMetaTest {

    private val context = InstrumentationRegistry.getInstrumentation().targetContext

    @Test
    fun modelMetaExposesConsistentInputsAndOutputs() {
        val interpreter = createInterpreter(context, "mnist.tflite", null)
        val meta = interpreter.getModelMeta()

        println("inputs:  " + meta.inputData.joinToString { "${it.index}:'${it.name}' ${it.shape}" })
        println("outputs: " + meta.outputData.joinToString { "${it.index}:'${it.name}' ${it.shape}" })

        assertTrue("model must expose at least one input", meta.inputData.isNotEmpty())
        assertTrue("model must expose at least one output", meta.outputData.isNotEmpty())

        (meta.inputData + meta.outputData).forEach { d ->
            assertTrue("tensor #${d.index} has a blank name", d.name.isNotBlank())
            assertTrue("tensor '${d.name}' has an empty shape", d.shape.isNotEmpty())
            assertTrue("tensor '${d.name}' has numElements=${d.numElements}", d.numElements > 0)
        }

        meta.inputData.forEachIndexed { i, d -> assertEquals("input index mismatch", i, d.index) }
        meta.outputData.forEachIndexed { i, d -> assertEquals("output index mismatch", i, d.index) }

        assertEquals("inputsByName lost entries", meta.inputData.size, meta.inputsByName.size)
        assertEquals("outputsByName lost entries", meta.outputData.size, meta.outputsByName.size)
        meta.outputData.forEach { d -> assertEquals(d, meta.outputsByName[d.name]) }
        meta.inputData.forEach { d -> assertEquals(d, meta.inputsByName[d.name]) }
    }

    @Test
    fun namedOutputProducesSameResultAsIndexBasedOutput() {
        val interpreter = createInterpreter(context, "mnist.tflite", null)
        val outputName = interpreter.getModelMeta().outputData.first().name

        val byIndex = Pipeline
            .input(Stage<Array<UByteArray>>().tensorize().floatify().normalize())
            .inference(interpreter)
            .output(
                index = 0,
                dataType = TensorDataType.Float32,
                shape = TensorShape(10),
                Stage<Tensor<Float>>().argmax()
            )
            .build()

        val byName = Pipeline
            .input(Stage<Array<UByteArray>>().tensorize().floatify().normalize())
            .inference(interpreter)
            .output(
                name = outputName,
                dataType = TensorDataType.Float32,
                shape = TensorShape(10),
                postprocessing = Stage<Tensor<Float>>().argmax()
            )
            .build()

        val data = loadDataset(context, "mnist.csv").take(25)
        assertTrue("dataset must not be empty", data.isNotEmpty())

        data.forEach { (_, image) ->
            assertEquals(
                "output '$outputName' must resolve to the same tensor as index 0",
                byIndex.run(tuple(image)).first,
                byName.run(tuple(image)).first
            )
        }
    }

    @Test
    fun unknownOutputNameFailsAndListsTheNamesTheModelDoesHave() {
        // Раньше неизвестное имя молча превращалось в индекс 0, и результат писался
        // не в тот буфер - на модели с несколькими выходами это не видно вообще
        val interpreter = createInterpreter(context, "mnist.tflite", null)
        val meta = interpreter.getModelMeta()
        val (_, image) = loadDataset(context, "mnist.csv").first()
        val input = (Tensor<UByte>(image).toFloatTensor() / 255f).toPhysical()
        val output = Tensor<Float>(shape = TensorShape(10)).toPhysical()

        val failure = assertFailsWith<IllegalArgumentException> {
            interpreter.run(listOf(input), mapOf("definitely-not-an-output" to output))
        }

        val message = failure.message.orEmpty()
        assertTrue("message was: $message", message.contains("definitely-not-an-output"))
        meta.outputData.forEach { data ->
            assertTrue("message must list '${data.name}', was: $message", message.contains(data.name))
        }
    }

    @Test
    fun pipelineBuilderRejectsAnUnknownOutputName() {
        val interpreter = createInterpreter(context, "mnist.tflite", null)

        val failure = assertFailsWith<IllegalStateException> {
            Pipeline
                .input(Stage<Array<UByteArray>>().tensorize().floatify().normalize())
                .inference(interpreter)
                .output(
                    name = "definitely-not-an-output",
                    dataType = TensorDataType.Float32,
                    shape = TensorShape(10),
                    postprocessing = Stage<Tensor<Float>>().argmax()
                )
        }

        assertTrue(
            "message was: ${failure.message}",
            failure.message.orEmpty().contains("definitely-not-an-output")
        )
    }
}
