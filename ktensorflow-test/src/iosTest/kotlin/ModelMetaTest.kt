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
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Заголовочная фича 2.0: интерпретатор извлекает метаданные модели (SignatureDef),
 * что позволяет адресовать выходы по человекочитаемому имени вместо индекса.
 * Проверяется на реальной модели через реальный интерпретатор.
 */
@OptIn(ExperimentalUnsignedTypes::class, ExperimentalKTensorFlowApi::class)
class ModelMetaTest {

    @Test
    fun `model meta exposes consistent inputs and outputs`() {
        val interpreter = createInterpreter("mnist", "tflite", null)
        val meta = interpreter.getModelMeta()

        println("inputs:  " + meta.inputData.joinToString { "${it.index}:'${it.name}' ${it.shape}" })
        println("outputs: " + meta.outputData.joinToString { "${it.index}:'${it.name}' ${it.shape}" })

        assertTrue(meta.inputData.isNotEmpty(), "model must expose at least one input")
        assertTrue(meta.outputData.isNotEmpty(), "model must expose at least one output")

        (meta.inputData + meta.outputData).forEach { d ->
            assertTrue(d.name.isNotBlank(), "tensor #${d.index} has a blank name")
            assertTrue(d.shape.isNotEmpty(), "tensor '${d.name}' has an empty shape")
            assertTrue(d.numElements > 0, "tensor '${d.name}' has numElements=${d.numElements}")
        }

        meta.inputData.forEachIndexed { i, d -> assertEquals(i, d.index, "input index mismatch") }
        meta.outputData.forEachIndexed { i, d -> assertEquals(i, d.index, "output index mismatch") }

        // имена должны быть уникальны, иначе адресация по имени теряет тензоры
        assertEquals(meta.inputData.size, meta.inputsByName.size, "inputsByName lost entries")
        assertEquals(meta.outputData.size, meta.outputsByName.size, "outputsByName lost entries")
        meta.outputData.forEach { d -> assertEquals(d, meta.outputsByName[d.name]) }
        meta.inputData.forEach { d -> assertEquals(d, meta.inputsByName[d.name]) }
    }

    @Test
    fun `named output produces the same result as index based output`() {
        val interpreter = createInterpreter("mnist", "tflite", null)
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

        val data = loadDataset("mnist", "csv").take(25)
        assertTrue(data.isNotEmpty(), "dataset must not be empty")

        data.forEach { (_, image) ->
            assertEquals(
                byIndex.run(tuple(image)).first,
                byName.run(tuple(image)).first,
                "output '$outputName' must resolve to the same tensor as index 0"
            )
        }
    }
}
