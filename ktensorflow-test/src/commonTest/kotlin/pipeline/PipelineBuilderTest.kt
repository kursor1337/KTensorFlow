package pipeline

import dev.kursor.ktensorflow.DataType
import dev.kursor.ktensorflow.ExperimentalKTensorFlowApi
import dev.kursor.ktensorflow.InternalKTensorFlowApi
import dev.kursor.ktensorflow.Interpreter
import dev.kursor.ktensorflow.ModelMeta
import dev.kursor.ktensorflow.ModelTensorData
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
import kotlin.test.assertFailsWith

/**
 * Билдер пайплайна связывает выходы модели со стадиями постобработки. Ошибка в этой связке
 * не падает, а молча отдаёт стадии чужой тензор, поэтому проверяется на фейковом
 * интерпретаторе, где каждый выход помечен своим индексом.
 */
@OptIn(ExperimentalKTensorFlowApi::class, InternalKTensorFlowApi::class)
class PipelineBuilderTest {

    /**
     * Пишет в каждый выход номер его индекса и запоминает первое значение каждого входа.
     * Выход с индексом i называется "out$i".
     */
    private class IndexEchoInterpreter(
        override val inputTensorCount: Int = 1,
        override val outputTensorCount: Int = 2
    ) : Interpreter {
        var receivedInputs: List<Float> = emptyList()
            private set

        override fun getModelMeta(): ModelMeta = ModelMeta(
            inputData = List(inputTensorCount) { tensorData(it, "in$it") },
            outputData = List(outputTensorCount) { tensorData(it, "out$it") }
        )

        private fun tensorData(index: Int, name: String) =
            ModelTensorData(index, name, name, DataType.Float32, listOf(1))

        override fun resizeInput(index: Int, dims: IntArray) = Unit

        override fun run(inputs: List<ByteArray>, outputs: Map<Int, ByteArray>) {
            receivedInputs = inputs.map { Float.fromBits(it.intAt(0)) }
            outputs.forEach { (index, bytes) ->
                val bits = index.toFloat().toBits()
                for (i in bytes.indices step 4) {
                    for (b in 0 until 4) bytes[i + b] = ((bits shr (8 * b)) and 0xFF).toByte()
                }
            }
        }

        override fun close() = Unit

        private fun ByteArray.intAt(offset: Int): Int =
            (0 until 4).fold(0) { acc, b -> acc or ((this[offset + b].toInt() and 0xFF) shl (8 * b)) }
    }

    private fun scalar(value: Float) = Tensor<Float>(shape = TensorShape(1)).apply { setFlat(0, value) }

    private val readValue = Stage<Tensor<Float>, Float> { it.getFlat(0) }

    @Test
    fun outputsReachTheirPostprocessingWhateverTheDeclarationOrder() {
        // Выходы объявлены не по возрастанию индекса: раньше кортеж сортировался по индексу,
        // а стадии шли в порядке объявления, и каждая получала чужой тензор
        val pipeline = Pipeline
            .input(preprocessing = Stage<Tensor<Float>>())
            .inference(IndexEchoInterpreter(outputTensorCount = 3))
            .output(index = 2, dataType = TensorDataType.Float32, shape = TensorShape(1), postprocessing = readValue)
            .output(index = 0, dataType = TensorDataType.Float32, shape = TensorShape(1), postprocessing = readValue)
            .output(index = 1, dataType = TensorDataType.Float32, shape = TensorShape(1), postprocessing = readValue)
            .build()

        val (first, second, third) = pipeline.run(tuple(scalar(0f)))

        assertEquals(listOf(2f, 0f, 1f), listOf(first, second, third))
    }

    @Test
    fun namedOutputsReachTheirPostprocessingWhateverTheDeclarationOrder() {
        // С именами ошибка проявлялась естественнее всего: выходы объявляют в смысловом
        // порядке, а индексы в модели идут как придётся
        val pipeline = Pipeline
            .input(preprocessing = Stage<Tensor<Float>>())
            .inference(IndexEchoInterpreter())
            .output(name = "out1", dataType = TensorDataType.Float32, shape = TensorShape(1), postprocessing = readValue)
            .output(name = "out0", dataType = TensorDataType.Float32, shape = TensorShape(1), postprocessing = readValue)
            .build()

        val (first, second) = pipeline.run(tuple(scalar(0f)))

        assertEquals(1f, first)
        assertEquals(0f, second)
    }

    @Test
    fun declaringTheSameOutputTwiceIsRejected() {
        // Повтор схлопывался в один ключ карты выходов, и кортеж выходил короче списка стадий
        val builder = Pipeline
            .input(preprocessing = Stage<Tensor<Float>>())
            .inference(IndexEchoInterpreter())
            .output(index = 0, dataType = TensorDataType.Float32, shape = TensorShape(1), postprocessing = readValue)
            .output(name = "out0", dataType = TensorDataType.Float32, shape = TensorShape(1), postprocessing = readValue)

        assertFailsWith<IllegalArgumentException> { builder.build() }
    }

    @Test
    fun aPipelineAcceptsThreeInputsInTheirDeclaredOrder() {
        // Раньше не компилировалось: input с двумя входами возвращал неверный тип билдера,
        // и третий вход добавить было нельзя
        val interpreter = IndexEchoInterpreter(inputTensorCount = 3, outputTensorCount = 1)
        val pipeline = Pipeline
            .input(preprocessing = Stage<Float, Tensor<*>> { scalar(it) })
            .input(preprocessing = Stage<Float, Tensor<*>> { scalar(it * 10) })
            .input(preprocessing = Stage<Float, Tensor<*>> { scalar(it * 100) })
            .inference(interpreter)
            .output(index = 0, dataType = TensorDataType.Float32, shape = TensorShape(1), postprocessing = readValue)
            .build()

        pipeline.run(tuple(1f, 2f, 3f))

        assertEquals(listOf(1f, 20f, 300f), interpreter.receivedInputs)
    }
}
