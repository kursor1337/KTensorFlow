package readme

import dev.kursor.ktensorflow.ExperimentalKTensorFlowApi
import dev.kursor.ktensorflow.Interpreter
import dev.kursor.ktensorflow.coroutines.mapAndClose
import dev.kursor.ktensorflow.coroutines.processFlowDropping
import dev.kursor.ktensorflow.coroutines.runSuspend
import dev.kursor.ktensorflow.pipeline.Pipeline
import dev.kursor.ktensorflow.pipeline.builder.inference
import dev.kursor.ktensorflow.pipeline.builder.input
import dev.kursor.ktensorflow.pipeline.builder.output
import dev.kursor.ktensorflow.pipeline.stage.Stage
import dev.kursor.ktensorflow.pipeline.stage.then
import dev.kursor.ktensorflow.pipeline.tuple
import dev.kursor.ktensorflow.tensor.Tensor
import dev.kursor.ktensorflow.tensor.TensorDataType
import dev.kursor.ktensorflow.tensor.TensorShape
import dev.kursor.ktensorflow.tensor.argmax
import dev.kursor.ktensorflow.tensor.run
import dev.kursor.ktensorflow.tensor.slice
import dev.kursor.ktensorflow.tensor.toArray
import dev.kursor.ktensorflow.tensor.transpose
import dev.kursor.ktensorflow.vision.Image
import dev.kursor.ktensorflow.vision.ImageTensor
import dev.kursor.ktensorflow.vision.ImageTensorLayout
import dev.kursor.ktensorflow.vision.Normalization
import dev.kursor.ktensorflow.vision.PaddedImage
import dev.kursor.ktensorflow.vision.Rect
import dev.kursor.ktensorflow.vision.nms
import dev.kursor.ktensorflow.vision.resizeWithPad
import dev.kursor.ktensorflow.vision.scaleForContainer
import dev.kursor.ktensorflow.vision.tensorizeFloat
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map
import kotlin.random.Random
import kotlin.test.Test

/**
 * Примеры из README.md, сохранённые в компилируемом виде.
 *
 * Тела функций намеренно не выполняются - утверждением здесь является сам факт компиляции.
 * Это защищает документацию от расхождения с реальным API: если сигнатура изменится или
 * функция исчезнет, сборка тестов упадёт, а не пользователь, скопировавший пример.
 */
@Suppress("UNUSED_PARAMETER", "unused")
@OptIn(ExperimentalKTensorFlowApi::class, ExperimentalUnsignedTypes::class)
class ReadmeExamplesTest {

    @Test
    fun readmeExamplesCompile() {
        // Компиляция этого файла и есть проверка. Тест держит класс "живым" для тест-раннера.
    }

    // --- README: Inference & Data Transformation ---
    private fun inference(interpreter: Interpreter) {
        val input = Tensor<Float>(Array(28) { FloatArray(28) { Random.nextFloat() } })
        val output = Tensor<Float>(shape = TensorShape(10), dataType = TensorDataType.Float32)

        interpreter.run(input, output)
        val result = output.argmax()[0]
    }

    // --- README: Zero-Copy Tensor Views ---
    private fun zeroCopyViews() {
        val tensor: Tensor<Float> = Tensor(Array(28) { FloatArray(28) { Random.nextFloat() } })

        val transposedView = tensor.transpose()
        val slicedView = tensor.slice(arrayOf(0..13, 0..27))

        val argmax: IntArray = tensor.argmax()
        val array = tensor.toPhysical().toArray<Array<FloatArray>>()
    }

    // --- README: Images & Tensorization ---
    private fun tensorization(originalImage: Image): ImageTensor<Float> {
        val paddedImage = originalImage.resizeWithPad(320, 320)

        return paddedImage.tensorizeFloat(
            layout = ImageTensorLayout.NHWC,
            normalization = Normalization.MinusOneToOne
        )
    }

    private data class DetectedObject(val label: Int, val confidence: Float, val rect: Rect)

    // --- README: Bounding Boxes & NMS ---
    private fun boundingBoxes(paddedImage: PaddedImage, count: Int): List<DetectedObject> {
        val detections = mutableListOf<DetectedObject>()

        for (i in 0 until count) {
            val rect = Rect.fromNormalized(
                ymin = 0f, xmin = 0f, ymax = 1f, xmax = 1f,
                padInfo = paddedImage.info
            )
            detections.add(DetectedObject(label = 0, confidence = 0.9f, rect = rect))
        }

        return detections.nms(
            iouThreshold = 0.5f,
            scoreThreshold = 0.3f,
            scoreSelector = { it.confidence },
            boxSelector = { it.rect },
            classSelector = { it.label }
        )
    }

    private fun drawRect(rect: Rect): Rect =
        rect.scaleForContainer(
            originalContainerWidth = 640,
            originalContainerHeight = 480,
            containerWidth = 1080f,
            containerHeight = 1920f,
            isCrop = true
        )

    // --- README: Pipelines (именованные выходы из SignatureDef) ---
    private fun detectionPipeline(interpreter: Interpreter) = Pipeline
        .input(
            Stage<PaddedImage>().then { image ->
                image.tensorizeFloat(normalization = Normalization.MinusOneToOne)
            }
        )
        .inference(interpreter)
        .output(
            name = "detection_boxes",
            dataType = TensorDataType.Float32,
            shape = TensorShape(1, 100, 4),
            postprocessing = Stage<Tensor<Float>>().then { tensor -> toBoundingBoxes(tensor) }
        )
        .output(
            name = "detection_classes",
            dataType = TensorDataType.Float32,
            shape = TensorShape(1, 100),
            postprocessing = Stage<Tensor<Float>>().then { tensor -> toClassIds(tensor) }
        )
        .build()

    private fun toBoundingBoxes(tensor: Tensor<Float>): List<Rect> = emptyList()

    private fun toClassIds(tensor: Tensor<Float>): List<Int> = emptyList()

    private fun runDetectionPipeline(interpreter: Interpreter, paddedImage: PaddedImage) {
        // Билдер строит Pipeline<Tuple.One<PaddedImage>, ...>, поэтому вход оборачивается в tuple()
        val (boxes, classes) = detectionPipeline(interpreter).run(tuple(paddedImage))
    }

    // --- README: Async Inference ---
    private suspend fun asyncInference(
        pipeline: Pipeline<PaddedImage, List<Rect>>,
        image: PaddedImage
    ): List<Rect> = pipeline.runSuspend(image)

    // --- README: Real-time Video Stream Processing (Backpressure / Frame Dropping) ---
    private fun videoStream(
        interpreter: Interpreter,
        frameFlow: Flow<Image>
    ): Flow<Int> = detectionPipeline(interpreter)
        // pipeline из билдера принимает Tuple.One<PaddedImage>, но Flow<PaddedImage> передаётся как есть
        .processFlowDropping(
            inputFlow = frameFlow.mapAndClose { it.resizeWithPad(300, 300) }
        )
        .map { tupleOutput -> tupleOutput.first.size + tupleOutput.second.size }
}
