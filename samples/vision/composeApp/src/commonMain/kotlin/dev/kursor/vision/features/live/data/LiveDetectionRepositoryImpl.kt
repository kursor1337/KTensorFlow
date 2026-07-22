@file:OptIn(ExperimentalKTensorFlowApi::class)

package dev.kursor.vision.features.live.data

import dev.kursor.ktensorflow.ExperimentalKTensorFlowApi
import dev.kursor.ktensorflow.Interpreter
import dev.kursor.ktensorflow.InterpreterOptions
import dev.kursor.ktensorflow.ModelDesc
import dev.kursor.ktensorflow.compose.ComposeUri
import dev.kursor.ktensorflow.pipeline.Pipeline
import dev.kursor.ktensorflow.pipeline.Tuple
import dev.kursor.ktensorflow.pipeline.builder.inference
import dev.kursor.ktensorflow.pipeline.builder.input
import dev.kursor.ktensorflow.pipeline.builder.output
import dev.kursor.ktensorflow.pipeline.stage.Stage
import dev.kursor.ktensorflow.pipeline.stage.then
import dev.kursor.ktensorflow.vision.Image
import dev.kursor.ktensorflow.vision.ImageTensor
import dev.kursor.ktensorflow.vision.ImageTensorLayout
import dev.kursor.ktensorflow.vision.PixelFormat
import dev.kursor.ktensorflow.vision.resize
import dev.kursor.ktensorflow.vision.tensorize
import dev.kursor.ktensorflow.vision.grayscale
import dev.kursor.ktensorflow.vision.toImageTensor
import dev.kursor.ktensorflow.tensor.Tensor
import dev.kursor.ktensorflow.tensor.TensorDataType
import dev.kursor.ktensorflow.tensor.TensorShape
import dev.kursor.ktensorflow.tensor.argmax
import dev.kursor.ktensorflow.tensor.get
import dev.kursor.ktensorflow.tensor.normalize
import dev.kursor.ktensorflow.tensor.run
import dev.kursor.ktensorflow.tensor.toArray
import dev.kursor.ktensorflow.tensor.toFlatArray
import dev.kursor.ktensorflow.vision.PadInfo
import dev.kursor.ktensorflow.vision.PaddedImage
import dev.kursor.ktensorflow.vision.Rect
import dev.kursor.ktensorflow.vision.nms
import dev.kursor.ktensorflow.vision.resizeWithPad
import dev.kursor.ktensorflow.vision.tensorizeFloat
import dev.kursor.vision.features.live.domain.DetectedObject
import dev.kursor.vision.features.live.domain.DetectionResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import ktensorflow.samples.vision.composeapp.generated.resources.Res
import kotlin.time.Clock
import kotlin.time.ExperimentalTime
import kotlin.time.measureTime

private const val MAX_DETECTIONS = 100

class LiveDetectionRepositoryImpl : LiveDetectionRepository {

    private val interpreter = Interpreter(
        modelDesc = ModelDesc.ComposeUri(Res.getUri("files/mobile_net_ssd_v2.tflite")),
        options = InterpreterOptions(
            numThreads = 4,
            useXNNPACK = true
        )
    )
        .apply {
            resizeInput(0, intArrayOf(1, 300, 300, 3))
        }
        .also {
            println("kursor1337: modelMeta: ${it.getModelMeta()}")
        }

    private val pipeline = Pipeline
        .input(
            preprocessing = Stage<PaddedImage>()
                .then { paddedImage ->
                    paddedImage.use {
                        it.tensorize<UByte>(
                            layout = ImageTensorLayout.NHWC,
                            pixelFormat = PixelFormat.RGB
                        )
                    }
                }
        )
        .inference(interpreter)
        .output(
            index = 2,
            dataType = TensorDataType.Float32,
            shape = TensorShape(1),
            postprocessing = Stage<Tensor<Float>>()
                .toDetectionCount()
        )
        .output(
            index = 4,
            dataType = TensorDataType.Float32,
            shape = TensorShape(1, MAX_DETECTIONS, 4),
            postprocessing = Stage<Tensor<Float>>()
                .toBoundingBoxes(MAX_DETECTIONS)
        )
        .output(
            index = 5,
            dataType = TensorDataType.Float32,
            shape = TensorShape(1, MAX_DETECTIONS),
            postprocessing = Stage<Tensor<Float>>()
                .toClassIds(MAX_DETECTIONS)
        )
        .output(
            index = 6,
            dataType = TensorDataType.Float32,
            shape = TensorShape(1, MAX_DETECTIONS),
            postprocessing = Stage<Tensor<Float>>()
                .toScores(MAX_DETECTIONS)
        )
        .build()


    val dispatcher = Dispatchers.Default.limitedParallelism(1)
    @OptIn(ExperimentalTime::class)
    override suspend fun detectObjects(image: Image): DetectionResult =
        withContext(dispatcher) {
            val detectionResult: DetectionResult
            val time = measureTime {

                val paddedImage = image.resizeWithPad(300, 300)
                val inferenceResult = pipeline.run(Tuple.One(paddedImage))
                val count = inferenceResult.first
                val boxes = inferenceResult.second
                val classIds = inferenceResult.third
                val scores = inferenceResult.fourth
                detectionResult = mapToDetectionResult(
                    count = count,
                    boxes = boxes,
                    classes = classIds,
                    scores = scores,
                    padInfo = paddedImage.info,
                    labels = Labels
                )
            }
            println("time: $time")
            detectionResult
        }
}

private fun <I> Stage<I, Tensor<Float>>.toDetectionCount(): Stage<I, Int> = this.then { tensor ->
    tensor[0].toInt()
}

private fun <I> Stage<I, Tensor<Float>>.toBoundingBoxes(maxDetections: Int): Stage<I, Array<FloatArray>> =
    this.then { tensor ->
        Array(maxDetections) { i ->
            floatArrayOf(
                tensor[0, i, 0], // ymin
                tensor[0, i, 1], // xmin
                tensor[0, i, 2], // ymax
                tensor[0, i, 3]  // xmax
            )
        }
    }

private fun <I> Stage<I, Tensor<Float>>.toClassIds(
    maxDetections: Int
): Stage<I, IntArray> = this.then { tensor ->
    IntArray(maxDetections) { i ->
        tensor[0, i].toInt()
    }
}

private fun <I> Stage<I, Tensor<Float>>.toScores(
    maxDetections: Int
): Stage<I, FloatArray> = this.then { tensor ->
    FloatArray(maxDetections) { i ->
        tensor[0, i]
    }
}

fun mapToDetectionResult(
    count: Int,
    boxes: Array<FloatArray>,
    classes: IntArray,
    scores: FloatArray,
    padInfo: PadInfo,
    labels: List<String>,
): DetectionResult {
    val detectedObjects = mutableListOf<DetectedObject>()

    for (i in 0 until count) {
        val score = scores[i]

        val classId = classes[i]
        val label = labels.getOrElse(classId) { "Unknown ($classId)" }

        val box = boxes[i]
        val rect = Rect.fromNormalized(
            ymin = box[0],
            xmin = box[1],
            ymax = box[2],
            xmax = box[3],
            padInfo = padInfo
        )

        detectedObjects.add(
            DetectedObject(
                label = label,
                confidence = score,
                boundingBox = rect
            )
        )
    }

    val result = detectedObjects.nms(
        iouThreshold = 0.5f,
        scoreThreshold = 0.3f,
        scoreSelector = DetectedObject::confidence,
        boxSelector = DetectedObject::boundingBox,
        classSelector = DetectedObject::label
    )

    return DetectionResult(objects = result)
}

private val Labels = listOf(
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush"
)