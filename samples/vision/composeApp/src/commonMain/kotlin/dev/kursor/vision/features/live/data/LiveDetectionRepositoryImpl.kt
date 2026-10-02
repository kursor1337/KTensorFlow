@file:OptIn(ExperimentalKTensorFlowApi::class)

package dev.kursor.vision.features.live.data

import dev.kursor.ktensorflow.ExperimentalKTensorFlowApi
import dev.kursor.ktensorflow.Interpreter
import dev.kursor.ktensorflow.InterpreterOptions
import dev.kursor.ktensorflow.ModelDesc
import dev.kursor.ktensorflow.compose.ComposeUri
import dev.kursor.ktensorflow.coroutines.processFlowDropping
import dev.kursor.ktensorflow.pipeline.Pipeline
import dev.kursor.ktensorflow.pipeline.builder.inference
import dev.kursor.ktensorflow.pipeline.builder.input
import dev.kursor.ktensorflow.pipeline.builder.output
import dev.kursor.ktensorflow.pipeline.linear
import dev.kursor.ktensorflow.pipeline.stage.Stage
import dev.kursor.ktensorflow.pipeline.stage.then
import dev.kursor.ktensorflow.pipeline.tuple
import dev.kursor.ktensorflow.tensor.Tensor
import dev.kursor.ktensorflow.tensor.TensorDataType
import dev.kursor.ktensorflow.tensor.TensorShape
import dev.kursor.ktensorflow.tensor.get
import dev.kursor.ktensorflow.vision.Image
import dev.kursor.ktensorflow.vision.ImageTensorLayout
import dev.kursor.ktensorflow.vision.PadInfo
import dev.kursor.ktensorflow.vision.PaddedImage
import dev.kursor.ktensorflow.vision.PixelFormat
import dev.kursor.ktensorflow.vision.Rect
import dev.kursor.ktensorflow.vision.nms
import dev.kursor.ktensorflow.vision.resizeWithPad
import dev.kursor.ktensorflow.vision.tensorize
import dev.kursor.vision.features.live.domain.DetectedObject
import dev.kursor.vision.features.live.domain.DetectionResult
import kotlinx.coroutines.flow.Flow
import ktensorflow.samples.vision.composeapp.generated.resources.Res

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

    private val modelPipeline = Pipeline
        .input(
            preprocessing = Stage<PaddedImage>()
                .tensorize()
        )
        .inference(interpreter)
        .output(
            name = "num_detections",
            dataType = TensorDataType.Float32,
            shape = TensorShape(1),
            postprocessing = Stage<Tensor<Float>>()
                .toDetectionCount()
        )
        .output(
            name = "detection_boxes",
            dataType = TensorDataType.Float32,
            shape = TensorShape(1, MAX_DETECTIONS, 4),
            postprocessing = Stage<Tensor<Float>>()
                .toBoundingBoxes(MAX_DETECTIONS)
        )
        .output(
            name = "detection_classes",
            dataType = TensorDataType.Float32,
            shape = TensorShape(1, MAX_DETECTIONS),
            postprocessing = Stage<Tensor<Float>>()
                .toClassIds(MAX_DETECTIONS)
        )
        .output(
            name = "detection_scores",
            dataType = TensorDataType.Float32,
            shape = TensorShape(1, MAX_DETECTIONS),
            postprocessing = Stage<Tensor<Float>>()
                .toScores(MAX_DETECTIONS)
        )
        .build()

    private val detection = Pipeline.linear<Image>()
        .then { frame -> frame.resizeWithPad(300, 300, closeOriginal = false) }
        .then { paddedImage -> paddedImage.info to modelPipeline.run(tuple(paddedImage)) }
        .then { (padInfo, result) ->
            val (count, boxes, classes, scores) = result
            mapToDetectionResult(
                count = count,
                boxes = boxes,
                classes = classes,
                scores = scores,
                padInfo = padInfo,
                labels = Labels
            )
        }

    override fun detectObjects(frames: Flow<Image>): Flow<DetectionResult> =
        detection.processFlowDropping(frames)
}

private fun <I> Stage<I, PaddedImage>.tensorize(): Stage<I, Tensor<UByte>> =
    this.then { paddedImage ->
        paddedImage.use {
            it.tensorize<UByte>(
                layout = ImageTensorLayout.NHWC,
                pixelFormat = PixelFormat.RGB
            )
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
                tensor[0, i, 3] // xmax
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
    labels: List<String?>,
): DetectionResult {
    // Число детекций приходит от модели: не больше, чем вмещают выходные буферы
    val detectedObjects = (0 until count.coerceIn(0, boxes.size)).map { i ->
        val score = scores[i]

        val classId = classes[i]
        val label = labels.getOrNull(classId) ?: "Unknown ($classId)"

        val box = boxes[i]
        val rect = Rect.fromNormalized(
            ymin = box[0],
            xmin = box[1],
            ymax = box[2],
            xmax = box[3],
            padInfo = padInfo
        )

        DetectedObject(
            label = label,
            confidence = score,
            boundingBox = rect,
            imageHeight = padInfo.originalHeight,
            imageWidth = padInfo.originalWidth
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

// Метки COCO в той нумерации, в которой их отдаёт модель: номер класса COCO минус один.
// В COCO номера идут с пропусками, и список из 80 меток подряд сдвигал все классы после
// fire hydrant - книга (83) получалась "Unknown". Пропуски здесь null
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
    null,
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
    null,
    "backpack",
    "umbrella",
    null,
    null,
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
    null,
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
    null,
    "dining table",
    null,
    null,
    "toilet",
    null,
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
    null,
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush"
)