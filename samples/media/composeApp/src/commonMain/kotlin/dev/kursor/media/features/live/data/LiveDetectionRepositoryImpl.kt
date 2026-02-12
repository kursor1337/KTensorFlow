package dev.kursor.media.features.live.data

import dev.kursor.ktensorflow.ExperimentalKTensorFlowApi
import dev.kursor.ktensorflow.Interpreter
import dev.kursor.ktensorflow.InterpreterOptions
import dev.kursor.ktensorflow.ModelDesc
import dev.kursor.ktensorflow.compose.ComposeUri
import dev.kursor.ktensorflow.gpu.GpuDelegate
import dev.kursor.ktensorflow.media.Image
import dev.kursor.ktensorflow.media.ImageTensor
import dev.kursor.ktensorflow.media.PixelFormat
import dev.kursor.ktensorflow.media.resize
import dev.kursor.ktensorflow.media.tensorize
import dev.kursor.ktensorflow.media.grayscale
import dev.kursor.ktensorflow.media.toImageTensor
import dev.kursor.ktensorflow.npu.NpuDelegate
import dev.kursor.ktensorflow.tensor.Tensor
import dev.kursor.ktensorflow.tensor.TensorDataType
import dev.kursor.ktensorflow.tensor.TensorShape
import dev.kursor.ktensorflow.tensor.argmax
import dev.kursor.ktensorflow.tensor.normalize
import dev.kursor.ktensorflow.tensor.run
import dev.kursor.ktensorflow.tensor.toArray
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import ktensorflow.samples.media.composeapp.generated.resources.Res
import kotlin.time.Clock
import kotlin.time.ExperimentalTime
import kotlin.time.measureTime

class LiveDetectionRepositoryImpl : LiveDetectionRepository {
    @OptIn(ExperimentalKTensorFlowApi::class)
    private val interpreter = Interpreter(
        modelDesc = ModelDesc.ComposeUri(Res.getUri("files/mnist.tflite")),
        options = InterpreterOptions(
            numThreads = 4,
            useXNNPACK = true,
            delegates = listOf()
        )
    )

    @OptIn(ExperimentalTime::class)
    override suspend fun detectDigit(image: Image): Int = withContext(Dispatchers.Default) {
        val start = Clock.System.now().toEpochMilliseconds()


        val stage1 = image
        val stage2: Image
        val resizeTime = measureTime {
            stage2 = stage1.resize(28, 28)
        }
        val stage3: Image
        val grayScaleTime = measureTime {
            stage3 = stage2.grayscale()
        }
        val stage4: ImageTensor<Float>
        val tensorizeTime = measureTime {
            stage4 = stage3.tensorize()
        }
        val stage5: Tensor<Float>
        val normalizeTime = measureTime {
            stage5 = stage4.normalize()
        }
        val stage6: ImageTensor<Float>
        val toImageTensor = measureTime {
            stage6 = stage5.toImageTensor(pixelFormat = PixelFormat.RGBA)
        }


//        val stage1 = image
//        val stage2: ImageTensor<Float>
//        val tensorizeTime = measureTime {
//            stage2 = stage1
//                .tensorize(pixelFormat = PixelFormat.RGBA)
//        }
//        val stage3: Tensor<Float>
//        val normalizeTime = measureTime {
//            stage3 = stage2.normalize()
//        }
//        val stage4: ImageTensor<Float>
//        val toImageTensor = measureTime {
//            stage4 = stage3.toImageTensor(pixelFormat = PixelFormat.RGBA)
//        }
//        val stage5: ImageTensor<Float>
//        val grayScaleTime = measureTime {
//            stage5 = stage4.toGrayscale()
//        }
//        val stage6: ImageTensor<Float>
//        val resizeTime = measureTime {
//            stage6 = stage5.resize(28, 28)
//        }

        val output = Tensor(
            dataType = TensorDataType.Float32,
            shape = TensorShape(10)
        )

        val inferenceTime = measureTime {
            interpreter.run(stage6, output)
        }
        val end = Clock.System.now().toEpochMilliseconds()
        println("inference time: $inferenceTime")
        println("time: ${end - start}")
        println("tensorize time: $tensorizeTime")
        println("normalize time: $normalizeTime")
        println("toImageTensor time: $toImageTensor")
        println("toGrayscale time: $grayScaleTime")
        println("resize time: $resizeTime")
        println("result: ${output.toArray<FloatArray>().toList()}")
        output.argmax()[0]
    }
}