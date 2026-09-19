package vision

import dev.kursor.ktensorflow.tensor.TensorDataType
import dev.kursor.ktensorflow.vision.Image
import dev.kursor.ktensorflow.vision.ImageTensorLayout
import dev.kursor.ktensorflow.vision.Normalization
import dev.kursor.ktensorflow.vision.PaddedImage
import dev.kursor.ktensorflow.vision.PixelFormat
import dev.kursor.ktensorflow.vision.resizeWithPad
import dev.kursor.ktensorflow.vision.tensorize
import dev.kursor.ktensorflow.vision.tensorizeFloat
import kotlin.random.Random
import kotlin.test.Test
import kotlin.time.measureTime

/**
 * Замер препроцессинга из README-сценария реального времени: кадр с камеры 640x480
 * -> resizeWithPad(300, 300) -> tensorizeFloat.
 *
 * Важно про методику: на Android код исполняется ART, которому нужны СОТНИ вызовов, чтобы
 * дойти до оптимизированного машинного кода. При коротком прогреве замер показывает стоимость
 * интерпретации, а не установившуюся производительность, поэтому здесь много итераций прогрева,
 * а вместе со средним печатается минимум - он и отражает установившееся время кадра.
 */
class VisionPerformanceTest {

    private val frameWidth = 640
    private val frameHeight = 480
    private val modelSize = 300
    private val warmupIterations = 150
    private val measuredIterations = 50

    private fun cameraFrame(): Image {
        val random = Random(42)
        val pixels = IntArray(frameWidth * frameHeight) {
            (0xFF shl 24) or
                (random.nextInt(256) shl 16) or
                (random.nextInt(256) shl 8) or
                random.nextInt(256)
        }
        return Image(frameWidth, frameHeight, PixelFormat.ARGB, pixels)
    }

    private fun measure(name: String, block: () -> Unit) {
        repeat(warmupIterations) { block() }

        var best = Double.MAX_VALUE
        var totalMs = 0.0
        repeat(measuredIterations) {
            val elapsed = measureTime { block() }.inWholeMicroseconds / 1000.0
            totalMs += elapsed
            if (elapsed < best) best = elapsed
        }

        val meanMs = totalMs / measuredIterations
        val bestFps = if (best > 0) (1000.0 / best).toInt() else -1
        println("PERF $name: best ${best}ms (~$bestFps FPS), mean ${meanMs}ms")
    }

    @Test
    fun preprocessingThroughputForARealisticCameraFrame() {
        val frame = cameraFrame()
        val padded: PaddedImage = frame.resizeWithPad(modelSize, modelSize, closeOriginal = false)

        measure("resizeWithPad ${frameWidth}x$frameHeight -> ${modelSize}x$modelSize") {
            frame.resizeWithPad(modelSize, modelSize, closeOriginal = false).close()
        }

        measure("tensorizeFloat ${modelSize}x$modelSize") {
            padded.tensorizeFloat(normalization = Normalization.MinusOneToOne)
        }

        // NCHW раньше шёл по медленному пути с пересчётом смещения по страйдам на каждый
        // канал; после объединения оба layout'а идут одним и тем же кодом.
        measure("tensorizeFloat ${modelSize}x$modelSize NCHW") {
            padded.tensorizeFloat(
                layout = ImageTensorLayout.NCHW,
                normalization = Normalization.MinusOneToOne
            )
        }

        measure("tensorize UByte ${modelSize}x$modelSize") {
            padded.tensorize(TensorDataType.UInt8)
        }

        measure("FULL preprocessing (resizeWithPad + tensorizeFloat)") {
            val p = frame.resizeWithPad(modelSize, modelSize, closeOriginal = false)
            p.tensorizeFloat(normalization = Normalization.MinusOneToOne)
            p.close()
        }

        padded.close()
        frame.close()
    }
}
