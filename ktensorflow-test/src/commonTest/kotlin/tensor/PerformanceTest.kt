package tensor

import dev.kursor.ktensorflow.media.ImageTensor
import dev.kursor.ktensorflow.media.PixelFormat
import dev.kursor.ktensorflow.media.resize
import dev.kursor.ktensorflow.media.grayscale
import dev.kursor.ktensorflow.tensor.Tensor
import dev.kursor.ktensorflow.tensor.TensorShape
import dev.kursor.ktensorflow.tensor.forEachIndexed
import dev.kursor.ktensorflow.tensor.map
import dev.kursor.ktensorflow.tensor.normalize
import dev.kursor.ktensorflow.tensor.toFlatIndex
import dev.kursor.ktensorflow.tensor.toNestedIndex
import kotlin.random.Random
import kotlin.test.Test
import kotlin.time.measureTime

class PerformanceTest {

    fun ByteArray.writeFloat(index: Int, value: Float) {
        val i = value.toBits()
        this[index] = (i shr 24 and 0xFF).toByte()
        this[index + 1] = (i shr 16 and 0xFF).toByte()
        this[index + 2] = (i shr 8 and 0xFF).toByte()
        this[index + 3] = (i and 0xFF).toByte()
    }

    @Test
    fun performanceTensor() {
        val data = FloatArray(10_000_000) { Random.nextFloat() }
        val timeBytes = measureTime {
            val array = ByteArray(40_000_000)
            for (i in data.indices) {
                array.writeFloat(i, data[i])
            }
        }
        val timeFloat = measureTime {
            val array = FloatArray(10_000_000)
            for (i in data.indices) {
                array[i] = data[i]
            }
        }

        println("Time bytes: $timeBytes")
        println("Time float: $timeFloat")

        val timeWithoutIndexTransformation = measureTime {
            val data2 = FloatArray(10_000_000)
            data.forEach {
                data2[it.toInt()] = it
            }
        }

        val shape = TensorShape(1000, 1000, 10)
        val multiDimIndexes = Array(10_000_000) { it.toNestedIndex(shape) }
        val timeToFlatIndex = measureTime {
            val data3 = FloatArray(10_000_000)
            data.forEachIndexed { index, it ->
                multiDimIndexes[index].toFlatIndex(shape)
                data3[it.toInt()] = it
            }
        }

        println("timeWithoutIndexTransformation: $timeWithoutIndexTransformation")
        println("timeToFlatIndex: $timeToFlatIndex")

        val tensorData = Array(1000) {
            Array(1000) {
                FloatArray(10) {
                    Random.nextFloat()
                }
            }
        }
        val tensor = Tensor<Float>(tensorData)
        val tensorForEachTimeInfer = measureTime {
            tensor.forEachIndexed { index, it ->
                val i = index[0]
            }
        }
        val tensorForEachTime = measureTime {
            val tensor = Tensor<Float>(shape)
            tensor.forEachIndexed { index, it ->
                val i = index[0]
            }
        }
        println("tensorForEachTimeInfer: $tensorForEachTimeInfer")
        println("tensorForEachTime: $tensorForEachTime")

        val mapTime = measureTime {
            tensor.map { it.toInt() }
        }
        println("mapTime: $mapTime")

        val normalizeTime = measureTime {
            tensor.normalize()
        }
        println("normalizeTime: $normalizeTime")
    }

    @Test
    fun performanceMedia() {
        val data = Array(2000) {
            Array(2000) {
                FloatArray(4) {
                    Random.nextFloat()
                }
            }
        }

        val imageTensor = ImageTensor(
            pixelFormat = PixelFormat.RGBA,
            data = data
        )
        val timeGrayscale = measureTime {
            imageTensor.grayscale()
        }

        println("timeGrayscale: $timeGrayscale")

        val timeResize = measureTime {
            imageTensor.resize(28, 28)
        }

        println("timeResize: $timeResize")
    }
}