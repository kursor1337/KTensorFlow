package vision

import dev.kursor.ktensorflow.tensor.TensorDataType
import dev.kursor.ktensorflow.vision.Image
import dev.kursor.ktensorflow.vision.ImageTensorLayout
import dev.kursor.ktensorflow.vision.Normalization
import dev.kursor.ktensorflow.vision.PixelFormat
import dev.kursor.ktensorflow.vision.tensorize
import dev.kursor.ktensorflow.vision.tensorizeBatch
import dev.kursor.ktensorflow.vision.tensorizeBatchFloat
import dev.kursor.ktensorflow.vision.tensorizeFloat
import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * Все варианты тензоризации должны раскладывать пиксели одинаково: значение канала `c`
 * пикселя `(x, y)` изображения `n` обязано оказаться в `tensor[n, y, x, c]`.
 *
 * Изображение строится так, что по значению канала однозначно восстанавливаются координаты,
 * поэтому перепутанные оси сразу видны, а не маскируются симметрией.
 */
class VisionTensorizeTest {

    private val format = PixelFormat.ARGB

    /** Пиксель (x, y) изображения с номером [tag]: r = x, g = y, b = tag. */
    private fun probeImage(width: Int, height: Int, tag: Int): Image {
        val pixels = IntArray(width * height) { i ->
            val x = i % width
            val y = i / width
            (0xFF shl 24) or (x shl 16) or (y shl 8) or tag
        }
        return Image(width, height, format, pixels)
    }

    @Test
    fun tensorizePlacesEveryChannelAtItsOwnCoordinates() {
        val width = 5
        val height = 3
        val image = probeImage(width, height, tag = 9)

        val tensor = image.tensorize(TensorDataType.UInt8)

        assertEquals(1, tensor.batch)
        assertEquals(width, tensor.width)
        assertEquals(height, tensor.height)

        for (y in 0 until height) {
            for (x in 0 until width) {
                assertEquals(x.toUByte(), tensor[0, y, x, format.rIndex], "red at ($x,$y)")
                assertEquals(y.toUByte(), tensor[0, y, x, format.gIndex], "green at ($x,$y)")
                assertEquals(9.toUByte(), tensor[0, y, x, format.bIndex], "blue at ($x,$y)")
                assertEquals(255.toUByte(), tensor[0, y, x, format.aIndex], "alpha at ($x,$y)")
            }
        }
    }

    @Test
    fun tensorizeBatchKeepsEveryImageAtItsOwnBatchIndex() {
        val width = 4
        val height = 3
        val images = listOf(probeImage(width, height, tag = 1), probeImage(width, height, tag = 2))

        val tensor = images.tensorizeBatch(TensorDataType.UInt8)

        assertEquals(2, tensor.batch)
        assertEquals(width, tensor.width)
        assertEquals(height, tensor.height)

        images.forEachIndexed { batchIndex, _ ->
            val tag = (batchIndex + 1).toUByte()
            for (y in 0 until height) {
                for (x in 0 until width) {
                    assertEquals(x.toUByte(), tensor[batchIndex, y, x, format.rIndex], "red at [$batchIndex]($x,$y)")
                    assertEquals(y.toUByte(), tensor[batchIndex, y, x, format.gIndex], "green at [$batchIndex]($x,$y)")
                    assertEquals(tag, tensor[batchIndex, y, x, format.bIndex], "blue at [$batchIndex]($x,$y)")
                }
            }
        }
    }

    @Test
    fun tensorizeBatchFloatKeepsEveryImageAtItsOwnBatchIndex() {
        val width = 4
        val height = 3
        val images = listOf(probeImage(width, height, tag = 1), probeImage(width, height, tag = 2))

        val tensor = images.tensorizeBatchFloat()

        assertEquals(2, tensor.batch)

        images.forEachIndexed { batchIndex, _ ->
            val tag = (batchIndex + 1).toFloat()
            for (y in 0 until height) {
                for (x in 0 until width) {
                    assertEquals(x.toFloat(), tensor[batchIndex, y, x, format.rIndex], "red at [$batchIndex]($x,$y)")
                    assertEquals(y.toFloat(), tensor[batchIndex, y, x, format.gIndex], "green at [$batchIndex]($x,$y)")
                    assertEquals(tag, tensor[batchIndex, y, x, format.bIndex], "blue at [$batchIndex]($x,$y)")
                }
            }
        }
    }

    @Test
    fun tensorizeBatchFloatAppliesNormalizationToEveryImage() {
        val images = listOf(probeImage(2, 2, tag = 10), probeImage(2, 2, tag = 20))

        val tensor = images.tensorizeBatchFloat(
            normalization = Normalization(meanB = 10f, stdB = 2f)
        )

        // (10 - 10) / 2 = 0 для первого изображения, (20 - 10) / 2 = 5 для второго
        assertEquals(0f, tensor[0, 0, 0, format.bIndex])
        assertEquals(5f, tensor[1, 0, 0, format.bIndex])
    }

    @Test
    fun tensorizeAndTensorizeFloatAgreeOnLayout() {
        val image = probeImage(5, 3, tag = 4)

        val asUByte = image.tensorize(TensorDataType.UInt8)
        val asFloat = image.tensorizeFloat()

        for (y in 0 until 3) {
            for (x in 0 until 5) {
                for (c in 0 until format.channels) {
                    assertEquals(
                        asUByte[0, y, x, c].toInt().toFloat(),
                        asFloat[0, y, x, c],
                        "channel $c at ($x,$y) must land at the same coordinates in both variants"
                    )
                }
            }
        }
    }

    @Test
    fun tensorizeSupportsNonDefaultLayout() {
        val width = 4
        val height = 3
        val image = probeImage(width, height, tag = 6)

        val nhwc = image.tensorize(TensorDataType.UInt8, layout = ImageTensorLayout.NHWC)
        val nchw = image.tensorize(TensorDataType.UInt8, layout = ImageTensorLayout.NCHW)

        assertEquals(nhwc.width, nchw.width)
        assertEquals(nhwc.height, nchw.height)
        for (y in 0 until height) {
            for (x in 0 until width) {
                for (c in 0 until format.channels) {
                    assertEquals(nhwc[0, y, x, c], nchw[0, y, x, c], "channel $c at ($x,$y)")
                }
            }
        }
    }
}
