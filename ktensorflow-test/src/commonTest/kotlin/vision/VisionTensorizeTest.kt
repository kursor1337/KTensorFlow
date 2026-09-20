package vision

import dev.kursor.ktensorflow.tensor.TensorDataType
import dev.kursor.ktensorflow.vision.Image
import dev.kursor.ktensorflow.vision.ImageTensor
import dev.kursor.ktensorflow.vision.ImageTensorLayout
import dev.kursor.ktensorflow.vision.Normalization
import dev.kursor.ktensorflow.vision.PixelFormat
import dev.kursor.ktensorflow.vision.tensorize
import dev.kursor.ktensorflow.vision.tensorizeBatch
import dev.kursor.ktensorflow.vision.tensorizeBatchFloat
import dev.kursor.ktensorflow.vision.tensorizeFloat
import dev.kursor.ktensorflow.vision.toImage
import dev.kursor.ktensorflow.vision.toImageTensor
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertSame
import kotlin.test.assertTrue

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

    // --- вырожденные размеры ---
    // Строка, столбец и один пиксель - те формы, на которых перепутанные width/height
    // и обход по страйдам ломаются, а на квадратной картинке выглядят правильно.

    @Test
    fun tensorizeHandlesASingleRowImage() {
        val image = probeImage(width = 6, height = 1, tag = 3)

        val tensor = image.tensorize(TensorDataType.UInt8)

        assertEquals(6, tensor.width)
        assertEquals(1, tensor.height)
        for (x in 0 until 6) {
            assertEquals(x.toUByte(), tensor[0, 0, x, format.rIndex], "red at ($x,0)")
            assertEquals(0.toUByte(), tensor[0, 0, x, format.gIndex], "green at ($x,0)")
        }
    }

    @Test
    fun tensorizeHandlesASingleColumnImage() {
        val image = probeImage(width = 1, height = 6, tag = 3)

        val tensor = image.tensorize(TensorDataType.UInt8)

        assertEquals(1, tensor.width)
        assertEquals(6, tensor.height)
        for (y in 0 until 6) {
            assertEquals(0.toUByte(), tensor[0, y, 0, format.rIndex], "red at (0,$y)")
            assertEquals(y.toUByte(), tensor[0, y, 0, format.gIndex], "green at (0,$y)")
        }
    }

    @Test
    fun tensorizeHandlesASinglePixelImage() {
        val image = probeImage(width = 1, height = 1, tag = 77)

        val tensor = image.tensorize(TensorDataType.UInt8)

        assertEquals(1, tensor.batch)
        assertEquals(1, tensor.width)
        assertEquals(1, tensor.height)
        assertEquals(77.toUByte(), tensor[0, 0, 0, format.bIndex])
    }

    @Test
    fun tensorizeBatchOfASingleImageMatchesTensorizeOfThatImage() {
        val image = probeImage(4, 3, tag = 5)

        val single = image.tensorize(TensorDataType.UInt8)
        val batched = listOf(image).tensorizeBatch(TensorDataType.UInt8)

        assertEquals(1, batched.batch)
        for (y in 0 until 3) {
            for (x in 0 until 4) {
                for (c in 0 until format.channels) {
                    assertEquals(single[0, y, x, c], batched[0, y, x, c], "channel $c at ($x,$y)")
                }
            }
        }
    }

    // --- валидация входа ---

    @Test
    fun tensorizeBatchRejectsAnEmptyList() {
        assertFailsWith<IllegalArgumentException> {
            emptyList<Image>().tensorizeBatch(TensorDataType.UInt8)
        }
        assertFailsWith<IllegalArgumentException> { emptyList<Image>().tensorizeBatchFloat() }
    }

    @Test
    fun tensorizeBatchRejectsImagesOfDifferentSizes() {
        val images = listOf(probeImage(4, 3, tag = 1), probeImage(3, 4, tag = 2))

        assertFailsWith<IllegalArgumentException> { images.tensorizeBatch(TensorDataType.UInt8) }
        assertFailsWith<IllegalArgumentException> { images.tensorizeBatchFloat() }
    }

    // --- обратное преобразование ---

    @Test
    fun toImageRestoresThePixelsProducedByTensorizeFloat() {
        val image = probeImage(5, 3, tag = 42)

        val restored = image.tensorizeFloat().toImage()

        assertEquals(5, restored.width)
        assertEquals(3, restored.height)
        assertPixelsApproxEqual(image.getPixels(), restored.getPixels())
    }

    @Test
    fun toImageUndoesTheNormalizationAppliedByTensorizeFloat() {
        val image = probeImage(4, 4, tag = 11)
        val normalization = Normalization.MinusOneToOne

        val tensor = image.tensorizeFloat(normalization = normalization)
        val restored = tensor.toImage(normalization = normalization)

        assertPixelsApproxEqual(image.getPixels(), restored.getPixels())
    }

    @Test
    fun toImageRoundTripSurvivesANonDefaultLayout() {
        val image = probeImage(5, 3, tag = 8)

        val restored = image
            .tensorizeFloat(layout = ImageTensorLayout.NCHW)
            .toImage()

        assertPixelsApproxEqual(image.getPixels(), restored.getPixels())
    }

    @Test
    fun toImageExtractsTheRequestedImageOfABatch() {
        val first = probeImage(4, 3, tag = 1)
        val second = probeImage(4, 3, tag = 2)

        val tensor = listOf(first, second).tensorizeBatchFloat()

        assertPixelsApproxEqual(first.getPixels(), tensor.toImage(batchIndex = 0).getPixels())
        assertPixelsApproxEqual(second.getPixels(), tensor.toImage(batchIndex = 1).getPixels())
    }

    @Test
    fun toImageRejectsABatchIndexOutsideTheTensor() {
        val tensor = listOf(probeImage(2, 2, tag = 1), probeImage(2, 2, tag = 2)).tensorizeBatchFloat()

        assertFailsWith<IllegalArgumentException> { tensor.toImage(batchIndex = 2) }
        assertFailsWith<IllegalArgumentException> { tensor.toImage(batchIndex = -1) }
    }

    @Test
    fun toImageClipsValuesOutsideTheColorRange() {
        val tensor = ImageTensor<Float>(1, 1, PixelFormat.RGB)
        val rgb = PixelFormat.RGB
        tensor[0, 0, rgb.rIndex] = -50f
        tensor[0, 0, rgb.gIndex] = 300f
        tensor[0, 0, rgb.bIndex] = 128f

        val pixel = tensor.toImage().getPixels()[0]

        assertEquals(0, (pixel shr 16) and 0xFF, "red must be clipped to 0")
        assertEquals(255, (pixel shr 8) and 0xFF, "green must be clipped to 255")
        assertEquals(128, pixel and 0xFF)
    }

    // --- toImageTensor ---

    @Test
    fun toImageTensorReturnsTheSameInstanceWhenFormatAndLayoutAlreadyMatch() {
        val tensor = probeImage(4, 3, tag = 1).tensorizeFloat()

        assertSame(tensor, tensor.toImageTensor(format, ImageTensorLayout.NHWC))
    }

    @Test
    fun toImageTensorRewrapsWhenTheLayoutDiffers() {
        val tensor = probeImage(4, 3, tag = 1).tensorizeFloat()

        val rewrapped = tensor.toImageTensor(format, ImageTensorLayout.NCHW)

        assertEquals(ImageTensorLayout.NCHW, rewrapped.layout)
        assertTrue(rewrapped !== tensor)
    }

    private fun assertPixelsApproxEqual(
        expected: IntArray,
        actual: IntArray,
        tolerance: Int = 2,
        where: String = ""
    ) {
        assertEquals(expected.size, actual.size, "pixel count $where")
        expected.indices.forEach { i ->
            for (shift in intArrayOf(0, 8, 16, 24)) {
                val e = (expected[i] shr shift) and 0xFF
                val a = (actual[i] shr shift) and 0xFF
                assertTrue(
                    abs(e - a) <= tolerance,
                    "$where pixel $i, channel at bit $shift: expected $e, got $a"
                )
            }
        }
    }

    // --- сверка с эталоном на полном переборе ---
    // Ниже значение каждого канала считается напрямую из пикселей, которые получает
    // тензоризация, и сверяется со всеми четырьмя вариантами сразу. Перебор идёт по всем
    // форматам, обоим layout'ам, вырожденным и обычным размерам и по размерам батча,
    // поэтому ошибка в раскладке не может спрятаться в комбинации, которую забыли проверить.

    private val allFormats = listOf(
        PixelFormat.Grayscale,
        PixelFormat.RGB,
        PixelFormat.BGR,
        PixelFormat.RGBA,
        PixelFormat.ARGB,
        PixelFormat.BGRA,
        PixelFormat.ABGR
    )

    private val allLayouts = listOf(ImageTensorLayout.NHWC, ImageTensorLayout.NCHW)

    private val allSizes = listOf(1 to 1, 5 to 1, 1 to 5, 3 to 4, 4 to 3)

    /** Ожидаемое значение канала [c] пикселя [p] для формата [pixelFormat]. */
    private fun expectedChannel(pixelFormat: PixelFormat, p: Int, c: Int): Int = when (pixelFormat) {
        PixelFormat.Grayscale -> p and 0xFF
        is PixelFormat.RGB -> when (c) {
            pixelFormat.rIndex -> (p shr 16) and 0xFF
            pixelFormat.gIndex -> (p shr 8) and 0xFF
            else -> p and 0xFF
        }

        is PixelFormat.RGBA -> when (c) {
            pixelFormat.rIndex -> (p shr 16) and 0xFF
            pixelFormat.gIndex -> (p shr 8) and 0xFF
            pixelFormat.bIndex -> p and 0xFF
            else -> (p shr 24) and 0xFF
        }
    }

    private fun colorfulImage(width: Int, height: Int, pixelFormat: PixelFormat, seed: Int): Image {
        var state = seed * 7919 + 13
        val pixels = IntArray(width * height) {
            state = state * 1103515245 + 12345
            // Альфа держим непрозрачной: на iOS она premultiplied, и полупрозрачные
            // пиксели округляются при обратном умножении - это отдельная тема, не раскладка
            (0xFF shl 24) or ((state ushr 8) and 0x00FFFFFF)
        }
        return Image(width, height, pixelFormat, pixels)
    }

    @Test
    fun everyTensorizeVariantMatchesTheReferenceForEveryFormatSizeAndLayout() {
        var checked = 0

        allFormats.forEach { pixelFormat ->
            allLayouts.forEach { layout ->
                allSizes.forEach { (width, height) ->
                    val image = colorfulImage(width, height, pixelFormat, seed = width * 31 + height)
                    val pixels = image.getPixels()

                    val asBytes = image.tensorize(TensorDataType.UInt8, layout, pixelFormat)
                    val asFloats = image.tensorizeFloat(layout, pixelFormat)
                    val batchedBytes = listOf(image).tensorizeBatch(TensorDataType.UInt8, layout, pixelFormat)
                    val batchedFloats = listOf(image).tensorizeBatchFloat(layout, pixelFormat)

                    for (y in 0 until height) {
                        for (x in 0 until width) {
                            val p = pixels[y * width + x]
                            for (c in 0 until pixelFormat.channels) {
                                val expected = expectedChannel(pixelFormat, p, c)
                                val where = "$pixelFormat $layout ${width}x$height at ($x,$y) c=$c"

                                assertEquals(expected.toUByte(), asBytes[0, y, x, c], "tensorize $where")
                                assertEquals(expected.toFloat(), asFloats[0, y, x, c], "tensorizeFloat $where")
                                assertEquals(
                                    expected.toUByte(),
                                    batchedBytes[0, y, x, c],
                                    "tensorizeBatch $where"
                                )
                                assertEquals(
                                    expected.toFloat(),
                                    batchedFloats[0, y, x, c],
                                    "tensorizeBatchFloat $where"
                                )
                                checked++
                            }
                        }
                    }
                }
            }
        }

        assertTrue(checked > 1000, "the sweep must actually compare something, checked=$checked")
    }

    @Test
    fun batchedTensorizeMatchesTheReferenceForEveryBatchSizeAndLayout() {
        listOf(1, 2, 3).forEach { batchSize ->
            allFormats.forEach { pixelFormat ->
                allLayouts.forEach { layout ->
                    val width = 4
                    val height = 3
                    val images = (0 until batchSize).map { index ->
                        colorfulImage(width, height, pixelFormat, seed = index + 1)
                    }
                    val pixelsPerImage = images.map { it.getPixels() }

                    val bytes = images.tensorizeBatch(TensorDataType.UInt8, layout, pixelFormat)
                    val floats = images.tensorizeBatchFloat(layout, pixelFormat)

                    assertEquals(batchSize, bytes.batch)
                    assertEquals(batchSize, floats.batch)

                    for (n in 0 until batchSize) {
                        for (y in 0 until height) {
                            for (x in 0 until width) {
                                val p = pixelsPerImage[n][y * width + x]
                                for (c in 0 until pixelFormat.channels) {
                                    val expected = expectedChannel(pixelFormat, p, c)
                                    val where = "$pixelFormat $layout n=$n ($x,$y) c=$c"

                                    assertEquals(expected.toUByte(), bytes[n, y, x, c], "tensorizeBatch $where")
                                    assertEquals(
                                        expected.toFloat(),
                                        floats[n, y, x, c],
                                        "tensorizeBatchFloat $where"
                                    )
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    @Test
    fun toImageIsTheExactInverseOfTensorizeFloatForEveryFormatAndLayout() {
        allFormats.forEach { pixelFormat ->
            allLayouts.forEach { layout ->
                val image = colorfulImage(4, 3, pixelFormat, seed = 5)
                val expected = image.getPixels()

                val restored = image.tensorizeFloat(layout, pixelFormat).toImage()

                assertPixelsApproxEqual(
                    expected,
                    restored.getPixels(),
                    where = "$pixelFormat $layout"
                )
            }
        }
    }
}
