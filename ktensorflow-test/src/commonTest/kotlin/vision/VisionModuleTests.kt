package vision

import dev.kursor.ktensorflow.vision.Image
import dev.kursor.ktensorflow.vision.ImageTensor
import dev.kursor.ktensorflow.vision.ImageTensorLayout
import dev.kursor.ktensorflow.vision.Normalization
import dev.kursor.ktensorflow.vision.PadInfo
import dev.kursor.ktensorflow.vision.PixelFormat
import dev.kursor.ktensorflow.vision.Rect
import dev.kursor.ktensorflow.vision.crop
import dev.kursor.ktensorflow.vision.grayscale
import dev.kursor.ktensorflow.vision.intersectionOverUnion
import dev.kursor.ktensorflow.vision.nms
import dev.kursor.ktensorflow.vision.resize
import dev.kursor.ktensorflow.vision.resizeWithPad
import dev.kursor.ktensorflow.vision.rotate
import dev.kursor.ktensorflow.vision.scaleForContainer
import dev.kursor.ktensorflow.vision.tensorize
import dev.kursor.ktensorflow.vision.tensorizeFloat
import dev.kursor.ktensorflow.vision.toImage
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFails
import kotlin.test.assertFailsWith
import kotlin.test.assertSame
import kotlin.test.assertTrue

class VisionModuleTests {

    // Вспомогательный метод для генерации однотонной залитой картинки
    private fun createSolidImage(width: Int, height: Int, colorArgb: Int): Image {
        val pixels = IntArray(width * height) { colorArgb }
        return Image(width, height, PixelFormat.ARGB, pixels)
    }

    // Вспомогательный метод для генерации картинки с уникальным цветом на каждый пиксель
    private fun createGradientImage(size: Int): Image {
        val pixels = IntArray(size * size) { i ->
            val x = i % size
            val y = i / size
            (0xFF shl 24) or (x shl 16) or (y shl 8)
        }
        return Image(size, size, PixelFormat.ARGB, pixels)
    }

    private fun assertFloatEquals(
        expected: Float,
        actual: Float,
        tolerance: Float = 1e-4f,
        message: String? = null
    ) {
        assertTrue(
            abs(expected - actual) <= tolerance,
            message ?: "expected=$expected actual=$actual (tolerance=$tolerance)"
        )
    }

    // Сравнение цветов ARGB с допуском - реальные платформенные реализации (Bitmap/CGImage)
    // могут давать расхождение в 1-2 единицы из-за округления (premultiplied alpha на iOS и т.д.)
    private fun assertColorApprox(expected: Int, actual: Int, tolerance: Int = 2) {
        for (shift in intArrayOf(24, 16, 8, 0)) {
            val e = (expected shr shift) and 0xFF
            val a = (actual shr shift) and 0xFF
            assertTrue(
                abs(e - a) <= tolerance,
                "channel at shift $shift differs: expected=$e actual=$a " +
                    "(expected=0x${expected.toUInt().toString(16)} actual=0x${actual.toUInt().toString(16)})"
            )
        }
    }

    // --- 1. ТЕСТЫ НА IMAGETENSOR И LAYOUT ---

    @Test
    fun imageTensorValidatesDimensionsUponCreation() {
        val tensor = ImageTensor<Float>(
            width = 100,
            height = 100,
            pixelFormat = PixelFormat.RGB,
            layout = ImageTensorLayout.NHWC,
            batchSize = 2
        )

        assertEquals(2, tensor.batch)
        assertEquals(100, tensor.width)
        assertEquals(100, tensor.height)
        assertEquals(3, tensor.channels)

        // Ранг должен быть ровно 4
        assertEquals(4, tensor.shape.rank)
    }

    @Test
    fun imageTensorWorksCorrectlyWithNCHWLayout() {
        val tensor = ImageTensor<Float>(
            width = 50,
            height = 60,
            pixelFormat = PixelFormat.RGBA,
            layout = ImageTensorLayout.NCHW, // Нестандартный лейаут
            batchSize = 1
        )

        // Проверяем, что API правильно мапит оси, независимо от внутреннего Shape
        assertEquals(1, tensor.batch)
        assertEquals(4, tensor.channels)
        assertEquals(60, tensor.height)
        assertEquals(50, tensor.width)

        tensor[0, 15, 25, 2] = 42f
        assertEquals(42f, tensor[0, 15, 25, 2])
    }

    // --- 2. ТЕСТЫ НА LETTERBOXING (RESIZE WITH PAD) ---

    @Test
    fun resizeWithPadScalesWideImageIntoSquareProperly() {
        // Исходник: Широкий 200x100
        val original = createSolidImage(200, 100, 0xFFFFFFFF.toInt())

        // Цель: Квадрат 200x200
        val padded = original.resizeWithPad(200, 200)

        assertEquals(200, padded.width)
        assertEquals(200, padded.height)

        val info = padded.info
        assertEquals(200, info.originalWidth)
        assertEquals(100, info.originalHeight)

        // Картинка должна была вписаться по ширине (scale = 1.0), а сверху/снизу получить паддинг по 50px
        assertEquals(1.0f, info.scale)
        assertEquals(0, info.padX)
        assertEquals(50, info.padY)
    }

    @Test
    fun resizeWithPadScalesTallImageIntoSquareProperly() {
        // Исходник: Высокий 100x400
        val original = createSolidImage(100, 400, 0xFFFFFFFF.toInt())

        // Цель: Квадрат 200x200
        val padded = original.resizeWithPad(200, 200)

        // Картинка должна была уменьшиться в 2 раза по высоте (scale = 0.5)
        // Новая ширина: 50. Значит паддинг по бокам (200 - 50) / 2 = 75px
        assertEquals(0.5f, padded.info.scale)
        assertEquals(75, padded.info.padX)
        assertEquals(0, padded.info.padY)
    }

    @Test
    fun resizeWithPadToTheExactSameSizeReturnsAUsableImage() {
        // Масштабировать нечего, и платформа возвращает ТОТ ЖЕ буфер. Если закрыть промежуточный
        // результат или исходник, отданное наружу изображение окажется закрытым.
        val color = (0xFF shl 24) or (10 shl 16) or (20 shl 8) or 30
        val original = createSolidImage(4, 4, color)

        val padded = original.resizeWithPad(4, 4)

        assertEquals(4, padded.width)
        assertEquals(4, padded.height)
        assertEquals(1.0f, padded.info.scale)
        assertEquals(0, padded.info.padX)
        assertEquals(0, padded.info.padY)
        padded.getPixels().forEach { assertColorApprox(color, it) }

        padded.close()
    }

    @Test
    fun resizeWithPadKeepsTheOriginalUsableWhenAskedNotToCloseIt() {
        val color = (0xFF shl 24) or (10 shl 16) or (20 shl 8) or 30
        val original = createSolidImage(4, 4, color)

        val padded = original.resizeWithPad(4, 4, closeOriginal = false)

        // Исходник обязан пережить преобразование, даже если результат разделяет с ним буфер
        original.getPixels().forEach { assertColorApprox(color, it) }
        padded.getPixels().forEach { assertColorApprox(color, it) }

        padded.close()
        original.close()
    }

    @Test
    fun resizeWithPadDownscalesWithoutPaddingForAMatchingAspectRatio() {
        val color = (0xFF shl 24) or (10 shl 16) or (20 shl 8) or 30
        val original = createSolidImage(8, 8, color)

        val padded = original.resizeWithPad(4, 4)

        assertEquals(4, padded.width)
        assertEquals(4, padded.height)
        assertEquals(0.5f, padded.info.scale)
        assertEquals(0, padded.info.padX)
        assertEquals(0, padded.info.padY)
        padded.getPixels().forEach { assertColorApprox(color, it) }

        padded.close()
    }

    @Test
    fun resizeWithPadFillsThePaddingWithTheRequestedColor() {
        val content = (0xFF shl 24) or (255 shl 16) or (255 shl 8) or 255
        val pad = (0xFF shl 24) or (0 shl 16) or (0 shl 8) or 0
        val original = createSolidImage(8, 4, content)

        val padded = original.resizeWithPad(8, 8, padColorArgb = pad)

        // Картинка вписывается по ширине, сверху и снизу остаётся по 2 строки паддинга
        assertEquals(2, padded.info.padY)
        val pixels = padded.getPixels()
        assertColorApprox(pad, pixels[0], tolerance = 2)
        assertColorApprox(pad, pixels[pixels.size - 1], tolerance = 2)
        assertColorApprox(content, pixels[4 * 8], tolerance = 2)

        padded.close()
    }

    // --- 3. ТЕСТЫ НА TENSORIZE И NORMALIZATION ---

    @Test
    fun tensorizeFloatCorrectlyAppliesMeanAndStd() {
        // Цвет (ARGB): A=255, R=100, G=150, B=200
        val color = (255 shl 24) or (100 shl 16) or (150 shl 8) or 200
        val img = createSolidImage(2, 2, color)

        val tensor = img.tensorizeFloat(
            normalization = Normalization(
                meanR = 100f,
                meanG = 100f,
                meanB = 100f,
                stdR = 50f,
                stdG = 50f,
                stdB = 50f
            )
        )

        // Проверяем формулу: (pixel - mean) / std
        // R = (100 - 100) / 50 = 0.0
        // G = (150 - 100) / 50 = 1.0
        // B = (200 - 100) / 50 = 2.0
        val pixelFormat = PixelFormat.ARGB
        assertEquals(0f, tensor[0, 0, 0, pixelFormat.rIndex])
        assertEquals(1f, tensor[0, 0, 0, pixelFormat.gIndex])
        assertEquals(2f, tensor[0, 0, 0, pixelFormat.bIndex])
    }

    @Test
    fun toImageCorrectlyDenormalizesAndClampsOutOfBoundPixels() {
        val tensor = ImageTensor<Float>(1, 1, PixelFormat.RGB)

        // Значения, которые при денормализации выйдут за пределы 0..255
        val pixelFormat = PixelFormat.RGB
        tensor[0, 0, 0, pixelFormat.rIndex] = -1.5f // Ожидаем 0
        tensor[0, 0, 0, pixelFormat.gIndex] = 0.5f // Ожидаем 127
        tensor[0, 0, 0, pixelFormat.bIndex] = 2.0f // Ожидаем 255

        val img = tensor.toImage(normalization = Normalization.ZeroToOne)
        val pixels = img.getPixels()

        val p = pixels[0]
        val r = (p shr 16) and 0xFF
        val g = (p shr 8) and 0xFF
        val b = p and 0xFF

        assertEquals(0, r, "Red channel should be clamped to 0")
        assertEquals(127, g, "Green channel should be 127")
        assertEquals(255, b, "Blue channel should be clamped to 255")
    }

    // --- 4. ТЕСТЫ НА МАППИНГ КООРДИНАТ (BOUNDING BOXES) ---

    @Test
    fun rectFromNormalizedTranslatesCoordinatesRemovingPadding() {
        val padInfo = PadInfo(
            originalWidth = 200,
            originalHeight = 100,
            targetWidth = 200,
            targetHeight = 200,
            padX = 0,
            padY = 50, // Картинка сдвинута вниз на 50px
            scale = 1.0f
        )

        // Представим, что сеть нашла объект ровно на всю оригинальную картинку.
        // На холсте 200x200 картинка лежит от Y=50 до Y=150.
        // Это нормализованные Y: 0.25 (50/200) и 0.75 (150/200).
        val rect = Rect.fromNormalized(
            ymin = 0.25f,
            xmin = 0.0f,
            ymax = 0.75f,
            xmax = 1.0f,
            padInfo = padInfo
        )

        // Ожидаем, что координаты вернутся к размеру оригинала 200x100
        assertEquals(0, rect.top)
        assertEquals(100, rect.bottom)
        assertEquals(0, rect.left)
        assertEquals(200, rect.right)
    }

    @Test
    fun rectFromNormalizedCoercesBoundsCorrectly() {
        val padInfo = PadInfo(
            originalWidth = 100,
            originalHeight = 100,
            targetWidth = 300,
            targetHeight = 300,
            padX = 0,
            padY = 0,
            scale = 3.0f
        )

        // Нейросеть выдала "шумные" координаты, выходящие за 0..1
        val rect = Rect.fromNormalized(
            ymin = -0.5f,
            xmin = -0.1f,
            ymax = 1.5f,
            xmax = 2.0f,
            padInfo = padInfo
        )

        // Координаты должны быть жестко усечены до размеров оригинала (0..100)
        assertEquals(0, rect.top)
        assertEquals(100, rect.bottom)
        assertEquals(0, rect.left)
        assertEquals(100, rect.right)
    }

    @Test
    fun rectFromYoloConvertsCenterSizeCoordinatesCorrectly() {
        val padInfo = PadInfo(100, 100, 100, 100, 0, 0, 1.0f)

        val rect = Rect.fromYolo(cx = 0.5f, cy = 0.5f, w = 0.4f, h = 0.2f, padInfo = padInfo)

        assertEquals(Rect(30, 40, 70, 60), rect)
    }

    @Test
    fun rectFromCocoConvertsTopLeftSizeCoordinatesCorrectly() {
        val padInfo = PadInfo(100, 100, 100, 100, 0, 0, 1.0f)

        val rect = Rect.fromCoco(x = 0.1f, y = 0.2f, w = 0.3f, h = 0.4f, padInfo = padInfo)

        assertEquals(Rect(10, 20, 40, 60), rect)
    }

    @Test
    fun intersectionOverUnionComputesCorrectRatioForPartiallyOverlappingRects() {
        val a = Rect(0, 0, 10, 10)
        val b = Rect(5, 5, 15, 15)

        assertFloatEquals(25f / 175f, a.intersectionOverUnion(b))
    }

    @Test
    fun intersectionOverUnionIsZeroForNonOverlappingRects() {
        val a = Rect(0, 0, 10, 10)
        val b = Rect(20, 20, 30, 30)

        assertEquals(0f, a.intersectionOverUnion(b))
    }

    @Test
    fun scaleForContainerMapsRectForCropAndFitModes() {
        val rect = Rect(10, 10, 20, 20)

        // Контейнер шире оригинала (200x100 из 100x100) - isCrop заполняет весь контейнер,
        // обрезая картинку сверху и снизу
        val cropped = rect.scaleForContainer(
            originalContainerWidth = 100,
            originalContainerHeight = 100,
            containerWidth = 200f,
            containerHeight = 100f,
            isCrop = true
        )
        assertEquals(Rect(20, -30, 40, -10), cropped)

        // isCrop = false - картинка вписывается целиком, с полями по бокам
        val fitted = rect.scaleForContainer(
            originalContainerWidth = 100,
            originalContainerHeight = 100,
            containerWidth = 200f,
            containerHeight = 100f,
            isCrop = false
        )
        assertEquals(Rect(60, 10, 70, 20), fitted)
    }

    @Test
    fun scaleForContainerReturnsEmptyRectForZeroSizeOriginalContainer() {
        val rect = Rect(1, 1, 2, 2).scaleForContainer(0, 0, 100f, 100f)

        assertEquals(Rect(0, 0, 0, 0), rect)
    }

    // --- 5. ТЕСТЫ НА NMS (NON-MAXIMUM SUPPRESSION) ---

    private data class Detection(val rect: Rect, val score: Float, val cls: Int = 0)

    @Test
    fun nmsSuppressesHeavilyOverlappingBoxesKeepingTheHighestScore() {
        val a = Detection(Rect(0, 0, 10, 10), 0.9f)
        val b = Detection(Rect(1, 1, 10, 10), 0.8f) // IoU с a = 0.81 > 0.45
        val c = Detection(Rect(50, 50, 60, 60), 0.7f) // без пересечения

        val result = listOf(a, b, c).nms<Detection, Any?>(
            iouThreshold = 0.45f,
            scoreThreshold = 0f,
            scoreSelector = { it.score },
            boxSelector = { it.rect }
        )

        assertEquals(listOf(a, c), result)
    }

    @Test
    fun nmsFiltersOutDetectionsBelowTheScoreThreshold() {
        val a = Detection(Rect(0, 0, 10, 10), 0.9f)
        val b = Detection(Rect(20, 20, 30, 30), 0.1f)

        val result = listOf(a, b).nms<Detection, Any?>(
            scoreThreshold = 0.5f,
            scoreSelector = { it.score },
            boxSelector = { it.rect }
        )

        assertEquals(listOf(a), result)
    }

    @Test
    fun nmsWithClassSelectorSuppressesIndependentlyPerClass() {
        val sameBox = Rect(0, 0, 10, 10)
        val a = Detection(sameBox, 0.9f, cls = 1)
        val b = Detection(sameBox, 0.85f, cls = 2)

        val withoutClasses = listOf(a, b).nms<Detection, Any?>(
            scoreThreshold = 0f,
            scoreSelector = { it.score },
            boxSelector = { it.rect }
        )
        assertEquals(listOf(a), withoutClasses, "single-class NMS should suppress the fully overlapping box")

        val withClasses = listOf(a, b).nms(
            scoreThreshold = 0f,
            scoreSelector = { it.score },
            boxSelector = { it.rect },
            classSelector = { it.cls }
        )
        assertEquals(
            setOf(a, b),
            withClasses.toSet(),
            "multi-class NMS should keep boxes from different classes independently"
        )
    }

    @Test
    fun nmsReturnsEmptyListForEmptyInput() {
        val result = emptyList<Detection>().nms<Detection, Any?>(
            scoreSelector = { it.score },
            boxSelector = { it.rect }
        )

        assertEquals(emptyList(), result)
    }

    @Test
    fun nmsKeepsABoxWhoseIouIsExactlyAtTheThreshold() {
        // Подавление объявлено как IoU СТРОГО больше порога, поэтому граничный бокс остаётся.
        // Значение зафиксировано тестом, иначе смена > на >= пройдёт незамеченной.
        val a = Detection(Rect(0, 0, 10, 10), 0.9f)
        val b = Detection(Rect(0, 0, 10, 5), 0.8f) // пересечение 50, объединение 100 -> IoU = 0.5

        val atThreshold = listOf(a, b).nms<Detection, Any?>(
            iouThreshold = 0.5f,
            scoreThreshold = 0f,
            scoreSelector = { it.score },
            boxSelector = { it.rect }
        )
        assertEquals(listOf(a, b), atThreshold, "IoU equal to the threshold must not suppress")

        val justBelowThreshold = listOf(a, b).nms<Detection, Any?>(
            iouThreshold = 0.49f,
            scoreThreshold = 0f,
            scoreSelector = { it.score },
            boxSelector = { it.rect }
        )
        assertEquals(listOf(a), justBelowThreshold, "IoU above the threshold must suppress")
    }

    @Test
    fun nmsKeepsADetectionWhoseScoreIsExactlyAtTheThreshold() {
        // Фильтр объявлен как score >= scoreThreshold, граница включительна
        val a = Detection(Rect(0, 0, 10, 10), 0.5f)

        val kept = listOf(a).nms<Detection, Any?>(
            scoreThreshold = 0.5f,
            scoreSelector = { it.score },
            boxSelector = { it.rect }
        )
        assertEquals(listOf(a), kept)

        val dropped = listOf(a).nms<Detection, Any?>(
            scoreThreshold = 0.5001f,
            scoreSelector = { it.score },
            boxSelector = { it.rect }
        )
        assertEquals(emptyList(), dropped)
    }

    @Test
    fun nmsKeepsEveryBoxWhenNothingOverlaps() {
        val boxes = (0 until 5).map { i ->
            Detection(Rect(i * 20, 0, i * 20 + 10, 10), 0.9f - i * 0.01f)
        }

        val result = boxes.nms<Detection, Any?>(
            scoreThreshold = 0f,
            scoreSelector = { it.score },
            boxSelector = { it.rect }
        )

        assertEquals(boxes, result)
    }

    @Test
    fun nmsCollapsesFullyIdenticalDetectionsToASingleOne() {
        val box = Rect(0, 0, 10, 10)
        val detections = List(4) { Detection(box, 0.9f) }

        val result = detections.nms<Detection, Any?>(
            scoreThreshold = 0f,
            scoreSelector = { it.score },
            boxSelector = { it.rect }
        )

        assertEquals(1, result.size, "identical boxes with identical scores must collapse to one")
    }

    @Test
    fun nmsReturnsEmptyListWhenEveryScoreIsBelowTheThreshold() {
        val detections = listOf(
            Detection(Rect(0, 0, 10, 10), 0.1f),
            Detection(Rect(50, 50, 60, 60), 0.2f)
        )

        val result = detections.nms<Detection, Any?>(
            scoreThreshold = 0.5f,
            scoreSelector = { it.score },
            boxSelector = { it.rect }
        )

        assertEquals(emptyList(), result)
    }

    @Test
    fun intersectionOverUnionOfARectWithItselfIsOne() {
        val rect = Rect(10, 20, 40, 60)

        assertFloatEquals(1f, rect.intersectionOverUnion(rect))
    }

    @Test
    fun intersectionOverUnionIsZeroForRectsThatOnlyTouchAtTheEdge() {
        val a = Rect(0, 0, 10, 10)
        val b = Rect(10, 0, 20, 10)

        assertFloatEquals(0f, a.intersectionOverUnion(b))
    }

    // --- 6. ТЕСТЫ НА ImageTensor grayscale/resize/crop (чистая логика над тензором) ---

    @Test
    fun imageTensorFloatGrayscaleAppliesITUR601LumaWeightsForRGB() {
        val tensor = ImageTensor<Float>(1, 1, PixelFormat.RGB)
        val pf = PixelFormat.RGB
        tensor[0, 0, pf.rIndex] = 100f
        tensor[0, 0, pf.gIndex] = 150f
        tensor[0, 0, pf.bIndex] = 200f

        val gray = tensor.grayscale()

        assertEquals(PixelFormat.Grayscale, gray.pixelFormat)
        assertFloatEquals(140.75f, gray[0, 0, 0], tolerance = 1e-3f)
    }

    @Test
    fun imageTensorFloatGrayscaleIgnoresAlphaForRGBA() {
        // Яркость не зависит от альфы: раньше она на неё умножалась, и при альфе = 255
        // (обычный тензор без нормализации) выходила в 255 раз больше
        val tensor = ImageTensor<Float>(1, 1, PixelFormat.ARGB)
        val pf = PixelFormat.ARGB
        tensor[0, 0, pf.rIndex] = 0.5f
        tensor[0, 0, pf.gIndex] = 0.6f
        tensor[0, 0, pf.bIndex] = 0.7f
        tensor[0, 0, pf.aIndex] = 0.5f

        val gray = tensor.grayscale()

        assertFloatEquals(0.5815f, gray[0, 0, 0], tolerance = 1e-3f)
    }

    @Test
    fun tensorGrayscaleOfAnUnnormalizedImageStaysWithinTheColorRange() {
        // Самый обычный путь: tensorizeFloat() без нормализации, затем grayscale
        val red = (0xFF shl 24) or (255 shl 16)
        val gray = Image(1, 1, PixelFormat.ARGB, intArrayOf(red)).tensorizeFloat().grayscale()

        assertFloatEquals(76.245f, gray[0, 0, 0], tolerance = 1e-2f)
    }

    @Test
    fun imageTensorFloatGrayscaleIsANoOpForAlreadyGrayscaleTensors() {
        val tensor = ImageTensor<Float>(2, 2, PixelFormat.Grayscale)

        assertSame(tensor, tensor.grayscale())
    }

    @Test
    fun imageTensorUByteGrayscaleUsesFixedPointLumaApproximation() {
        val tensor = ImageTensor<UByte>(1, 1, PixelFormat.RGB)
        val pf = PixelFormat.RGB
        tensor[0, 0, pf.rIndex] = 100.toUByte()
        tensor[0, 0, pf.gIndex] = 150.toUByte()
        tensor[0, 0, pf.bIndex] = 200.toUByte()

        val gray = tensor.grayscale()

        // (100*77 + 150*150 + 200*29) shr 8 = 36000 shr 8 = 140
        assertEquals(140.toUByte(), gray[0, 0, 0])
    }

    @Test
    fun imageTensorFloatResizeReturnsSameInstanceWhenDimensionsAreUnchanged() {
        val tensor = ImageTensor<Float>(4, 4, PixelFormat.Grayscale)

        assertSame(tensor, tensor.resize(4, 4))
    }

    @Test
    fun imageTensorFloatResizeBilinearlyInterpolatesBetweenSourcePixels() {
        val tensor = ImageTensor<Float>(2, 2, PixelFormat.Grayscale)
        tensor[0, 0, 0] = 0.0f // h=0, w=0
        tensor[0, 1, 0] = 0.3f // h=0, w=1
        tensor[1, 0, 0] = 0.6f // h=1, w=0
        tensor[1, 1, 0] = 1.0f // h=1, w=1

        val resized = tensor.resize(3, 3)

        // Углы должны сохраниться точно, середина - линейная интерполяция всех четырёх соседей
        assertFloatEquals(0.0f, resized[0, 0, 0])
        assertFloatEquals(1.0f, resized[2, 2, 0])
        assertFloatEquals(0.475f, resized[1, 1, 0], tolerance = 1e-4f)
    }

    @Test
    fun imageTensorUByteResizeUsesNearestNeighborSampling() {
        val tensor = ImageTensor<UByte>(2, 2, PixelFormat.Grayscale)
        tensor[0, 0, 0] = 10.toUByte() // h=0, w=0
        tensor[0, 1, 0] = 20.toUByte() // h=0, w=1
        tensor[1, 0, 0] = 30.toUByte() // h=1, w=0
        tensor[1, 1, 0] = 40.toUByte() // h=1, w=1

        val resized = tensor.resize(4, 4)

        assertEquals(10.toUByte(), resized[0, 0, 0])
        assertEquals(20.toUByte(), resized[1, 2, 0])
        assertEquals(30.toUByte(), resized[2, 0, 0])
        assertEquals(40.toUByte(), resized[3, 3, 0])
    }

    // Батч, не-NHWC layout и неквадратный размер - ровно те формы, на которых ошибки в
    // преобразованиях над тензором не видны при batch = 1 и квадратной картинке.

    @Test
    fun imageTensorFloatGrayscaleConvertsEveryImageOfABatch() {
        val pf = PixelFormat.RGB
        val tensor = ImageTensor<Float>(1, 1, pf, batchSize = 2)
        tensor[0, 0, 0, pf.rIndex] = 100f
        tensor[0, 0, 0, pf.gIndex] = 150f
        tensor[0, 0, 0, pf.bIndex] = 200f
        tensor[1, 0, 0, pf.rIndex] = 200f
        tensor[1, 0, 0, pf.gIndex] = 100f
        tensor[1, 0, 0, pf.bIndex] = 50f

        val gray = tensor.grayscale()

        assertEquals(2, gray.batch)
        assertFloatEquals(140.75f, gray[0, 0, 0, 0], tolerance = 1e-3f)
        assertFloatEquals(124.2f, gray[1, 0, 0, 0], tolerance = 1e-3f)
    }

    @Test
    fun imageTensorUByteGrayscaleConvertsEveryImageOfABatch() {
        val pf = PixelFormat.RGB
        val tensor = ImageTensor<UByte>(1, 1, pf, batchSize = 2)
        tensor[0, 0, 0, pf.rIndex] = 100.toUByte()
        tensor[0, 0, 0, pf.gIndex] = 150.toUByte()
        tensor[0, 0, 0, pf.bIndex] = 200.toUByte()
        tensor[1, 0, 0, pf.rIndex] = 200.toUByte()
        tensor[1, 0, 0, pf.gIndex] = 100.toUByte()
        tensor[1, 0, 0, pf.bIndex] = 50.toUByte()

        val gray = tensor.grayscale()

        assertEquals(2, gray.batch)
        // (100*77 + 150*150 + 200*29) shr 8 = 140, (200*77 + 100*150 + 50*29) shr 8 = 124
        assertEquals(140.toUByte(), gray[0, 0, 0, 0])
        assertEquals(124.toUByte(), gray[1, 0, 0, 0])
    }

    @Test
    fun imageTensorFloatGrayscaleKeepsTheSourceLayout() {
        val tensor = ImageTensor<Float>(2, 2, PixelFormat.RGB, layout = ImageTensorLayout.NCHW)

        val gray = tensor.grayscale()

        assertEquals(ImageTensorLayout.NCHW, gray.layout)
    }

    @Test
    fun imageTensorFloatResizeSupportsNonSquareTargets() {
        val tensor = ImageTensor<Float>(2, 2, PixelFormat.Grayscale)
        tensor[0, 0, 0] = 0.0f
        tensor[0, 1, 0] = 0.3f
        tensor[1, 0, 0] = 0.6f
        tensor[1, 1, 0] = 1.0f

        val resized = tensor.resize(newWidth = 4, newHeight = 2)

        assertEquals(4, resized.width)
        assertEquals(2, resized.height)
        // углы исходной картинки должны сохраниться точно
        assertFloatEquals(0.0f, resized[0, 0, 0])
        assertFloatEquals(0.3f, resized[0, 3, 0])
        assertFloatEquals(0.6f, resized[1, 0, 0])
        assertFloatEquals(1.0f, resized[1, 3, 0])
    }

    @Test
    fun imageTensorFloatResizeKeepsValuesOutsideTheZeroOneRange() {
        // tensorizeFloat() без нормализации даёт значения 0..255, и ресайз не вправе их обрезать
        val tensor = ImageTensor<Float>(2, 2, PixelFormat.Grayscale)
        tensor[0, 0, 0] = 0f
        tensor[0, 1, 0] = 255f
        tensor[1, 0, 0] = 128f
        tensor[1, 1, 0] = 255f

        val resized = tensor.resize(3, 3)

        assertFloatEquals(255f, resized[0, 2, 0], tolerance = 1e-2f)
        assertFloatEquals(128f, resized[2, 0, 0], tolerance = 1e-2f)
    }

    @Test
    fun imageTensorFloatResizeResizesEveryImageOfABatch() {
        val tensor = ImageTensor<Float>(2, 2, PixelFormat.Grayscale, batchSize = 2)
        tensor[0, 0, 0, 0] = 0.0f
        tensor[0, 0, 1, 0] = 0.3f
        tensor[0, 1, 0, 0] = 0.6f
        tensor[0, 1, 1, 0] = 1.0f
        tensor[1, 0, 0, 0] = 1.0f
        tensor[1, 0, 1, 0] = 0.7f
        tensor[1, 1, 0, 0] = 0.4f
        tensor[1, 1, 1, 0] = 0.0f

        val resized = tensor.resize(3, 3)

        assertEquals(2, resized.batch)
        assertFloatEquals(0.0f, resized[0, 0, 0, 0])
        assertFloatEquals(1.0f, resized[0, 2, 2, 0])
        assertFloatEquals(1.0f, resized[1, 0, 0, 0])
        assertFloatEquals(0.0f, resized[1, 2, 2, 0])
    }

    @Test
    fun imageTensorFloatResizeToASinglePixelStaysFinite() {
        val tensor = ImageTensor<Float>(2, 2, PixelFormat.Grayscale)
        tensor[0, 0, 0] = 0.25f
        tensor[0, 1, 0] = 0.5f
        tensor[1, 0, 0] = 0.75f
        tensor[1, 1, 0] = 1.0f

        val resized = tensor.resize(1, 1)

        assertEquals(1, resized.width)
        assertEquals(1, resized.height)
        assertFloatEquals(0.25f, resized[0, 0, 0])
    }

    // --- 7. ТЕСТЫ НА РЕАЛЬНОЙ ПЛАТФОРМЕННОЙ КАРТИНКЕ (Bitmap на Android, CGImage на iOS) ---
    // Все нижеследующие тесты гоняются как на реальном Android-устройстве/эмуляторе
    // (connectedAndroidTest), так и на реальном iOS-симуляторе (iosSimulatorArm64Test) -
    // Image() создаёт настоящий android.graphics.Bitmap / CGBitmapContext, моков нет.

    @Test
    fun imageCropAndImageTensorCropAgreeOnTheSameRegion() {
        val size = 6
        val original = createGradientImage(size)
        val rect = Rect(left = 1, top = 2, right = 5, bottom = 5)

        val croppedImage = original.crop(rect, closeOriginal = false)
        val fromImageCrop = croppedImage.tensorizeFloat()
        croppedImage.close()

        val fromTensorCrop = original.tensorizeFloat().crop(rect)

        original.close()

        assertEquals(rect.width, fromImageCrop.width)
        assertEquals(rect.height, fromImageCrop.height)
        assertEquals(fromImageCrop.width, fromTensorCrop.width)
        assertEquals(fromImageCrop.height, fromTensorCrop.height)

        for (h in 0 until rect.height) {
            for (w in 0 until rect.width) {
                for (c in 0 until fromImageCrop.channels) {
                    assertFloatEquals(
                        fromTensorCrop[h, w, c],
                        fromImageCrop[h, w, c],
                        tolerance = 1f,
                        message = "mismatch at h=$h w=$w c=$c"
                    )
                }
            }
        }
    }

    @Test
    fun imageResizePreservesASolidColorWhenScalingUpAndDown() {
        // Непрозрачный цвет без альфы, чтобы не задевать округление premultiplied alpha на iOS
        val color = (0xFF shl 24) or (10 shl 16) or (20 shl 8) or 30
        val original = createSolidImage(4, 4, color)

        val upscaled = original.resize(8, 8, closeOriginal = false)
        val downscaled = original.resize(2, 2, closeOriginal = true)

        assertEquals(8, upscaled.width)
        assertEquals(8, upscaled.height)
        upscaled.getPixels().forEach { assertColorApprox(color, it) }

        assertEquals(2, downscaled.width)
        assertEquals(2, downscaled.height)
        downscaled.getPixels().forEach { assertColorApprox(color, it) }

        upscaled.close()
        downscaled.close()
    }

    @Test
    fun imageRotate180DegreesPreservesDimensionsAndColorForASolidImage() {
        val color = (0xFF shl 24) or (50 shl 16) or (60 shl 8) or 70
        val original = createSolidImage(4, 6, color)

        val rotated = original.rotate(180f)

        assertEquals(4, rotated.width)
        assertEquals(6, rotated.height)
        rotated.getPixels().forEach { assertColorApprox(color, it) }

        rotated.close()
    }

    @Test
    fun imageGrayscalePreservesLuminanceForAnAlreadyNeutralColor() {
        // При r=g=b итоговая яркость не зависит от весов формулы (ITU-R 601 vs 709),
        // поэтому тест остаётся детерминированным на обеих платформах
        val gray = 150
        val color = (0xFF shl 24) or (gray shl 16) or (gray shl 8) or gray
        val original = createSolidImage(4, 4, color)

        val grayscaled = original.grayscale()

        assertEquals(PixelFormat.Grayscale, grayscaled.pixelFormat)
        grayscaled.getPixels().forEach { p ->
            val luma = p and 0xFF
            assertTrue(abs(luma - gray) <= 2, "expected ~$gray, got $luma")
        }

        grayscaled.close()
    }

    @Test
    fun rotateBy90DegreesSwapsDimensionsAndTurnsClockwiseOnBothPlatforms() {
        // Поворот кадра камеры на 90 градусов - самый частый случай, и направление
        // поворота обязано совпадать на обеих платформах: Matrix.postRotate и
        // CGContextRotateCTM считают угол в разных системах координат.
        val red = (0xFF shl 24) or (255 shl 16)
        val blue = (0xFF shl 24) or 255
        val original = Image(2, 1, PixelFormat.ARGB, intArrayOf(red, blue))

        val rotated = original.rotate(90f)

        assertEquals(1, rotated.width, "width and height must swap")
        assertEquals(2, rotated.height, "width and height must swap")

        // По часовой стрелке левый край становится верхним: левый пиксель уезжает наверх
        val pixels = rotated.getPixels()
        assertColorApprox(red, pixels[0], tolerance = 8)
        assertColorApprox(blue, pixels[1], tolerance = 8)

        rotated.close()
    }

    @Test
    fun grayscaleWorksForEveryPixelFormat() {
        // На iOS реализация принимала только RGBA и падала на RGB и на уже сером
        // изображении, тогда как на Android работала для любого формата
        val gray = 150
        val argb = (0xFF shl 24) or (gray shl 16) or (gray shl 8) or gray

        listOf(PixelFormat.ARGB, PixelFormat.RGBA, PixelFormat.RGB, PixelFormat.BGR).forEach { format ->
            val image = Image(2, 2, format, IntArray(4) { argb })

            val grayscaled = image.grayscale()

            assertEquals(PixelFormat.Grayscale, grayscaled.pixelFormat, "format $format")
            grayscaled.getPixels().forEach { p ->
                assertTrue(abs((p and 0xFF) - gray) <= 2, "format $format: expected ~$gray, got ${p and 0xFF}")
            }
            grayscaled.close()
        }
    }

    @Test
    fun grayscaleOfAnAlreadyGrayscaleImageIsStable() {
        val gray = 150
        val argb = (0xFF shl 24) or (gray shl 16) or (gray shl 8) or gray
        val once = Image(2, 2, PixelFormat.ARGB, IntArray(4) { argb }).grayscale()

        val twice = once.grayscale(closeOriginal = false)

        assertEquals(PixelFormat.Grayscale, twice.pixelFormat)
        twice.getPixels().forEach { p ->
            assertTrue(abs((p and 0xFF) - gray) <= 2, "expected ~$gray, got ${p and 0xFF}")
        }

        twice.close()
        once.close()
    }

    @Test
    fun grayscaleUsesRec601WeightsOnBothPlatforms() {
        // Чистый красный: Rec.601 даёт 0.299 * 255 = 76. Android раньше считал по Rec.709
        // через ColorMatrix.setSaturation и давал 54, то есть другое значение, чем iOS
        val red = (0xFF shl 24) or (255 shl 16)
        val green = (0xFF shl 24) or (255 shl 8)
        val blue = (0xFF shl 24) or 255

        listOf(red to 76, green to 150, blue to 29).forEach { (color, expected) ->
            val gray = Image(2, 2, PixelFormat.ARGB, IntArray(4) { color }).grayscale()
            val luma = gray.getPixels()[0] and 0xFF
            assertTrue(abs(luma - expected) <= 2, "color ${color.toString(16)}: expected ~$expected, got $luma")
            gray.close()
        }
    }

    // Случайные непрозрачные цвета с фиксированным зерном: одинаковые на обеих платформах
    private fun randomColors(count: Int): IntArray {
        var seed = 12345
        return IntArray(count) {
            seed = seed * 1103515245 + 12345
            (0xFF shl 24) or ((seed ushr 8) and 0xFFFFFF)
        }
    }

    private fun rec601(argb: Int): Float {
        val r = (argb shr 16) and 0xFF
        val g = (argb shr 8) and 0xFF
        val b = argb and 0xFF
        return 0.299f * r + 0.587f * g + 0.114f * b
    }

    @Test
    fun grayscaleFollowsRec601WithinOneLevelOnBothPlatforms() {
        // Яркость считают нативные реализации (ColorMatrix на Android, vImage на iOS), и
        // округляют они по-своему, поэтому допуск - один уровень. iOS раньше обрезал дробь,
        // и серый 128 превращался в 127
        val colors = randomColors(4096)
        val levels = IntArray(256) { (0xFF shl 24) or (it shl 16) or (it shl 8) or it }

        for (input in listOf(colors, levels)) {
            val side = if (input.size == 4096) 64 else 16
            val gray = Image(side, side, PixelFormat.ARGB, input).grayscale()

            val actual = gray.getPixels()
            input.forEachIndexed { i, color ->
                val expected = rec601(color)
                val luma = actual[i] and 0xFF
                assertTrue(
                    abs(luma - expected) <= 1f,
                    "color 0x${color.toUInt().toString(16)}: expected ~$expected, got $luma"
                )
            }
            gray.close()
        }
    }

    @Test
    fun ubyteTensorGrayscaleAgreesWithImageGrayscale() {
        // Два пути к серому - через Image и через ImageTensor<UByte> - дают одно и то же с
        // точностью до округления: тензор считает целочисленным приближением весов
        val colors = randomColors(4096)
        val image = Image(64, 64, PixelFormat.ARGB, colors)

        val fromTensor = image.tensorize<UByte>().grayscale()
        val fromImage = image.grayscale(closeOriginal = true)

        val expected = fromImage.getPixels()
        for (i in colors.indices) {
            val diff = abs((expected[i] and 0xFF) - fromTensor.getFlat(i).toInt())
            assertTrue(diff <= 2, "pixel $i differs by $diff")
        }
        fromImage.close()
    }

    // --- 3-канальные изображения ---

    @Test
    fun threeChannelImagesCanBeResizedCroppedAndRotated() {
        // На iOS CoreGraphics не поддерживает 3-канальные буферы, и resize/crop/rotate
        // RGB-изображения падали, тогда как на Android работали
        val color = (0xFF shl 24) or (10 shl 16) or (120 shl 8) or 230

        for (format in listOf(PixelFormat.RGB, PixelFormat.BGR)) {
            val image = createSolidImage(4, 6, color).let { argb ->
                Image(4, 6, format, argb.getPixels()).also { argb.close() }
            }

            val resized = image.resize(8, 3, closeOriginal = false)
            val cropped = image.crop(Rect(left = 1, top = 1, right = 3, bottom = 5), closeOriginal = false)
            val rotated = image.rotate(180f, closeOriginal = true)

            listOf(resized to (8 to 3), cropped to (2 to 4), rotated to (4 to 6)).forEach { (result, size) ->
                assertEquals(format, result.pixelFormat, "format $format")
                assertEquals(size.first, result.width, "format $format")
                assertEquals(size.second, result.height, "format $format")
                result.getPixels().forEach { assertColorApprox(color, it) }
                result.close()
            }
        }
    }

    @Test
    fun threeChannelImageKeepsPixelPositionsWhenCroppedAndRotated() {
        // Однотонная картинка не заметила бы перепутанных каналов или строк
        for (format in listOf(PixelFormat.RGB, PixelFormat.BGR)) {
            val red = (0xFF shl 24) or (255 shl 16)
            val green = (0xFF shl 24) or (255 shl 8)
            val blue = (0xFF shl 24) or 255
            val image = Image(3, 1, format, intArrayOf(red, green, blue))

            val cropped = image.crop(Rect(left = 1, top = 0, right = 3, bottom = 1), closeOriginal = false)
            val rotated = image.rotate(90f, closeOriginal = true)

            assertContentEquals(intArrayOf(green, blue), cropped.getPixels(), "crop $format")
            assertEquals(1, rotated.width, "rotate $format")
            assertEquals(3, rotated.height, "rotate $format")
            val column = rotated.getPixels()
            listOf(red, green, blue).forEachIndexed { i, expected ->
                assertColorApprox(expected, column[i], tolerance = 8)
            }
            cropped.close()
            rotated.close()
        }
    }

    @Test
    fun resizeWithPadLetterboxesAnExtremelyThinImage() {
        // 1000x3 в 300x300: масштаб 0.3, высота 0.9 раньше округлялась до нуля, и ресайз падал
        val color = (0xFF shl 24) or (10 shl 16) or (20 shl 8) or 30
        val image = Image(1000, 3, PixelFormat.ARGB, IntArray(3000) { color })

        val padded = image.resizeWithPad(300, 300)

        assertEquals(300, padded.width)
        assertEquals(300, padded.height)
        // От картинки остаётся одна строка, и лежит она ровно на padY
        assertColorApprox(color, padded[150, padded.info.padY], tolerance = 8)
        padded.close()
    }

    @Test
    fun resizeWithPadRejectsANonPositiveTarget() {
        val image = createSolidImage(4, 4, (0xFF shl 24) or 0x102030)

        assertFailsWith<IllegalArgumentException> { image.resizeWithPad(0, 10, closeOriginal = false) }
        assertFailsWith<IllegalArgumentException> { image.resizeWithPad(10, -1, closeOriginal = false) }

        image.close()
    }

    @Test
    fun resizeDoesNotFlipTheImageVertically() {
        // У CGBitmapContext ось Y направлена вверх, и лишний переворот не виден ни на
        // однотонной картинке, ни на картинке высотой в один пиксель
        val red = (0xFF shl 24) or (255 shl 16)
        val blue = (0xFF shl 24) or 255
        val original = Image(1, 2, PixelFormat.ARGB, intArrayOf(red, blue))

        val resized = original.resize(1, 2)

        val pixels = resized.getPixels()
        assertColorApprox(red, pixels[0], tolerance = 8)
        assertColorApprox(blue, pixels[1], tolerance = 8)

        resized.close()
    }

    @Test
    fun cropRejectsARectOutsideTheImageTheSameWayOnBothPlatforms() {
        // Раньше Android бросал IllegalArgumentException, а iOS - IllegalStateException,
        // и поймать это в общем коде было нечем
        val image = createSolidImage(4, 4, (0xFF shl 24) or (10 shl 16) or (20 shl 8) or 30)

        assertFailsWith<IllegalArgumentException> { image.crop(Rect(0, 0, 5, 4), closeOriginal = false) }
        assertFailsWith<IllegalArgumentException> { image.crop(Rect(-1, 0, 4, 4), closeOriginal = false) }
        assertFailsWith<IllegalArgumentException> { image.crop(Rect(0, 0, 0, 4), closeOriginal = false) }
        assertFailsWith<IllegalArgumentException> { image.crop(Rect(2, 2, 1, 4), closeOriginal = false) }

        image.close()
    }

    @Test
    fun cropDoesNotFlipTheImageVertically() {
        val red = (0xFF shl 24) or (255 shl 16)
        val blue = (0xFF shl 24) or 255
        val original = Image(1, 2, PixelFormat.ARGB, intArrayOf(red, blue))

        val cropped = original.crop(Rect(0, 0, 1, 2))

        val pixels = cropped.getPixels()
        assertColorApprox(red, pixels[0], tolerance = 8)
        assertColorApprox(blue, pixels[1], tolerance = 8)

        cropped.close()
    }

    @Test
    fun rotateByFullTurnRestoresTheOriginalLayout() {
        val red = (0xFF shl 24) or (255 shl 16)
        val blue = (0xFF shl 24) or 255
        val original = Image(2, 1, PixelFormat.ARGB, intArrayOf(red, blue))
        val expected = original.getPixels()

        val rotated = original.rotate(360f, closeOriginal = false)

        assertEquals(2, rotated.width)
        assertEquals(1, rotated.height)
        val pixels = rotated.getPixels()
        assertColorApprox(expected[0], pixels[0], tolerance = 8)
        assertColorApprox(expected[1], pixels[1], tolerance = 8)

        rotated.close()
        original.close()
    }

    @Test
    fun indexedAccessAndGetPixelsAgreeForASemiTransparentImage() {
        // На iOS буфер хранит premultiplied alpha. getPixels делал обратное умножение,
        // а image[x, y] - нет, и один и тот же пиксель читался по-разному.
        val color = (128 shl 24) or (200 shl 16) or (150 shl 8) or 100
        val image = createSolidImage(3, 2, color)

        val pixels = image.getPixels()
        for (y in 0 until 2) {
            for (x in 0 until 3) {
                assertEquals(pixels[y * 3 + x], image[x, y], "pixel at ($x,$y)")
            }
        }

        image.close()
    }

    @Test
    fun indexedAccessOutsideTheImageReturnsZero() {
        val image = createSolidImage(2, 2, (0xFF shl 24) or (10 shl 16) or (20 shl 8) or 30)

        assertEquals(0, image[-1, 0])
        assertEquals(0, image[0, -1])
        assertEquals(0, image[2, 0])
        assertEquals(0, image[0, 2])

        image.close()
    }

    @Test
    fun grayscaleImagePixelsAreReturnedAsPackedArgbWithEqualChannels() {
        // Контракт Image.getPixels: упакованный ARGB, у серого R=G=B. iOS отдавал голое
        // значение канала, и расхождение с Android было не видно, пока проверялся
        // только младший байт.
        val gray = 200
        val color = (0xFF shl 24) or (gray shl 16) or (gray shl 8) or gray
        val image = createSolidImage(3, 2, color).grayscale()

        assertEquals(PixelFormat.Grayscale, image.pixelFormat)
        image.getPixels().forEach { p ->
            val a = (p shr 24) and 0xFF
            val r = (p shr 16) and 0xFF
            val g = (p shr 8) and 0xFF
            val b = p and 0xFF

            assertEquals(255, a, "grayscale pixels must be opaque")
            assertEquals(r, g, "R and G must be equal in a grayscale image")
            assertEquals(g, b, "G and B must be equal in a grayscale image")
            assertTrue(abs(r - gray) <= 2, "expected ~$gray, got $r")
        }

        image.close()
    }

    @Test
    fun grayscaleImageIndexedAccessAgreesWithGetPixels() {
        // Значения выше 127 не должны приходить наружу отрицательными
        val gray = 200
        val color = (0xFF shl 24) or (gray shl 16) or (gray shl 8) or gray
        val image = createSolidImage(3, 2, color).grayscale()

        val pixels = image.getPixels()
        for (y in 0 until 2) {
            for (x in 0 until 3) {
                assertEquals(pixels[y * 3 + x], image[x, y], "pixel at ($x,$y)")
                assertTrue(image[x, y] and 0xFF > 127, "a light gray must not wrap to a negative value")
            }
        }

        image.close()
    }

    @Test
    fun grayscaleImageBuiltFromColoredPixelsKeepsTheLowestByteOnBothPlatforms() {
        // Документированное соглашение: для Grayscale хранится младший байт - тот же,
        // который читает tensorize. Платформы обязаны вести себя одинаково.
        val pixels = IntArray(4) { (0xFF shl 24) or (200 shl 16) or (150 shl 8) or 50 }
        val image = Image(2, 2, PixelFormat.Grayscale, pixels)

        image.getPixels().forEach { p ->
            assertEquals(50, p and 0xFF)
            assertEquals(50, (p shr 8) and 0xFF)
            assertEquals(50, (p shr 16) and 0xFF)
        }

        image.close()
    }

    // --- 8. КОНТРАКТ ЗАКРЫТИЯ ---
    // Обращение к закрытому изображению должно падать одинаково на обеих платформах:
    // на Android Bitmap уже переработан, на iOS буфер пикселей отпущен. Без этой
    // симметрии ошибки жизненного цикла не видны в iOS-тестах и всплывают только на
    // устройстве - именно так пряталась ошибка в resizeWithPad.

    // Преобразование, которое ничего не меняет, платформа может выполнить возвратом ИСХОДНОГО
    // буфера (Bitmap.scale и Bitmap.createBitmap на Android делают именно так). Если при этом
    // закрыть оригинал, закрытым окажется и результат - ровно так пряталась ошибка в
    // resizeWithPad. Здесь проверяется, что результат остаётся читаемым.

    @Test
    fun resizeToTheSameSizeReturnsAUsableImage() {
        val color = (0xFF shl 24) or (10 shl 16) or (20 shl 8) or 30
        val original = createSolidImage(4, 4, color)

        val resized = original.resize(4, 4)

        assertEquals(4, resized.width)
        assertEquals(4, resized.height)
        resized.getPixels().forEach { assertColorApprox(color, it) }

        resized.close()
    }

    @Test
    fun cropOfTheWholeImageReturnsAUsableImage() {
        val color = (0xFF shl 24) or (10 shl 16) or (20 shl 8) or 30
        val original = createSolidImage(4, 4, color)

        val cropped = original.crop(Rect(0, 0, 4, 4))

        assertEquals(4, cropped.width)
        assertEquals(4, cropped.height)
        cropped.getPixels().forEach { assertColorApprox(color, it) }

        cropped.close()
    }

    @Test
    fun rotateByZeroDegreesReturnsAUsableImage() {
        val color = (0xFF shl 24) or (10 shl 16) or (20 shl 8) or 30
        val original = createSolidImage(4, 4, color)

        val rotated = original.rotate(0f)

        assertEquals(4, rotated.width)
        assertEquals(4, rotated.height)
        rotated.getPixels().forEach { assertColorApprox(color, it) }

        rotated.close()
    }

    @Test
    fun readingPixelsFromAClosedImageFails() {
        val image = createSolidImage(2, 2, 0xFF102030.toInt())

        image.close()

        assertFailsWith<IllegalStateException> { image.getPixels() }
        assertFailsWith<IllegalStateException> { image[0, 0] }
    }

    @Test
    fun transformingAClosedImageFails() {
        val image = createSolidImage(4, 4, 0xFF102030.toInt())

        image.close()

        // тип исключения задаёт платформа, важен сам факт отказа
        assertFails { image.resize(2, 2) }
    }

    @Test
    fun closingAnImageTwiceIsSafe() {
        val image = createSolidImage(2, 2, 0xFF102030.toInt())

        image.close()
        image.close()
    }
}
