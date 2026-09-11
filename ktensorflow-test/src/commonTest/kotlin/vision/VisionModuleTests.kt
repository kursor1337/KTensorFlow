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
import dev.kursor.ktensorflow.vision.tensorizeFloat
import dev.kursor.ktensorflow.vision.toImage
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
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
    fun `ImageTensor validates dimensions upon creation`() {
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
    fun `ImageTensor works correctly with NCHW layout`() {
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
    fun `resizeWithPad scales wide image into square properly`() {
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
    fun `resizeWithPad scales tall image into square properly`() {
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

    // --- 3. ТЕСТЫ НА TENSORIZE И NORMALIZATION ---

    @Test
    fun `tensorizeFloat correctly applies mean and std`() {
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
    fun `toImage correctly denormalizes and clamps out-of-bound pixels`() {
        val tensor = ImageTensor<Float>(1, 1, PixelFormat.RGB)

        // Значения, которые при денормализации выйдут за пределы 0..255
        val pixelFormat = PixelFormat.RGB
        tensor[0, 0, 0, pixelFormat.rIndex] = -1.5f // Ожидаем 0
        tensor[0, 0, 0, pixelFormat.gIndex] = 0.5f  // Ожидаем 127
        tensor[0, 0, 0, pixelFormat.bIndex] = 2.0f  // Ожидаем 255

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
    fun `Rect fromNormalized translates coordinates removing padding`() {
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
    fun `Rect fromNormalized coerces bounds correctly`() {
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
    fun `Rect fromYolo converts center-size coordinates correctly`() {
        val padInfo = PadInfo(100, 100, 100, 100, 0, 0, 1.0f)

        val rect = Rect.fromYolo(cx = 0.5f, cy = 0.5f, w = 0.4f, h = 0.2f, padInfo = padInfo)

        assertEquals(Rect(30, 40, 70, 60), rect)
    }

    @Test
    fun `Rect fromCoco converts top-left-size coordinates correctly`() {
        val padInfo = PadInfo(100, 100, 100, 100, 0, 0, 1.0f)

        val rect = Rect.fromCoco(x = 0.1f, y = 0.2f, w = 0.3f, h = 0.4f, padInfo = padInfo)

        assertEquals(Rect(10, 20, 40, 60), rect)
    }

    @Test
    fun `intersectionOverUnion computes correct ratio for partially overlapping rects`() {
        val a = Rect(0, 0, 10, 10)
        val b = Rect(5, 5, 15, 15)

        assertFloatEquals(25f / 175f, a.intersectionOverUnion(b))
    }

    @Test
    fun `intersectionOverUnion is zero for non overlapping rects`() {
        val a = Rect(0, 0, 10, 10)
        val b = Rect(20, 20, 30, 30)

        assertEquals(0f, a.intersectionOverUnion(b))
    }

    @Test
    fun `scaleForContainer maps rect for crop and fit modes`() {
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
    fun `scaleForContainer returns empty rect for zero-size original container`() {
        val rect = Rect(1, 1, 2, 2).scaleForContainer(0, 0, 100f, 100f)

        assertEquals(Rect(0, 0, 0, 0), rect)
    }

    // --- 5. ТЕСТЫ НА NMS (NON-MAXIMUM SUPPRESSION) ---

    private data class Detection(val rect: Rect, val score: Float, val cls: Int = 0)

    @Test
    fun `nms suppresses heavily overlapping boxes keeping the highest score`() {
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
    fun `nms filters out detections below the score threshold`() {
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
    fun `nms with classSelector suppresses independently per class`() {
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
    fun `nms returns empty list for empty input`() {
        val result = emptyList<Detection>().nms<Detection, Any?>(
            scoreSelector = { it.score },
            boxSelector = { it.rect }
        )

        assertEquals(emptyList(), result)
    }

    // --- 6. ТЕСТЫ НА ImageTensor grayscale/resize/crop (чистая логика над тензором) ---

    @Test
    fun `ImageTensor Float grayscale applies ITU-R 601 luma weights for RGB`() {
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
    fun `ImageTensor Float grayscale multiplies luma by alpha for RGBA`() {
        val tensor = ImageTensor<Float>(1, 1, PixelFormat.ARGB)
        val pf = PixelFormat.ARGB
        tensor[0, 0, pf.rIndex] = 0.5f
        tensor[0, 0, pf.gIndex] = 0.6f
        tensor[0, 0, pf.bIndex] = 0.7f
        tensor[0, 0, pf.aIndex] = 1.0f

        val gray = tensor.grayscale()

        assertFloatEquals(0.5815f, gray[0, 0, 0], tolerance = 1e-3f)
    }

    @Test
    fun `ImageTensor Float grayscale is a no-op for already-grayscale tensors`() {
        val tensor = ImageTensor<Float>(2, 2, PixelFormat.Grayscale)

        assertSame(tensor, tensor.grayscale())
    }

    @Test
    fun `ImageTensor UByte grayscale uses fixed-point luma approximation`() {
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
    fun `ImageTensor Float resize returns same instance when dimensions are unchanged`() {
        val tensor = ImageTensor<Float>(4, 4, PixelFormat.Grayscale)

        assertSame(tensor, tensor.resize(4, 4))
    }

    @Test
    fun `ImageTensor Float resize bilinearly interpolates between source pixels`() {
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
    fun `ImageTensor UByte resize uses nearest neighbor sampling`() {
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

    // --- 7. ТЕСТЫ НА РЕАЛЬНОЙ ПЛАТФОРМЕННОЙ КАРТИНКЕ (Bitmap на Android, CGImage на iOS) ---
    // Все нижеследующие тесты гоняются как на реальном Android-устройстве/эмуляторе
    // (connectedAndroidTest), так и на реальном iOS-симуляторе (iosSimulatorArm64Test) -
    // Image() создаёт настоящий android.graphics.Bitmap / CGBitmapContext, моков нет.

    @Test
    fun `Image crop and ImageTensor crop agree on the same region`() {
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
    fun `Image resize preserves a solid color when scaling up and down`() {
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
    fun `Image rotate 180 degrees preserves dimensions and color for a solid image`() {
        val color = (0xFF shl 24) or (50 shl 16) or (60 shl 8) or 70
        val original = createSolidImage(4, 6, color)

        val rotated = original.rotate(180f)

        assertEquals(4, rotated.width)
        assertEquals(6, rotated.height)
        rotated.getPixels().forEach { assertColorApprox(color, it) }

        rotated.close()
    }

    @Test
    fun `Image grayscale preserves luminance for an already-neutral color`() {
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
}
