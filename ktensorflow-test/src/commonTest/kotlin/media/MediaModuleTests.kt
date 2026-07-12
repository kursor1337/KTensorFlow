package media

import dev.kursor.ktensorflow.vision.Image
import dev.kursor.ktensorflow.vision.ImageTensor
import dev.kursor.ktensorflow.vision.ImageTensorLayout
import dev.kursor.ktensorflow.vision.Normalization
import dev.kursor.ktensorflow.vision.PadInfo
import dev.kursor.ktensorflow.vision.PixelFormat
import dev.kursor.ktensorflow.vision.Rect
import dev.kursor.ktensorflow.vision.resizeWithPad
import dev.kursor.ktensorflow.vision.tensorizeFloat
import dev.kursor.ktensorflow.vision.toImage
import kotlin.test.Test
import kotlin.test.assertEquals

class MediaModuleTests {

    // Вспомогательный метод для генерации однотонной залитой картинки
    private fun createSolidImage(width: Int, height: Int, colorArgb: Int): Image {
        val pixels = IntArray(width * height) { colorArgb }
        return Image(width, height, PixelFormat.ARGB, pixels)
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
}