import dev.kursor.ktensorflow.vision.Image
import dev.kursor.ktensorflow.vision.PixelFormat
import dev.kursor.ktensorflow.vision.grayscale
import dev.kursor.ktensorflow.vision.withCGImage
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.UByteVar
import kotlinx.cinterop.alloc
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.ptr
import kotlinx.cinterop.reinterpret
import kotlinx.cinterop.set
import kotlinx.cinterop.value
import platform.CoreVideo.CVPixelBufferCreate
import platform.CoreVideo.CVPixelBufferGetBaseAddress
import platform.CoreVideo.CVPixelBufferGetBaseAddressOfPlane
import platform.CoreVideo.CVPixelBufferGetBytesPerRow
import platform.CoreVideo.CVPixelBufferGetBytesPerRowOfPlane
import platform.CoreVideo.CVPixelBufferLockBaseAddress
import platform.CoreVideo.CVPixelBufferRef
import platform.CoreVideo.CVPixelBufferRefVar
import platform.CoreVideo.CVPixelBufferRelease
import platform.CoreVideo.CVPixelBufferUnlockBaseAddress
import platform.CoreVideo.kCVPixelFormatType_32BGRA
import platform.CoreVideo.kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
import platform.CoreVideo.kCVPixelFormatType_OneComponent32Float
import platform.CoreVideo.kCVReturnSuccess
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Фабрики Image из CGImage и CVPixelBuffer - единственный способ завести в библиотеку кадр
 * камеры или картинку из UIKit на iOS, поэтому проверяется каждый путь конвертации: все
 * раскладки каналов, серое, выровненные строки CVPixelBuffer, YUV-кадр и неподдерживаемый формат.
 */
@OptIn(ExperimentalForeignApi::class)
class NativeImageTest {

    private val semiTransparent = (128 shl 24) or (200 shl 16) or (100 shl 8) or 50

    // Уникальный непрозрачный цвет на каждый пиксель: перепутанные каналы или строки
    // сразу дают другой цвет
    private fun gradient(width: Int, height: Int) = IntArray(width * height) { i ->
        (0xFF shl 24) or ((i * 7 % 256) shl 16) or ((i * 13 % 256) shl 8) or (i * 29 % 256)
    }

    private fun assertColorApprox(expected: Int, actual: Int, tolerance: Int, message: String) {
        for (shift in intArrayOf(24, 16, 8, 0)) {
            val e = (expected shr shift) and 0xFF
            val a = (actual shr shift) and 0xFF
            assertTrue(
                abs(e - a) <= tolerance,
                "$message: channel at shift $shift expected=$e actual=$a"
            )
        }
    }

    private fun pixelBuffer(
        width: Int,
        height: Int,
        format: UInt,
        fill: (CVPixelBufferRef) -> Unit
    ): CVPixelBufferRef = memScoped {
        val out = alloc<CVPixelBufferRefVar>()
        val status = CVPixelBufferCreate(null, width.toULong(), height.toULong(), format, null, out.ptr)
        check(status == kCVReturnSuccess) { "CVPixelBufferCreate failed: $status" }
        val buffer = out.value!!
        CVPixelBufferLockBaseAddress(buffer, 0u)
        fill(buffer)
        CVPixelBufferUnlockBaseAddress(buffer, 0u)
        buffer
    }

    /** 32BGRA-буфер; [pixels] - premultiplied ARGB, как их хранит CoreVideo. */
    private fun bgraBuffer(width: Int, height: Int, pixels: IntArray): CVPixelBufferRef =
        pixelBuffer(width, height, kCVPixelFormatType_32BGRA) { buffer ->
            val rowBytes = CVPixelBufferGetBytesPerRow(buffer).toInt()
            val bytes = CVPixelBufferGetBaseAddress(buffer)!!.reinterpret<UByteVar>()
            for (y in 0 until height) {
                for (x in 0 until width) {
                    val p = pixels[y * width + x]
                    val o = y * rowBytes + x * 4
                    bytes[o] = (p and 0xFF).toUByte()
                    bytes[o + 1] = ((p shr 8) and 0xFF).toUByte()
                    bytes[o + 2] = ((p shr 16) and 0xFF).toUByte()
                    bytes[o + 3] = ((p shr 24) and 0xFF).toUByte()
                }
            }
        }

    // --- CGImage ---

    @Test
    fun cgImageKeepsEveryPixelInEveryFourChannelLayout() {
        val pixels = gradient(7, 5)
        val source = Image(7, 5, PixelFormat.BGRA, pixels)

        for (format in listOf(PixelFormat.BGRA, PixelFormat.RGBA, PixelFormat.ARGB, PixelFormat.ABGR)) {
            // Изображение создаётся внутри withCGImage и читается после освобождения CGImage:
            // так заодно проверяется, что пиксели скопированы, а не взяты по ссылке
            val image = source.withCGImage { Image(it, format) }

            assertEquals(format, image.pixelFormat)
            assertEquals(7, image.width)
            assertEquals(5, image.height)
            assertContentEquals(pixels, image.getPixels(), "layout $format")
            image.close()
        }
        source.close()
    }

    @Test
    fun cgImageToThreeChannelFormatsKeepsTheColor() {
        // CoreGraphics не умеет рисовать в 24-битный буфер, для RGB работает отдельный путь
        val pixels = gradient(6, 4)
        val source = Image(6, 4, PixelFormat.ARGB, pixels)

        for (format in listOf(PixelFormat.RGB, PixelFormat.BGR)) {
            val image = source.withCGImage { Image(it, format) }

            assertEquals(format, image.pixelFormat)
            assertContentEquals(pixels, image.getPixels(), "layout $format")
            image.close()
        }
        source.close()
    }

    @Test
    fun cgImageToGrayscaleMatchesGrayscaleOfTheSameImage() {
        // Серый контекст CoreGraphics считает яркость по своей формуле; фабрика обязана давать
        // тот же Rec.601, что grayscale() и Android
        val source = Image(8, 8, PixelFormat.BGRA, gradient(8, 8))

        val fromCgImage = source.withCGImage { Image(it, PixelFormat.Grayscale) }
        val expected = source.grayscale(closeOriginal = true)

        assertEquals(PixelFormat.Grayscale, fromCgImage.pixelFormat)
        assertContentEquals(expected.getPixels(), fromCgImage.getPixels())
        fromCgImage.close()
        expected.close()
    }

    @Test
    fun grayCgImageIsCopiedWithoutLoss() {
        // Серое проходит через RGB и яркость Rec.601: каждый из 256 уровней обязан остаться собой,
        // иначе округление или управление цветом сдвигали бы уровни (раньше 128 -> 127)
        val levels = IntArray(256) { it }
        val source = Image(16, 16, PixelFormat.Grayscale, levels)

        val image = source.withCGImage { Image(it, PixelFormat.Grayscale) }

        assertContentEquals(levels, image.getPixels().map { it and 0xFF }.toIntArray())
        image.close()
        source.close()
    }

    @Test
    fun semiTransparentPixelSurvivesTheCgImageRoundTrip() {
        val source = Image(1, 1, PixelFormat.ARGB, intArrayOf(semiTransparent))

        val image = source.withCGImage { Image(it) }

        assertColorApprox(semiTransparent, image[0, 0], tolerance = 2, message = "semi-transparent")
        image.close()
        source.close()
    }

    @Test
    fun threeChannelImagesCanBeDrawnThroughCGImage() {
        // CoreGraphics не знает 3-канальных раскладок, и withCGImage для RGB и BGR падал
        val pixels = gradient(6, 4)

        for (format in listOf(PixelFormat.RGB, PixelFormat.BGR)) {
            val source = Image(6, 4, format, pixels)

            val image = source.withCGImage { Image(it) }

            assertContentEquals(pixels, image.getPixels(), "layout $format")
            image.close()
            source.close()
        }
    }

    // --- CVPixelBuffer ---

    @Test
    fun bgraPixelBufferWithPaddedRowsIsCopiedExactly() {
        val width = 5
        val height = 3
        val pixels = gradient(width, height)
        val buffer = bgraBuffer(width, height, pixels)
        // Без выравнивания строк тест не проверял бы построчное копирование
        assertTrue(CVPixelBufferGetBytesPerRow(buffer).toInt() > width * 4, "rows must be padded")

        val image = Image(buffer)
        CVPixelBufferRelease(buffer)

        // Буфер уже освобождён: изображение живёт на своей копии пикселей
        assertEquals(PixelFormat.BGRA, image.pixelFormat)
        assertEquals(width, image.width)
        assertEquals(height, image.height)
        assertContentEquals(pixels, image.getPixels())
        image.close()
    }

    @Test
    fun bgraPixelBufferGivesTheSameColorsInEveryFormat() {
        // BGRA копируется напрямую, остальные форматы идут через VideoToolbox: оба пути
        // обязаны одинаково понимать premultiplied alpha
        val pixels = gradient(4, 2).also { it[3] = (128 shl 24) or (100 shl 16) or (50 shl 8) or 25 }
        val buffer = bgraBuffer(4, 2, pixels)

        val direct = Image(buffer)
        val expected = direct.getPixels()
        for (format in listOf(PixelFormat.ARGB, PixelFormat.RGBA, PixelFormat.ABGR)) {
            val converted = Image(buffer, format)
            assertEquals(format, converted.pixelFormat)
            converted.getPixels().forEachIndexed { i, pixel ->
                assertColorApprox(expected[i], pixel, tolerance = 1, message = "$format pixel $i")
            }
            converted.close()
        }
        // (a=128, 100) premultiplied - это прямой цвет 200
        assertColorApprox(
            (128 shl 24) or (200 shl 16) or (100 shl 8) or 50,
            expected[3],
            tolerance = 2,
            message = "premultiplied alpha"
        )

        val gray = Image(buffer, PixelFormat.Grayscale)
        val expectedGray = direct.grayscale(closeOriginal = true)
        assertContentEquals(expectedGray.getPixels(), gray.getPixels())

        gray.close()
        expectedGray.close()
        CVPixelBufferRelease(buffer)
    }

    @Test
    fun yuvCameraBufferIsConverted() {
        // Формат камеры по умолчанию - 420 biplanar; нейтральная цветность (128) даёт серый
        val buffer = pixelBuffer(4, 2, kCVPixelFormatType_420YpCbCr8BiPlanarFullRange) { b ->
            val luma = CVPixelBufferGetBaseAddressOfPlane(b, 0u)!!.reinterpret<UByteVar>()
            val lumaRow = CVPixelBufferGetBytesPerRowOfPlane(b, 0u).toInt()
            for (y in 0 until 2) for (x in 0 until 4) luma[y * lumaRow + x] = 100u
            val chroma = CVPixelBufferGetBaseAddressOfPlane(b, 1u)!!.reinterpret<UByteVar>()
            for (i in 0 until 4) chroma[i] = 128u
        }

        val image = Image(buffer)
        val gray = Image(buffer, PixelFormat.Grayscale)
        CVPixelBufferRelease(buffer)

        assertEquals(4, image.width)
        assertEquals(2, image.height)
        val expected = (0xFF shl 24) or (100 shl 16) or (100 shl 8) or 100
        image.getPixels().forEach { assertColorApprox(expected, it, tolerance = 1, message = "yuv") }
        gray.getPixels().forEach { assertColorApprox(expected, it, tolerance = 1, message = "yuv gray") }
        image.close()
        gray.close()
    }

    @Test
    fun unsupportedPixelBufferFormatFailsWithItsName() {
        val buffer = pixelBuffer(2, 2, kCVPixelFormatType_OneComponent32Float) { }

        val error = assertFailsWith<IllegalArgumentException> { Image(buffer) }

        assertTrue("L00f" in error.message.orEmpty(), "message names the format: ${error.message}")
        CVPixelBufferRelease(buffer)
    }
}
