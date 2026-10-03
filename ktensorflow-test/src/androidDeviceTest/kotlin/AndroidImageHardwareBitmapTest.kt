import android.graphics.Bitmap
import android.os.Build
import androidx.test.ext.junit.runners.AndroidJUnit4
import dev.kursor.ktensorflow.vision.AndroidImage
import dev.kursor.ktensorflow.vision.PixelFormat
import dev.kursor.ktensorflow.vision.grayscale
import dev.kursor.ktensorflow.vision.resize
import dev.kursor.ktensorflow.vision.resizeWithPad
import dev.kursor.ktensorflow.vision.tensorizeFloat
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * ImageDecoder на Android 9+ по умолчанию отдаёт HARDWARE-битмапы: их пиксели лежат в памяти GPU,
 * и ни getPixels, ни программный Canvas с ними не работают. Раньше на таком битмапе падали
 * tensorize, resize, resizeWithPad и grayscale - то есть на любом фото из галереи.
 *
 * Имя класса начинается с AndroidImage намеренно: раннер идёт по алфавиту, и так тест выполняется
 * раньше тестов GPU-делегата. На эмуляторе (gfxstream) любое создание CompatibilityList в процессе -
 * даже без close - ломает GL-состояние, и последующее чтение HARDWARE-битмапа падает с SIGSEGV
 * в RenderThread. Это воспроизводится и без библиотеки, одним CompatibilityList через рефлексию.
 */
@RunWith(AndroidJUnit4::class)
class AndroidImageHardwareBitmapTest {

    private val color = (0xFF shl 24) or (10 shl 16) or (120 shl 8) or 230

    private fun hardwareBitmap(): Bitmap {
        val software = Bitmap.createBitmap(8, 4, Bitmap.Config.ARGB_8888)
        software.eraseColor(color)
        return software.copy(Bitmap.Config.HARDWARE, false).also { software.recycle() }
    }

    @Test
    fun everyOperationWorksOnAHardwareBitmap() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return

        val image = AndroidImage(hardwareBitmap(), PixelFormat.ARGB)

        assertEquals(Bitmap.Config.ARGB_8888, image.platformImage.config)
        assertEquals(color, image[3, 2])
        assertEquals(10f, image.tensorizeFloat()[0, 0, PixelFormat.ARGB.rIndex])
        image.resize(4, 2, closeOriginal = false).close()
        image.grayscale(closeOriginal = false).close()
        image.resizeWithPad(16, 16, closeOriginal = false).close()
        image.close()
    }

    @Test
    fun theHardwareOriginalIsReleasedBecauseTheImageOwnsIt() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
        val hardware = hardwareBitmap()

        val image = AndroidImage(hardware, PixelFormat.ARGB)

        assertTrue(hardware.isRecycled, "the GPU copy must not outlive the conversion")
        image.close()
        assertTrue(image.platformImage.isRecycled)
    }

    @Test
    fun aSoftwareBitmapIsWrappedWithoutCopying() {
        val software = Bitmap.createBitmap(2, 2, Bitmap.Config.ARGB_8888)

        val image = AndroidImage(software, PixelFormat.ARGB)

        assertTrue(image.platformImage === software)
        image.close()
    }
}
