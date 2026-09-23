import android.app.Application
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import dev.kursor.ktensorflow.ExperimentalKTensorFlowApi
import dev.kursor.ktensorflow.ModelDesc
import dev.kursor.ktensorflow.TensorFlowException
import dev.kursor.ktensorflow.compose.ComposeUri
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.test.assertFailsWith
import kotlin.test.assertSame
import kotlin.test.assertTrue
import dev.kursor.ktensorflow.compose.ContextInitializer as ComposeContextInitializer
import dev.kursor.ktensorflow.moko.ContextInitializer as MokoContextInitializer

/**
 * ktensorflow-compose и ktensorflow-moko получают Context на Android через androidx.startup,
 * без единой строки кода со стороны приложения. Это самая молчаливая точка отказа модулей:
 * стоит приложению убрать InitializationProvider из манифеста, и загрузка модели упадёт уже
 * в рантайме. Здесь проверяется, что провайдер действительно отработал.
 *
 * Сквозная загрузка модели из compose-resources и moko-resources отсюда не проверяется:
 * KMP-плагин AGP не пакует ни src/androidDeviceTest/assets, ни src/androidDeviceTest/res
 * в тестовый APK, поэтому положить туда ресурс нечем.
 */
@OptIn(ExperimentalKTensorFlowApi::class)
@RunWith(AndroidJUnit4::class)
class ResourceModulesTest {

    private val context = InstrumentationRegistry.getInstrumentation().targetContext

    @Test
    fun composeUriResolvesAgainstAssetsWithoutAnyManualInitialization() {
        // Никакой ручной инициализации: если androidx.startup не отработал, упадёт
        // IllegalStateException про отсутствующий Context, а не про отсутствующий ассет
        val failure = assertFailsWith<TensorFlowException> {
            ModelDesc.ComposeUri("file:///android_asset/there-is-no-such-model.tflite")
        }

        assertTrue(
            failure.message.orEmpty().contains("there-is-no-such-model.tflite"),
            "the asset path must survive prefix stripping, was: ${failure.message}"
        )
    }

    @Test
    fun composeUriAcceptsAPlainAssetPathWithoutTheFileScheme() {
        val failure = assertFailsWith<TensorFlowException> {
            ModelDesc.ComposeUri("there-is-no-such-model.tflite")
        }

        assertTrue(
            failure.message.orEmpty().contains("there-is-no-such-model.tflite"),
            "a plain path must be passed through unchanged, was: ${failure.message}"
        )
    }

    @Test
    fun composeContextInitializerStoresTheApplicationContext() {
        val initializer = ComposeContextInitializer()

        val stored = initializer.create(context)

        assertSame(context.applicationContext, stored)
        assertTrue(stored is Application, "the stored context must be the application, not an activity")
        assertTrue(initializer.dependencies().isEmpty())
    }

    @Test
    fun mokoContextInitializerStoresTheApplicationContext() {
        val initializer = MokoContextInitializer()

        val stored = initializer.create(context)

        assertSame(context.applicationContext, stored)
        assertTrue(stored is Application, "the stored context must be the application, not an activity")
        assertTrue(initializer.dependencies().isEmpty())
    }
}
