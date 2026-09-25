import dev.kursor.ktensorflow.ExperimentalKTensorFlowApi
import dev.kursor.ktensorflow.Interpreter
import dev.kursor.ktensorflow.InterpreterOptions
import dev.kursor.ktensorflow.ModelDesc
import dev.kursor.ktensorflow.compose.ComposeUri
import kotlinx.cinterop.ExperimentalForeignApi
import platform.Foundation.NSBundle
import platform.Foundation.NSFileManager
import platform.Foundation.NSTemporaryDirectory
import platform.Foundation.NSURL
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * ktensorflow-compose переводит URI из compose-resources в [ModelDesc].
 * На iOS `Res.getUri()` отдаёт путь со схемой `file://`, которую модуль обязан снять.
 * Проверяется не только преобразование строки, но и то, что по полученному описанию
 * реально загружается модель.
 */
@OptIn(ExperimentalKTensorFlowApi::class, ExperimentalForeignApi::class)
class ComposeResourcesTest {

    private fun modelPathInBundle(): String =
        requireNotNull(NSBundle.mainBundle.pathForResource("mnist", "tflite")) {
            "mnist.tflite is missing from the test bundle"
        }

    @Test
    fun composeUriStripsTheFileScheme() {
        val path = modelPathInBundle()

        assertEquals(ModelDesc.PathInBundle(path), ModelDesc.ComposeUri("file://$path"))
    }

    @Test
    fun composeUriDecodesAPercentEncodedPath() {
        // Res.getUri на iOS - это NSURL.fileURLWithPath(...).toString(): пробел приходит как %20,
        // кириллица - как %D0%9C... Раньше такой путь уходил в TensorFlow Lite как есть, и модель
        // из приложения с пробелом в имени не загружалась вовсе
        for (directoryName in listOf("My App", "Модели")) {
            val directory = NSTemporaryDirectory() + directoryName
            NSFileManager.defaultManager.createDirectoryAtPath(directory, true, null, null)
            val path = "$directory/mnist.tflite"
            NSFileManager.defaultManager.removeItemAtPath(path, null)
            NSFileManager.defaultManager.copyItemAtPath(modelPathInBundle(), path, null)
            val uri = NSURL.fileURLWithPath(path).toString()

            val desc = ModelDesc.ComposeUri(uri)

            assertEquals(ModelDesc.PathInBundle(path), desc, "uri $uri")
            Interpreter(desc, InterpreterOptions()).close()
        }
    }

    @Test
    fun composeUriKeepsAPlainPathUnchanged() {
        val path = modelPathInBundle()

        assertEquals(ModelDesc.PathInBundle(path), ModelDesc.ComposeUri(path))
    }

    @Test
    fun modelDescFromComposeUriLoadsARealModel() {
        val desc = ModelDesc.ComposeUri("file://${modelPathInBundle()}")

        val interpreter = Interpreter(
            modelDesc = desc,
            options = InterpreterOptions(numThreads = 1, useXNNPACK = false, delegates = emptyList())
        )
        val meta = interpreter.getModelMeta()

        assertTrue(meta.inputData.isNotEmpty(), "model loaded via ComposeUri must expose inputs")
        assertTrue(meta.outputData.isNotEmpty(), "model loaded via ComposeUri must expose outputs")
    }
}
