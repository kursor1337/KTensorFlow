import dev.kursor.ktensorflow.ExperimentalKTensorFlowApi
import dev.kursor.ktensorflow.Interpreter
import dev.kursor.ktensorflow.InterpreterOptions
import dev.kursor.ktensorflow.ModelDesc
import dev.kursor.ktensorflow.compose.ComposeUri
import platform.Foundation.NSBundle
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * ktensorflow-compose переводит URI из compose-resources в [ModelDesc].
 * На iOS `Res.getUri()` отдаёт путь со схемой `file://`, которую модуль обязан снять.
 * Проверяется не только преобразование строки, но и то, что по полученному описанию
 * реально загружается модель.
 */
@OptIn(ExperimentalKTensorFlowApi::class)
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
