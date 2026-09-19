import dev.icerock.moko.resources.AssetResource
import dev.icerock.moko.resources.FileResource
import dev.kursor.ktensorflow.Interpreter
import dev.kursor.ktensorflow.InterpreterOptions
import dev.kursor.ktensorflow.ModelDesc
import platform.Foundation.NSBundle
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import dev.kursor.ktensorflow.moko.AssetResource as assetResourceModelDesc
import dev.kursor.ktensorflow.moko.FileResource as fileResourceModelDesc

/**
 * ktensorflow-moko переводит ресурсы moko-resources в [ModelDesc].
 *
 * Ресурсы создаются напрямую через публичные конструкторы moko, без генерации кода плагином,
 * поэтому модуль проверяется на настоящих объектах, а не на подменах. На iOS moko различает
 * два вида ресурсов: [FileResource] ищет файл в подпапке `files` бандла, [AssetResource] -
 * в корне, поэтому модель лежит в обоих местах.
 */
class MokoResourcesTest {

    private val modelName = "mnist"
    private val modelExtension = "tflite"

    private fun options() = InterpreterOptions(
        numThreads = 1,
        useXNNPACK = false,
        delegates = emptyList()
    )

    private fun assertModelIsLoadable(desc: ModelDesc) {
        val interpreter = Interpreter(modelDesc = desc, options = options())
        val meta = interpreter.getModelMeta()

        assertTrue(meta.inputData.isNotEmpty(), "model must expose inputs")
        assertTrue(meta.outputData.isNotEmpty(), "model must expose outputs")
    }

    @Test
    fun fileResourceResolvesToThePathInsideTheBundleFilesDirectory() {
        val resource = FileResource(fileName = modelName, extension = modelExtension)

        val desc = ModelDesc.fileResourceModelDesc(resource)

        assertEquals(ModelDesc.PathInBundle(resource.path), desc)
        assertTrue(resource.path.endsWith("files/$modelName.$modelExtension"), "unexpected path: ${resource.path}")
    }

    @Test
    fun modelDescFromFileResourceLoadsARealModel() {
        val resource = FileResource(fileName = modelName, extension = modelExtension)

        assertModelIsLoadable(ModelDesc.fileResourceModelDesc(resource))
    }

    @Test
    fun assetResourceResolvesToThePathInTheBundleRoot() {
        val resource = AssetResource(
            originalPath = "$modelName.$modelExtension",
            fileName = modelName,
            extension = modelExtension
        )

        val desc = ModelDesc.assetResourceModelDesc(resource)

        assertEquals(ModelDesc.PathInBundle(resource.path), desc)
        assertEquals(
            NSBundle.mainBundle.pathForResource(modelName, modelExtension),
            resource.path
        )
    }

    @Test
    fun modelDescFromAssetResourceLoadsARealModel() {
        val resource = AssetResource(
            originalPath = "$modelName.$modelExtension",
            fileName = modelName,
            extension = modelExtension
        )

        assertModelIsLoadable(ModelDesc.assetResourceModelDesc(resource))
    }
}
