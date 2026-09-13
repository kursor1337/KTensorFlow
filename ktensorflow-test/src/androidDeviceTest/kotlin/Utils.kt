import android.content.Context
import dev.kursor.ktensorflow.Delegate
import dev.kursor.ktensorflow.Interpreter
import dev.kursor.ktensorflow.InterpreterOptions
import dev.kursor.ktensorflow.ModelDesc
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

// Модель и датасет лежат в src/androidDeviceTest/resources и попадают в тестовый APK как
// java-ресурсы: новый KMP-плагин AGP не подключает src/androidDeviceTest/assets к device-тесту,
// поэтому AssetManager их не видит.
private fun openTestResource(fileName: String): InputStream =
    checkNotNull(Utils::class.java.classLoader?.getResourceAsStream(fileName)) {
        "Test resource '$fileName' not found on the test classpath"
    }

private object Utils

fun loadModel(context: Context, fileName: String): ModelDesc {
    val bytes = openTestResource(fileName).use { it.readBytes() }
    val byteBuffer = ByteBuffer
        .allocateDirect(bytes.size)
        .order(ByteOrder.nativeOrder())
        .put(bytes)
        .apply { rewind() }
    return ModelDesc.ByteBuffer(byteBuffer)
}

@OptIn(ExperimentalUnsignedTypes::class)
fun loadDataset(context: Context, fileName: String): List<Pair<Byte, Array<UByteArray>>> {
    val csvDataFrame = CsvDataFrame(openTestResource(fileName))

    return csvDataFrame.extractImages()
}

fun createInterpreter(
    context: Context,
    modelFileName: String,
    delegate: Delegate?
): Interpreter {
    val modelDesc = loadModel(context, modelFileName)

    val options = InterpreterOptions(
        numThreads = 4,
        useXNNPACK = true,
        delegates = delegate?.let(::listOf).orEmpty()
    )

    return Interpreter(modelDesc, options)
}
