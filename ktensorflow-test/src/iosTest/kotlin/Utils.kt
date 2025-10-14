import dev.kursor.ktensorflow.Interpreter
import dev.kursor.ktensorflow.InterpreterOptions
import dev.kursor.ktensorflow.ModelDesc
import kotlinx.cinterop.ExperimentalForeignApi
import platform.Foundation.NSBundle
import platform.Foundation.NSString
import platform.Foundation.NSUTF8StringEncoding
import platform.Foundation.stringWithContentsOfFile

fun loadModel(name: String, extension: String): ModelDesc {
    return NSBundle
        .mainBundle
        .pathForResource(name, extension)!!
        .let(ModelDesc::PathInBundle)
}

@OptIn(ExperimentalUnsignedTypes::class, ExperimentalForeignApi::class)
fun loadDataset(name: String, extension: String): List<Pair<Byte, Array<UByteArray>>> {
    val path = NSBundle
        .mainBundle
        .pathForResource(name, extension)!!

    val text = NSString.stringWithContentsOfFile(
        path = path,
        encoding = NSUTF8StringEncoding,
        error = null
    )!!

    val csvDataFrame = CsvDataFrame(text)

    return csvDataFrame.extractImages()
}

fun createInterpreter(modelFileName: String, modelFileExtension: String): Interpreter {
    val modelDesc = loadModel(modelFileName, modelFileExtension)

    val options = InterpreterOptions(
        numThreads = 4,
        useXNNPACK = true
    )

    return Interpreter(modelDesc, options)
}