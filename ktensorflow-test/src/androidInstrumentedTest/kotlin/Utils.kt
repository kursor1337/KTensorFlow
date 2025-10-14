import android.content.Context
import dev.kursor.ktensorflow.Interpreter
import dev.kursor.ktensorflow.InterpreterOptions
import dev.kursor.ktensorflow.ModelDesc
import java.io.FileInputStream
import java.nio.channels.FileChannel

fun loadModel(context: Context, fileName: String): ModelDesc {
    val fileDescriptor = context.assets.openFd(fileName)
    val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
    val fileChannel = inputStream.channel
    val byteBuffer = fileChannel.map(
        FileChannel.MapMode.READ_ONLY,
        fileDescriptor.startOffset,
        fileDescriptor.declaredLength
    )
    return ModelDesc.ByteBuffer(byteBuffer)
}

@OptIn(ExperimentalUnsignedTypes::class)
fun loadDataset(context: Context, fileName: String): List<Pair<Byte, Array<UByteArray>>> {
    val dataset = context
        .assets
        .open(fileName)

    val csvDataFrame = CsvDataFrame(dataset)

    return csvDataFrame.extractImages()
}

fun createInterpreter(context: Context, modelFileName: String): Interpreter {
    val modelDesc = loadModel(context, modelFileName)

    val options = InterpreterOptions(
        numThreads = 4,
        useXNNPACK = true
    )

    return Interpreter(modelDesc, options)
}