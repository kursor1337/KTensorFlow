package dev.kursor.chess.core.model

import android.content.Context
import dev.kursor.ktensorflow.api.ModelDesc
import ktensorflow.samples.compose_resources.composeapp.generated.resources.Res
import java.io.File

class AndroidModelLoader(
    private val context: Context
) : ModelLoader {
    override suspend fun loadModel(): ModelDesc {
        val bytes = Res.readBytes("files/chess-ai.tflite")
        val tmpFile = File.createTempFile("prefix", "suffix", context.cacheDir)
        tmpFile.writeBytes(bytes)
        return ModelDesc.File(tmpFile)
    }
}