package dev.kursor.chess.core.model

import dev.kursor.ktensorflow.api.ModelDesc
import ktensorflow.samples.compose_resources.composeapp.generated.resources.Res

class IosModelLoader : ModelLoader {
    override suspend fun loadModel(): ModelDesc {
        return ModelDesc.PathInBundle(Res.getUri("files/chess-ai.tflite").removePrefix("file://"))
    }
}