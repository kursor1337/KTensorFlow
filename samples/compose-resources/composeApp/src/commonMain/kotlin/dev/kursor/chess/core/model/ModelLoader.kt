package dev.kursor.chess.core.model

import dev.kursor.ktensorflow.api.ModelDesc

interface ModelLoader {
    suspend fun loadModel(): ModelDesc
}