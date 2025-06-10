package dev.kursor.chess.core.model

import dev.kursor.ktensorflow.ModelDesc

interface ModelLoader {
    suspend fun loadModel(): ModelDesc
}