package dev.kursor.chess.features.game.presentation.ai

import dev.kursor.chess.engine.model.Board
import kotlinx.coroutines.flow.StateFlow

interface AiGameComponent {

    val board: StateFlow<Board>
}