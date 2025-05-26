package dev.kursor.chess.features.game.presentation.ai

import dev.kursor.chess.engine.model.Board
import dev.kursor.chess.engine.model.Square
import kotlinx.coroutines.flow.StateFlow

interface AiGameComponent {

    val board: StateFlow<Board>
}