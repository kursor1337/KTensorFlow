package dev.kursor.chess.engine.api

import dev.kursor.chess.engine.logic.Move
import dev.kursor.chess.engine.logic.state.GameState
import kotlinx.coroutines.flow.StateFlow

interface ChessEngine {
    val state: StateFlow<GameState>
    fun applyMove(move: Move): Boolean
}
