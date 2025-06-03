package dev.kursor.chess.engine.api

import dev.kursor.chess.engine.logic.Move
import dev.kursor.chess.engine.logic.applyMove
import dev.kursor.chess.engine.logic.move_generator.generateLegalMoves
import dev.kursor.chess.engine.logic.state.GameState
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.update

class BasicChessEngine(initialState: GameState = GameState()) : ChessEngine {

    override val state = MutableStateFlow(initialState)

    override fun applyMove(move: Move): Boolean {
        if (!state.value.generateLegalMoves().contains(move)) return false
        state.update { it.applyMove(move) }
        return true
    }
}
