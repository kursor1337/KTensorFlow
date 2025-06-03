package dev.kursor.chess.features.game.domain

import dev.kursor.chess.engine.logic.Move
import dev.kursor.chess.engine.logic.state.GameState

interface ChessAiMoveRepository {

    suspend fun getMove(gameState: GameState): Move
}