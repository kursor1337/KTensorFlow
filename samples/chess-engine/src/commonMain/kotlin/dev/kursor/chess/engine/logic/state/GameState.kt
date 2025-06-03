package dev.kursor.chess.engine.logic.state

import dev.kursor.chess.engine.model.Board
import dev.kursor.chess.engine.model.Color
import dev.kursor.chess.engine.model.Square

data class GameState(
    val board: Board = Board(),
    val activeColor: Color = Color.White,
    val castlingRights: CastlingRights = CastlingRights(),
    val enPassantTarget: Square? = null,
    val halfMoveClock: Int = 0,
    val fullMoveNumber: Int = 1
)

data class CastlingRights(
    val whiteKingSide: Boolean = true,
    val whiteQueenSide: Boolean = true,
    val blackKingSide: Boolean = true,
    val blackQueenSide: Boolean = true
) {
    companion object {
        val Empty = CastlingRights(
            whiteKingSide = false,
            whiteQueenSide = false,
            blackKingSide = false,
            blackQueenSide = false
        )
    }
}
