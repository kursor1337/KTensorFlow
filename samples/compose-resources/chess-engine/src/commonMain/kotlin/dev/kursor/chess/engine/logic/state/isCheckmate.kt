package dev.kursor.chess.engine.logic.state

import dev.kursor.chess.engine.logic.applyMove
import dev.kursor.chess.engine.logic.move_generator.generateLegalMoves

fun GameState.isCheckmate(): Boolean {
    if (!isCheck(board, activeColor)) return false
    
    // Check if there are any legal moves left to escape check
    val legalMoves = generateLegalMoves()
    return legalMoves.none { move ->
        val newState = applyMove(move)
        !isCheck(newState.board, activeColor)
    }
}
