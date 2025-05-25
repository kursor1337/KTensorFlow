package dev.kursor.chess.engine.logic.state

import dev.kursor.chess.engine.logic.move_generator.generateLegalMoves

fun GameState.isStalemate(): Boolean {
    if (isCheck(board, activeColor)) return false
    
    // Check if there are no legal moves left
    val legalMoves = generateLegalMoves()
    return legalMoves.isEmpty()
}
