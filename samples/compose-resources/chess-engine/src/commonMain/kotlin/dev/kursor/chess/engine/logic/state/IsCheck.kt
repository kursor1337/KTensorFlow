package dev.kursor.chess.engine.logic.state

import dev.kursor.chess.engine.logic.move_generator.generatePseudoLegalMoves
import dev.kursor.chess.engine.model.Board
import dev.kursor.chess.engine.model.Color
import dev.kursor.chess.engine.model.PieceType
import dev.kursor.chess.engine.model.Square

fun isCheck(board: Board, color: Color): Boolean {
    val kingSquare = board.findKing(color) ?: return true
    val threats = generatePseudoLegalMoves(board, color.opposing, CastlingRights.Empty, null)
    return threats.any { it.to == kingSquare }
}

fun Board.findKing(color: Color): Square? {
    for (rank in 0..7) {
        for (file in 0..7) {
            val sq = Square(file, rank)
            val piece = get(sq)
            if (piece?.type == PieceType.King && piece.color == color) {
                return sq
            }
        }
    }
    return null
}
