package dev.kursor.chess.engine.logic.move_generator

import dev.kursor.chess.engine.logic.Move
import dev.kursor.chess.engine.model.Board
import dev.kursor.chess.engine.model.Color
import dev.kursor.chess.engine.model.PieceType
import dev.kursor.chess.engine.model.Square

fun generatePawnMoves(
    board: Board,
    square: Square,
    color: Color,
    enPassantTarget: Square?
): List<Move> {
    val moves = mutableListOf<Move>()
    val dir = if (color == Color.White) 1 else -1
    val startRank = if (color == Color.White) 1 else 6
    val promotionRank = if (color == Color.White) 7 else 0

    val forward = Square(square.file, square.rank + dir)
    if (forward.rank in 0..7 && board[forward] == null) {
        // Promotion
        if (forward.rank == promotionRank) {
            for (promo in listOf(PieceType.Queen, PieceType.Rook, PieceType.Bishop, PieceType.Knight, null)) {
                moves.add(Move(square, forward, promo))
            }
        } else {
            moves.add(Move(square, forward))
        }

        // Double move
        if (square.rank == startRank) {
            val twoForward = Square(square.file, square.rank + 2 * dir)
            if (board[twoForward] == null) {
                moves.add(Move(square, twoForward))
            }
        }
    }

    // Diagonal captures
    for (dx in listOf(-1, 1)) {
        val targetFile = square.file + dx
        val targetRank = square.rank + dir
        if (targetFile !in 0..7 || targetRank !in 0..7) continue

        val target = Square(targetFile, targetRank)
        val targetPiece = board[target]
        if (targetPiece != null && targetPiece.color != color) {
            // Promotion by capture
            if (target.rank == promotionRank) {
                for (promo in listOf(PieceType.Queen, PieceType.Rook, PieceType.Bishop, PieceType.Knight, null)) {
                    moves.add(Move(square, target, promo))
                }
            } else {
                moves.add(Move(square, target))
            }
        }

        // En passant
        if (enPassantTarget != null && enPassantTarget == target) {
            moves.add(Move(square, enPassantTarget))
        }
    }

    return moves
}
