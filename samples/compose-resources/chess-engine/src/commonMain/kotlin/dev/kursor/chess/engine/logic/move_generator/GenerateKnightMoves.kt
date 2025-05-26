package dev.kursor.chess.engine.logic.move_generator

import dev.kursor.chess.engine.logic.Move
import dev.kursor.chess.engine.model.Board
import dev.kursor.chess.engine.model.Color
import dev.kursor.chess.engine.model.Square

fun generateKnightMoves(board: Board, square: Square, color: Color): List<Move> {
    val moves = mutableListOf<Move>()

    val offsets = listOf(
        Pair(-2, -1), Pair(-1, -2), Pair(1, -2), Pair(2, -1),
        Pair(2, 1), Pair(1, 2), Pair(-1, 2), Pair(-2, 1)
    )

    for ((dx, dy) in offsets) {
        val file = square.file + dx
        val rank = square.rank + dy
        if (file !in 0..7 || rank !in 0..7) continue

        val target = Square(file, rank)
        val targetPiece = board[target]
        if (targetPiece == null || targetPiece.color != color) {
            moves.add(Move(square, target))
        }
    }

    return moves
}
