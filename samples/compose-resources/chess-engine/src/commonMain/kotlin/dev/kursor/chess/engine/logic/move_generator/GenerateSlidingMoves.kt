package dev.kursor.chess.engine.logic.move_generator

import dev.kursor.chess.engine.logic.Move
import dev.kursor.chess.engine.model.Board
import dev.kursor.chess.engine.model.Color
import dev.kursor.chess.engine.model.Square

@Suppress("NoMultipleSpaces")
val rookDirections = listOf(
    Pair(0, 1),   // up
    Pair(1, 0),   // right
    Pair(0, -1),  // down
    Pair(-1, 0)   // left
)

@Suppress("NoMultipleSpaces")
val bishopDirections = listOf(
    Pair(1, 1),    // up-right
    Pair(1, -1),   // down-right
    Pair(-1, -1),  // down-left
    Pair(-1, 1)    // up-left
)

val queenDirections = rookDirections + bishopDirections

fun generateSlidingMoves(
    board: Board,
    square: Square,
    color: Color,
    directions: List<Pair<Int, Int>>
): List<Move> {
    val moves = mutableListOf<Move>()

    for ((dx, dy) in directions) {
        var file = square.file + dx
        var rank = square.rank + dy

        while (file in 0..7 && rank in 0..7) {
            val target = Square(file, rank)
            val targetPiece = board[target]

            if (targetPiece == null) {
                moves.add(Move(square, target))
            } else {
                if (targetPiece.color != color) {
                    moves.add(Move(square, target)) // capture
                }
                break // can't go beyond
            }

            file += dx
            rank += dy
        }
    }

    return moves
}

fun generateRookMoves(board: Board, square: Square, color: Color): List<Move> {
    return generateSlidingMoves(board, square, color, rookDirections)
}

fun generateBishopMoves(board: Board, square: Square, color: Color): List<Move> {
    return generateSlidingMoves(board, square, color, bishopDirections)
}

fun generateQueenMoves(board: Board, square: Square, color: Color): List<Move> {
    return generateSlidingMoves(board, square, color, queenDirections)
}
