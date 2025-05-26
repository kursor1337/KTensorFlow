package dev.kursor.chess.engine.logic.move_generator

import dev.kursor.chess.engine.logic.Move
import dev.kursor.chess.engine.logic.state.CastlingRights
import dev.kursor.chess.engine.model.Board
import dev.kursor.chess.engine.model.Color
import dev.kursor.chess.engine.model.Square

fun generateAllKingMoves(board: Board, square: Square, color: Color, castlingRights: CastlingRights): List<Move> {
    val standardMoves = generateKingMoves(board, square, color)
    val castlingMoves = generateCastlingMoves(board, color, castlingRights)
    return standardMoves + castlingMoves
}

@Suppress("NoMultipleSpaces")
fun generateKingMoves(board: Board, square: Square, color: Color): List<Move> {
    val moves = mutableListOf<Move>()
    val offsets = listOf(
        -1 to -1, 0 to -1, 1 to -1,
        -1 to 0,           1 to 0,
        -1 to 1,  0 to 1,  1 to 1
    )

    for ((dx, dy) in offsets) {
        val file = square.file + dx
        val rank = square.rank + dy
        if (file !in 0..7 || rank !in 0..7) continue

        val target = Square(file, rank)
        val piece = board[target]
        if (piece == null || piece.color != color) {
            moves.add(Move(square, target))
        }
    }

    return moves
}

fun generateCastlingMoves(board: Board, color: Color, castlingRights: CastlingRights): List<Move> {
    val moves = mutableListOf<Move>()
    val rank = if (color == Color.White) 0 else 7
    val kingSquare = Square(4, rank)

    if ((color == Color.White && castlingRights.whiteKingSide) ||
        (color == Color.Black && castlingRights.blackKingSide)) {
        // Kingside
        if (board[5, rank] == null &&
            board[6, rank] == null) {
            moves.add(Move(kingSquare, Square(6, rank))) // e1→g1 / e8→g8
        }
    }

    if ((color == Color.White && castlingRights.whiteQueenSide) ||
        (color == Color.Black && castlingRights.blackQueenSide)) {
        // Queenside
        if (board[3, rank] == null &&
            board[2, rank] == null &&
            board[1, rank] == null) {
            moves.add(Move(kingSquare, Square(2, rank))) // e1→c1 / e8→c8
        }
    }

    return moves
}
