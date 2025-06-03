package dev.kursor.chess.engine.logic.move_generator

import dev.kursor.chess.engine.logic.Move
import dev.kursor.chess.engine.logic.state.CastlingRights
import dev.kursor.chess.engine.model.Board
import dev.kursor.chess.engine.model.Color
import dev.kursor.chess.engine.model.PieceType
import dev.kursor.chess.engine.model.Square

fun generatePseudoLegalMoves(
    board: Board,
    color: Color,
    castlingRights: CastlingRights,
    enPassantTarget: Square?
): List<Move> {
    val moves = mutableListOf<Move>()

    for (rank in 0..7) {
        for (file in 0..7) {
            val square = Square(file, rank)
            val piece = board[square] ?: continue
            if (piece.color != color) continue

            val pieceMoves = when (piece.type) {
                PieceType.Pawn -> generatePawnMoves(board, square, color, enPassantTarget)
                PieceType.Knight -> generateKnightMoves(board, square, color)
                PieceType.Bishop -> generateBishopMoves(board, square, color)
                PieceType.Rook -> generateRookMoves(board, square, color)
                PieceType.Queen -> generateQueenMoves(board, square, color)
                PieceType.King -> generateAllKingMoves(board, square, color, castlingRights)
            }

            moves += pieceMoves
        }
    }

    return moves
}
