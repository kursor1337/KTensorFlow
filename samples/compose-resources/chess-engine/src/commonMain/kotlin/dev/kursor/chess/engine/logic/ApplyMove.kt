package dev.kursor.chess.engine.logic

import dev.kursor.chess.engine.logic.state.GameState
import dev.kursor.chess.engine.model.Color
import dev.kursor.chess.engine.model.Piece
import dev.kursor.chess.engine.model.PieceType
import dev.kursor.chess.engine.model.Square
import dev.kursor.chess.engine.model.movePiece
import dev.kursor.chess.engine.model.setPiece

fun GameState.applyMove(move: Move): GameState {
    // Update the board with the move
    var updatedBoard = board.movePiece(move.from, move.to)

    // Handle special moves
    val movedPiece = updatedBoard[move.to]
    val updatedCastlingRights = if (movedPiece != null) {
        // Castling: handle rook and king moves
        when (movedPiece.type) {
            PieceType.King -> {
                if (activeColor == Color.White) {
                    castlingRights.copy(
                        whiteKingSide = false,
                        whiteQueenSide = false
                    )
                } else {
                    castlingRights.copy(
                        blackKingSide = false,
                        blackQueenSide = false
                    )
                }
            }
            PieceType.Rook -> {
                if (activeColor == Color.White) {
                    when (move.from) {
                        Square(0, 0) -> castlingRights.copy(whiteQueenSide = false)
                        Square(7, 0) -> castlingRights.copy(whiteKingSide = false)
                        else -> castlingRights
                    }
                } else {
                    when (move.from) {
                        Square(0, 7) -> castlingRights.copy(blackQueenSide = false)
                        Square(7, 7) -> castlingRights.copy(blackKingSide = false)
                        else -> castlingRights
                    }
                }
            }
            else -> {
                castlingRights
            }
        }
    } else {
        castlingRights
    }

    // Handle promotion (pawn reaches the 8th rank)
    if (movedPiece != null &&
        movedPiece.type == PieceType.Pawn &&
        (move.to.rank == 0 || move.to.rank == 7)
        ) {
        val promotedPiece = move.promotion ?: PieceType.Queen // Default to queen if not specified
        updatedBoard = updatedBoard.setPiece(move.to, Piece(promotedPiece, activeColor))
    }

    // En Passant (if applicable)
    if (enPassantTarget == move.to) {
        val enPassantCapturedSquare = Square(move.to.file, move.from.rank)
        updatedBoard = updatedBoard.setPiece(enPassantCapturedSquare, null) // Remove the captured pawn
    }

    // Update En Passant Target (if a pawn moves two squares forward)
    val enPassantSquare = if (updatedBoard[move.from]?.type == PieceType.Pawn &&
        (move.from.rank == 1 && move.to.rank == 3 || move.from.rank == 6 && move.to.rank == 4)
    ) {
        Square(move.to.file, if (activeColor == Color.White) 2 else 5)
    } else {
        null
    }

    // Switch player (active color)
    val newColor = if (activeColor == Color.White) Color.Black else Color.White

    // Increment move counters
    val newFullMoveNumber =
        if (newColor == Color.White) fullMoveNumber + 1 else fullMoveNumber
    val newHalfMoveClock =
        if (newColor == Color.White && move.from != move.to) 0 else halfMoveClock + 1

    // Return updated game state
    return copy(
        board = updatedBoard,
        activeColor = newColor,
        castlingRights = updatedCastlingRights,
        enPassantTarget = enPassantSquare,
        halfMoveClock = newHalfMoveClock,
        fullMoveNumber = newFullMoveNumber
    )
}
