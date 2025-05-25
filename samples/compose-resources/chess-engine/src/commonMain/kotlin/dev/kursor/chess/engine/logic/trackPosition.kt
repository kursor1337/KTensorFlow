package dev.kursor.chess.engine.logic

import dev.kursor.chess.engine.logic.state.GameState
import dev.kursor.chess.engine.model.Color

fun trackPosition(state: GameState, positionsHistory: MutableList<String>) {
    val currentFEN = getFenFromState(state)
    positionsHistory.add(currentFEN)
}

fun getFenFromState(state: GameState): String {
    // 1. Piece placement
    val boardFen = state.board.squares.map { row ->
        row.joinToString("") { square ->
            when (val piece = square) {
                null -> "1"  // empty square
                else -> when (piece.color) {
                    Color.White -> piece.type.name[0].uppercaseChar().toString()
                    Color.Black -> piece.type.name[0].lowercaseChar().toString()
                }
            }
        }
    }.joinToString("/") { it.replace("1{4,}", "1") }  // Simplify empty squares

    // 2. Active color
    val activeColorFen = if (state.activeColor == Color.White) "w" else "b"

    // 3. Castling rights
    val castlingFen = buildString {
        if (state.castlingRights.whiteKingSide) append("K")
        if (state.castlingRights.whiteQueenSide) append("Q")
        if (state.castlingRights.blackKingSide) append("k")
        if (state.castlingRights.blackQueenSide) append("q")
        if (isEmpty()) append("-")  // No castling rights
    }

    // 4. En passant target square
    val enPassantFen = state.enPassantTarget?.let {
        "${'a' + it.file}${8 - it.rank}"
    } ?: "-"

    // 5. Halfmove clock
    val halfMoveClockFen = state.halfMoveClock.toString()

    // 6. Fullmove number
    val fullMoveNumberFen = state.fullMoveNumber.toString()

    return listOf(boardFen, activeColorFen, castlingFen, enPassantFen, halfMoveClockFen, fullMoveNumberFen).joinToString(" ")
}

