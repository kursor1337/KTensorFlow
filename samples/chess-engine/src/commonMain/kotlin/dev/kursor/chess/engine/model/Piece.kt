package dev.kursor.chess.engine.model

enum class PieceType {
    King, Queen, Rook, Bishop, Knight, Pawn
}

enum class Color {
    White, Black;

    val opposing: Color
        get() = if (this == White) Black else White
}

data class Piece(
    val type: PieceType,
    val color: Color
)
