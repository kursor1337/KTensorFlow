package dev.kursor.chess.engine.model

data class Board(
    val squares: List<List<Piece?>> = initialSquares()
) {

    operator fun get(square: Square): Piece? {
        return squares[square.rank][square.file]
    }

    operator fun get(rank: Int, file: Int): Piece? {
        return squares[rank][file]
    }

    override fun toString(): String {
        return squares.reversed().joinToString("\n") { row ->
            row.joinToString(" ") { it?.let { "${it.color.name[0]}${it.type.name[0]}" } ?: "." }
        }
    }
}

fun Board.movePiece(
    from: Square,
    to: Square
): Board = copy(
    squares = squares.mapIndexed { rank, row ->
        row.mapIndexed { file, piece ->
            when {
                rank == from.rank && file == from.file -> null
                rank == to.rank && file == to.file -> this[from]
                else -> piece
            }
        }
    }
)

fun Board.setPiece(square: Square, piece: Piece?): Board = copy(
    squares = squares.mapIndexed { rank, row ->
        row.mapIndexed { file, currentPiece ->
            if (rank == square.rank && file == square.file) {
                piece
            } else {
                currentPiece
            }
        }
    }
)

private fun initialSquares(): List<List<Piece?>> {
    val types = listOf(
        PieceType.Rook, PieceType.Knight, PieceType.Bishop,
        PieceType.Queen, PieceType.King,
        PieceType.Bishop, PieceType.Knight, PieceType.Rook
    )
    return List(8) { rank ->
        List(8) { file ->
            when (rank) {
                0 -> Piece(types[file], Color.White)
                1 -> Piece(PieceType.Pawn, Color.White)
                6 -> Piece(PieceType.Pawn, Color.Black)
                7 -> Piece(types[file], Color.Black)
                else -> null
            }
        }
    }
}
