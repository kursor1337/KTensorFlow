package dev.kursor.chess.engine.logic

import dev.kursor.chess.engine.model.PieceType
import dev.kursor.chess.engine.model.Square

data class Move(
    val from: Square,
    val to: Square,
    val promotion: PieceType? = null
)
