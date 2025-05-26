package dev.kursor.chess.engine.logic.move_generator

import dev.kursor.chess.engine.logic.Move
import dev.kursor.chess.engine.logic.applyMove
import dev.kursor.chess.engine.logic.state.GameState
import dev.kursor.chess.engine.logic.state.isCheck

fun GameState.generateLegalMoves(): Set<Move> {
    val pseudoMoves = generatePseudoLegalMoves(
        board = board,
        color = activeColor,
        castlingRights = castlingRights,
        enPassantTarget = enPassantTarget
    )

    return pseudoMoves
        .filter { move ->
            val newState = applyMove(move)
            !isCheck(newState.board, activeColor)
        }
        .toSet()
}
