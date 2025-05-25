package dev.kursor.chess.engine.logic.state

fun isFiftyMoveRule(state: GameState): Boolean {
    return state.halfMoveClock >= 50
}
