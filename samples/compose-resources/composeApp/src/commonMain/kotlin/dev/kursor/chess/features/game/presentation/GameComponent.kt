package dev.kursor.chess.features.game.presentation

import dev.kursor.chess.engine.model.Board
import dev.kursor.chess.engine.model.Square
import kotlinx.coroutines.flow.StateFlow

interface GameComponent {

    val board: StateFlow<Board>
    val isMyTurn: StateFlow<Boolean>
    val highlightedMoves: StateFlow<Set<Square>>
    val promotionSquare: StateFlow<Square?>

    fun onSquareClick(square: Square)

    sealed interface Output {
        data object GameLost : Output
        data object GameWon : Output
        data object GameDraw : Output
    }
}