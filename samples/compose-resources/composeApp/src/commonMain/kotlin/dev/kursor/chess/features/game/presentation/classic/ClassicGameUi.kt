package dev.kursor.chess.features.game.presentation.classic

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import dev.kursor.chess.features.game.presentation.Board

@Composable
fun ClassicGameUi(
    component: ClassicGameComponent,
    modifier: Modifier = Modifier
) {
    val board by component.board.collectAsState()
    val isMyTurn by component.isMyTurn.collectAsState()
    val highlightedMoves by component.highlightedMoves.collectAsState()

    Column(
        modifier = modifier
            .fillMaxSize(),
        verticalArrangement = Arrangement.Center,
    ) {
        Board(
            board = board,
            isMyTurn = isMyTurn,
            highlightedSquares = highlightedMoves,
            onSquareClick = component::onSquareClick
        )
    }

}