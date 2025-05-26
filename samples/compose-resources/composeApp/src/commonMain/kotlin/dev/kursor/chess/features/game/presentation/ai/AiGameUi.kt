package dev.kursor.chess.features.game.presentation.ai

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import dev.kursor.chess.engine.model.Square
import dev.kursor.chess.features.game.presentation.Board

@Composable
fun AiGameUi(
    component: AiGameComponent,
    modifier: Modifier = Modifier
) {
    val board by component.board.collectAsState()

    Column(
        modifier = modifier
            .fillMaxSize(),
        verticalArrangement = Arrangement.Center,
    ) {
        Board(
            board = board,
            isMyTurn = false,
            highlightedSquares = emptySet(),
            onSquareClick = {}
        )
    }

}