package dev.kursor.chess.features.game.presentation

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.blur
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import dev.kursor.chess.engine.model.Board
import dev.kursor.chess.engine.model.Piece
import dev.kursor.chess.engine.model.PieceType
import dev.kursor.chess.engine.model.Square
import dev.kursor.chess.engine.model.isLight
import ktensorflow.samples.compose_resources.composeapp.generated.resources.Res
import ktensorflow.samples.compose_resources.composeapp.generated.resources.ic_45_black_bishop
import ktensorflow.samples.compose_resources.composeapp.generated.resources.ic_45_black_king
import ktensorflow.samples.compose_resources.composeapp.generated.resources.ic_45_black_knight
import ktensorflow.samples.compose_resources.composeapp.generated.resources.ic_45_black_pawn
import ktensorflow.samples.compose_resources.composeapp.generated.resources.ic_45_black_queen
import ktensorflow.samples.compose_resources.composeapp.generated.resources.ic_45_black_rook
import ktensorflow.samples.compose_resources.composeapp.generated.resources.ic_45_white_bishop
import ktensorflow.samples.compose_resources.composeapp.generated.resources.ic_45_white_king
import ktensorflow.samples.compose_resources.composeapp.generated.resources.ic_45_white_knight
import ktensorflow.samples.compose_resources.composeapp.generated.resources.ic_45_white_pawn
import ktensorflow.samples.compose_resources.composeapp.generated.resources.ic_45_white_queen
import ktensorflow.samples.compose_resources.composeapp.generated.resources.ic_45_white_rook
import org.jetbrains.compose.resources.DrawableResource
import org.jetbrains.compose.resources.painterResource
import dev.kursor.chess.engine.model.Color as ChessColor

@Composable
fun Board(
    board: Board,
    isMyTurn: Boolean,
    highlightedSquares: Set<Square>,
    onSquareClick: (Square) -> Unit,
    modifier: Modifier = Modifier
) {
    Column(modifier = modifier) {
        board.squares
            .reversed()
            .map { it.reversed() }
            .forEachIndexed { rank, row ->
                Row {
                    row.forEachIndexed { file, piece ->
                        val square = Square(8 - 1 - file, 8 - 1 - rank)
                        Square(
                            square = square,
                            piece = piece,
                            isHighlighted = square in highlightedSquares,
                            isMyTurn = isMyTurn,
                            onClick = { onSquareClick(square) },
                            modifier = Modifier.weight(1f)
                        )
                    }
                }
            }
    }
}

@Composable
private fun Square(
    square: Square,
    piece: Piece?,
    isHighlighted: Boolean,
    isMyTurn: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    Box(
        modifier = modifier
            .aspectRatio(1f)
            .then(
                if (square.isLight) {
                    Modifier.background(Color.LightGray)
                } else {
                    Modifier.background(Color.DarkGray)
                }
            )
            .clickable(enabled = isMyTurn, onClick = onClick)
    ) {
        piece?.imageResource?.let {
            Image(
                painter = painterResource(it),
                contentDescription = null,
                modifier = Modifier
                    .matchParentSize()
            )
        }
        if (isHighlighted) {
            Box(
                modifier = Modifier
                    .size(24.dp)
                    .background(
                        color = Color.Green.copy(alpha = 0.5f),
                        shape = CircleShape
                    )
                    .align(Alignment.Center)
                    .blur(radius = 24.dp)
            )
        }
    }
}

private val Piece.imageResource: DrawableResource
    get() = when (color) {
        ChessColor.White -> {
            when (type) {
                PieceType.King -> Res.drawable.ic_45_white_king
                PieceType.Queen -> Res.drawable.ic_45_white_queen
                PieceType.Rook -> Res.drawable.ic_45_white_rook
                PieceType.Bishop -> Res.drawable.ic_45_white_bishop
                PieceType.Knight -> Res.drawable.ic_45_white_knight
                PieceType.Pawn -> Res.drawable.ic_45_white_pawn
            }
        }

        ChessColor.Black -> {
            when (type) {
                PieceType.King -> Res.drawable.ic_45_black_king
                PieceType.Queen -> Res.drawable.ic_45_black_queen
                PieceType.Rook -> Res.drawable.ic_45_black_rook
                PieceType.Bishop -> Res.drawable.ic_45_black_bishop
                PieceType.Knight -> Res.drawable.ic_45_black_knight
                PieceType.Pawn -> Res.drawable.ic_45_black_pawn
            }
        }
    }