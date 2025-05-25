package dev.kursor.chess.features.game.presentation

import com.arkivanov.decompose.ComponentContext
import dev.kursor.chess.core.state.computed
import dev.kursor.chess.core.utils.componentScope
import dev.kursor.chess.engine.api.BasicChessEngine
import dev.kursor.chess.engine.api.ChessEngine
import dev.kursor.chess.engine.logic.Move
import dev.kursor.chess.engine.logic.move_generator.generateLegalMoves
import dev.kursor.chess.engine.model.Board
import dev.kursor.chess.engine.model.Color
import dev.kursor.chess.engine.model.Square
import dev.kursor.chess.features.game.domain.ChessAiMoveRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

class RealGameComponent(
    componentContext: ComponentContext,
    onOutput: (GameComponent.Output) -> Unit,
    private val chessAiMoveRepository: ChessAiMoveRepository
) : ComponentContext by componentContext, GameComponent {

    private val engine: ChessEngine = BasicChessEngine()

    override val board: StateFlow<Board> = computed(engine.state) {
        it.board
    }

    override val isMyTurn: StateFlow<Boolean> = computed(engine.state) {
        it.activeColor == Color.White
    }

    private val highlightedSquare = MutableStateFlow<Square?>(null)

    override val promotionSquare = MutableStateFlow<Square?>(null)

    override val highlightedMoves: StateFlow<Set<Square>> = computed(
        engine.state,
        highlightedSquare
    ) { state, square ->
        state
            .generateLegalMoves()
            .filter { it.from == square }
            .map { it.to }
            .toSet()
    }

    override fun onSquareClick(square: Square) {
        val highlighted = highlightedSquare.value
        when {
            highlighted == null && board.value[square] != null -> {
                highlightedSquare.value = square
            }

            highlighted == square -> {
                highlightedSquare.value = null
            }

            highlighted != null && square !in highlightedMoves.value -> {
                highlightedSquare.value = null
            }

            highlighted != null && square in highlightedMoves.value -> {
                engine.applyMove(Move(highlighted, square))
                highlightedSquare.value = null
                componentScope.launch {
                    val aiMove = chessAiMoveRepository.getMove(engine.state.value)
                    if (engine.state.value.activeColor == Color.Black) {
                        engine.applyMove(aiMove)
                    }
                }
            }
        }
    }
}