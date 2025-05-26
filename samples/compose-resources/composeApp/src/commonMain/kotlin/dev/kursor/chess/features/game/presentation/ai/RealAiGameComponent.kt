package dev.kursor.chess.features.game.presentation.ai

import com.arkivanov.decompose.ComponentContext
import dev.kursor.chess.core.state.computed
import dev.kursor.chess.core.utils.componentScope
import dev.kursor.chess.engine.api.BasicChessEngine
import dev.kursor.chess.engine.api.ChessEngine
import dev.kursor.chess.engine.model.Board
import dev.kursor.chess.features.game.domain.ChessAiMoveRepository
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.launchIn
import kotlinx.coroutines.flow.onEach

class RealAiGameComponent(
    componentContext: ComponentContext,
    private val chessAiMoveRepository: ChessAiMoveRepository
) : ComponentContext by componentContext, AiGameComponent {

    private val engine: ChessEngine = BasicChessEngine()

    override val board: StateFlow<Board> = computed(engine.state) {
        it.board
    }

    init {
        engine.state.onEach {
            val move = chessAiMoveRepository
                .getMove(gameState = engine.state.value)
            engine.applyMove(move)
        }.launchIn(componentScope)
    }
}