package dev.kursor.chess.features.game.data

import dev.kursor.chess.core.model.ModelLoader
import dev.kursor.chess.core.utils.async
import dev.kursor.chess.engine.logic.Move
import dev.kursor.chess.engine.logic.move_generator.generateLegalMoves
import dev.kursor.chess.engine.logic.state.GameState
import dev.kursor.chess.engine.model.Board
import dev.kursor.chess.engine.model.Color
import dev.kursor.chess.engine.model.PieceType
import dev.kursor.chess.engine.model.Square
import dev.kursor.chess.features.game.domain.ChessAiMoveRepository
import dev.kursor.ktensorflow.api.Interpreter
import dev.kursor.ktensorflow.api.InterpreterOptions
import dev.kursor.ktensorflow.api.Tensor
import dev.kursor.ktensorflow.api.TensorDataType
import dev.kursor.ktensorflow.api.TensorShape
import dev.kursor.ktensorflow.api.typedData
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.withContext
import kotlin.time.ExperimentalTime
import kotlin.time.measureTime

class ChessAiMoveRepositoryImpl(
    modelLoader: ModelLoader
) : ChessAiMoveRepository {

    private val coroutineScope = CoroutineScope(Dispatchers.Main.immediate + SupervisorJob())

    private val interpreter: Interpreter by coroutineScope.async {
        Interpreter(
            modelDesc = modelLoader.loadModel(),
            options = InterpreterOptions()
        )
    }

    @OptIn(ExperimentalTime::class)
    override suspend fun getMove(gameState: GameState): Move = withContext(Dispatchers.Default) {
        val output = Tensor(
            dataType = TensorDataType.Float32,
            shape = TensorShape(4096)
        )
        val time = measureTime {
            interpreter.run(
                inputs = listOf(gameState.board.toTensor()),
                outputs = listOf(output)
            )
        }
        println("Inference time: ${time.inWholeMilliseconds} ms")
        val legalMoves = gameState.generateLegalMoves()
        output
            .typedData<FloatArray>()
            .withIndex()
            .map {
                val from = it.index / 64
                val to = it.index % 64

                it.value to Move(
                    from = Square(rank = from / 8, file = from % 8),
                    to = Square(rank = to / 8, file = to % 8),
                )
            }

            .filter { it.second in legalMoves }
            .maxBy { it.first }
            .second
    }
}

private fun Board.toTensor(): Tensor {
    val data = Array(8) { Array(8) { FloatArray(12) } }
    squares.forEachIndexed { i, row ->
        row.forEachIndexed { j, square ->
            if (square != null) {
                val color = when (square.color) {
                    Color.White -> 0
                    Color.Black -> 1
                }
                val piece = when (square.type) {
                    PieceType.Pawn -> 1
                    PieceType.Knight -> 2
                    PieceType.Bishop -> 3
                    PieceType.Rook -> 4
                    PieceType.Queen -> 5
                    PieceType.King -> 6
                }
                data[i][j][piece + color] = 1f
            }
        }
    }
    return Tensor(data)
}
