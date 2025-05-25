package dev.kursor.chess.features.game

import com.arkivanov.decompose.ComponentContext
import dev.kursor.chess.core.ComponentFactory
import dev.kursor.chess.features.game.data.ChessAiMoveRepositoryImpl
import dev.kursor.chess.features.game.domain.ChessAiMoveRepository
import dev.kursor.chess.features.game.presentation.GameComponent
import dev.kursor.chess.features.game.presentation.RealGameComponent
import org.koin.core.component.get
import org.koin.dsl.module

val gameModule = module {
    single<ChessAiMoveRepository> { ChessAiMoveRepositoryImpl() }
}

fun ComponentFactory.createGameComponent(
    componentContext: ComponentContext,
    onOutput: (GameComponent.Output) -> Unit,
): GameComponent {
    return RealGameComponent(
        componentContext,
        onOutput,
        get()
    )
}