package dev.kursor.chess.features.game

import com.arkivanov.decompose.ComponentContext
import dev.kursor.chess.core.ComponentFactory
import dev.kursor.chess.features.game.data.ChessAiMoveRepositoryImpl
import dev.kursor.chess.features.game.domain.ChessAiMoveRepository
import dev.kursor.chess.features.game.presentation.ai.AiGameComponent
import dev.kursor.chess.features.game.presentation.ai.RealAiGameComponent
import dev.kursor.chess.features.game.presentation.classic.ClassicGameComponent
import dev.kursor.chess.features.game.presentation.classic.RealClassicGameComponent
import org.koin.core.component.get
import org.koin.dsl.module

val gameModule = module {
    single<ChessAiMoveRepository> { ChessAiMoveRepositoryImpl(get()) }
}

fun ComponentFactory.createClassicGameComponent(
    componentContext: ComponentContext,
    onOutput: (ClassicGameComponent.Output) -> Unit,
): ClassicGameComponent {
    return RealClassicGameComponent(
        componentContext,
        onOutput,
        get()
    )
}

fun ComponentFactory.createAiGameComponent(
    componentContext: ComponentContext
): AiGameComponent {
    return RealAiGameComponent(
        componentContext,
        get()
    )
}