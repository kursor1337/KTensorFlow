package dev.kursor.chess.features.root.presentation

import com.arkivanov.decompose.router.stack.ChildStack
import dev.kursor.chess.features.game.presentation.ai.AiGameComponent
import dev.kursor.chess.features.game.presentation.classic.ClassicGameComponent
import dev.kursor.chess.features.menu.presentation.MenuComponent
import kotlinx.coroutines.flow.StateFlow

interface RootComponent {

    val childStack: StateFlow<ChildStack<*, Child>>

    sealed interface Child {
        data class Menu(val component: MenuComponent) : Child
        data class ClassicGame(val component: ClassicGameComponent) : Child
        data class AiGame(val component: AiGameComponent) : Child
    }
}