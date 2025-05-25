package dev.kursor.chess.features.root.presentation

import com.arkivanov.decompose.router.stack.ChildStack
import dev.kursor.chess.features.game.presentation.GameComponent
import kotlinx.coroutines.flow.StateFlow

interface RootComponent {

    val childStack: StateFlow<ChildStack<*, Child>>

    sealed interface Child {
        data class Game(val component: GameComponent) : Child
    }
}