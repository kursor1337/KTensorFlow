package dev.kursor.media.features.root.presentation

import com.arkivanov.decompose.router.stack.ChildStack
import dev.kursor.media.features.live.presentation.LiveDetectionComponent
import dev.kursor.media.features.menu.presentation.MenuComponent
import kotlinx.coroutines.flow.StateFlow

interface RootComponent {

    val childStack: StateFlow<ChildStack<*, Child>>

    sealed interface Child {
        data class Menu(val component: MenuComponent) : Child
        data class LiveDetection(val component: LiveDetectionComponent) : Child
    }
}