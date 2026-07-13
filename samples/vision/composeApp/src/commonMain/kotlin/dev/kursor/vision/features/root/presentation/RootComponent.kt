package dev.kursor.vision.features.root.presentation

import com.arkivanov.decompose.router.stack.ChildStack
import dev.kursor.vision.features.live.presentation.LiveDetectionComponent
import dev.kursor.vision.features.menu.presentation.MenuComponent
import kotlinx.coroutines.flow.StateFlow

interface RootComponent {

    val childStack: StateFlow<ChildStack<*, Child>>

    sealed interface Child {
        data class Menu(val component: MenuComponent) : Child
        data class LiveDetection(val component: LiveDetectionComponent) : Child
    }
}