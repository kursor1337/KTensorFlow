package dev.kursor.chess.features.menu

import com.arkivanov.decompose.ComponentContext
import dev.kursor.chess.core.ComponentFactory
import dev.kursor.chess.features.menu.presentation.MenuComponent
import dev.kursor.chess.features.menu.presentation.RealMenuComponent

fun ComponentFactory.createMenuComponent(
    componentContext: ComponentContext,
    onOutput: (MenuComponent.Output) -> Unit
): MenuComponent {
    return RealMenuComponent(
        componentContext,
        onOutput
    )
}