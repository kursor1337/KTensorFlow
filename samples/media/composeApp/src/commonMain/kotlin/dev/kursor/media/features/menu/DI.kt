package dev.kursor.media.features.menu

import com.arkivanov.decompose.ComponentContext
import dev.kursor.media.core.ComponentFactory
import dev.kursor.media.features.menu.presentation.MenuComponent
import dev.kursor.media.features.menu.presentation.RealMenuComponent

fun ComponentFactory.createMenuComponent(
    componentContext: ComponentContext,
    onOutput: (MenuComponent.Output) -> Unit
): MenuComponent {
    return RealMenuComponent(
        componentContext,
        onOutput
    )
}