package dev.kursor.vision.features.menu

import com.arkivanov.decompose.ComponentContext
import dev.kursor.vision.core.ComponentFactory
import dev.kursor.vision.features.menu.presentation.MenuComponent
import dev.kursor.vision.features.menu.presentation.RealMenuComponent

fun ComponentFactory.createMenuComponent(
    componentContext: ComponentContext,
    onOutput: (MenuComponent.Output) -> Unit
): MenuComponent {
    return RealMenuComponent(
        componentContext,
        onOutput
    )
}