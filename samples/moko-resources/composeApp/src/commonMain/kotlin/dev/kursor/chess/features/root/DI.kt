package dev.kursor.chess.features.root

import com.arkivanov.decompose.ComponentContext
import dev.kursor.chess.core.ComponentFactory
import dev.kursor.chess.features.root.presentation.RealRootComponent
import dev.kursor.chess.features.root.presentation.RootComponent
import org.koin.core.component.get

fun ComponentFactory.createRootComponent(
    componentContext: ComponentContext
): RootComponent {
    return RealRootComponent(
        componentContext,
        get(),
    )
}