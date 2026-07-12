package dev.kursor.media.features.root

import com.arkivanov.decompose.ComponentContext
import dev.kursor.media.core.ComponentFactory
import dev.kursor.media.features.root.presentation.RealRootComponent
import dev.kursor.media.features.root.presentation.RootComponent
import org.koin.core.component.get

fun ComponentFactory.createRootComponent(
    componentContext: ComponentContext
): RootComponent {
    return RealRootComponent(
        componentContext,
        get(),
    )
}