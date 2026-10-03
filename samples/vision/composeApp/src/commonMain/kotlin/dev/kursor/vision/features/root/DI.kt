package dev.kursor.vision.features.root

import com.arkivanov.decompose.ComponentContext
import dev.kursor.vision.core.ComponentFactory
import dev.kursor.vision.features.root.presentation.RealRootComponent
import dev.kursor.vision.features.root.presentation.RootComponent
import org.koin.core.component.get

fun ComponentFactory.createRootComponent(
    componentContext: ComponentContext
): RootComponent {
    return RealRootComponent(
        componentContext,
        get(),
    )
}