package dev.kursor.media.features.root.presentation

import com.arkivanov.decompose.ComponentContext
import com.arkivanov.decompose.router.stack.ChildStack
import com.arkivanov.decompose.router.stack.StackNavigation
import com.arkivanov.decompose.router.stack.childStack
import com.arkivanov.decompose.router.stack.pushNew
import dev.kursor.media.core.ComponentFactory
import dev.kursor.media.core.utils.toStateFlow
import dev.kursor.media.features.live.createLiveDetectionComponent
import dev.kursor.media.features.menu.createMenuComponent
import dev.kursor.media.features.menu.presentation.MenuComponent
import dev.kursor.media.features.root.presentation.RootComponent.Child.LiveDetection
import dev.kursor.media.features.root.presentation.RootComponent.Child.Menu
import kotlinx.coroutines.flow.StateFlow
import kotlinx.serialization.Serializable

class RealRootComponent(
    componentContext: ComponentContext,
    private val componentFactory: ComponentFactory
) : ComponentContext by componentContext, RootComponent {

    private val navigation = StackNavigation<ChildConfig>()

    override val childStack: StateFlow<ChildStack<*, RootComponent.Child>> = childStack(
        source = navigation,
        initialConfiguration = ChildConfig.Menu,
        serializer = ChildConfig.serializer(),
        handleBackButton = true,
        childFactory = ::createChild
    ).toStateFlow(lifecycle)

    private fun createChild(
        childConfig: ChildConfig,
        componentContext: ComponentContext
    ): RootComponent.Child = when (childConfig) {
        ChildConfig.Menu -> Menu(
            componentFactory.createMenuComponent(
                componentContext,
                ::onMenuOutput
            )
        )
        ChildConfig.LiveDetection -> LiveDetection(
            componentFactory.createLiveDetectionComponent(componentContext)
        )
    }

    private fun onMenuOutput(output: MenuComponent.Output) {
        when (output) {
            MenuComponent.Output.LiveDetectionRequested -> {
                navigation.pushNew(ChildConfig.LiveDetection)
            }
        }
    }

    @Serializable
    sealed interface ChildConfig {
        @Serializable
        data object Menu : ChildConfig

        @Serializable
        data object LiveDetection : ChildConfig
    }
}