package dev.kursor.chess.features.root.presentation

import com.arkivanov.decompose.ComponentContext
import com.arkivanov.decompose.router.stack.ChildStack
import com.arkivanov.decompose.router.stack.StackNavigation
import com.arkivanov.decompose.router.stack.childStack
import com.arkivanov.decompose.router.stack.pushNew
import dev.kursor.chess.core.ComponentFactory
import dev.kursor.chess.core.utils.toStateFlow
import dev.kursor.chess.features.game.createAiGameComponent
import dev.kursor.chess.features.game.createClassicGameComponent
import dev.kursor.chess.features.game.presentation.classic.ClassicGameComponent
import dev.kursor.chess.features.menu.createMenuComponent
import dev.kursor.chess.features.menu.presentation.MenuComponent
import dev.kursor.chess.features.root.presentation.RootComponent.Child.*
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
        ChildConfig.ClassicGame -> ClassicGame(
            componentFactory.createClassicGameComponent(
                componentContext,
                ::onGameOutput
            )
        )

        ChildConfig.AiGame -> AiGame(
            componentFactory.createAiGameComponent(
                componentContext
            )
        )
    }

    private fun onMenuOutput(output: MenuComponent.Output) {
        when (output) {
            MenuComponent.Output.AiGameRequested -> navigation.pushNew(ChildConfig.AiGame)
            MenuComponent.Output.ClassicGameRequested -> navigation.pushNew(ChildConfig.ClassicGame)
        }
    }

    private fun onGameOutput(output: ClassicGameComponent.Output) = Unit

    @Serializable
    sealed interface ChildConfig {
        @Serializable
        data object Menu : ChildConfig

        @Serializable
        data object ClassicGame : ChildConfig

        @Serializable
        data object AiGame : ChildConfig
    }
}