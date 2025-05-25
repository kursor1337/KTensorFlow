package dev.kursor.chess.features.root.presentation

import com.arkivanov.decompose.ComponentContext
import com.arkivanov.decompose.router.stack.ChildStack
import com.arkivanov.decompose.router.stack.StackNavigation
import com.arkivanov.decompose.router.stack.childStack
import dev.kursor.chess.core.ComponentFactory
import dev.kursor.chess.core.utils.toStateFlow
import dev.kursor.chess.features.game.createGameComponent
import dev.kursor.chess.features.game.presentation.GameComponent
import kotlinx.coroutines.flow.StateFlow
import kotlinx.serialization.Serializable

class RealRootComponent(
    componentContext: ComponentContext,
    private val componentFactory: ComponentFactory
) : ComponentContext by componentContext, RootComponent {

    private val navigation = StackNavigation<ChildConfig>()

    override val childStack: StateFlow<ChildStack<*, RootComponent.Child>> = childStack(
        source = navigation,
        initialConfiguration = ChildConfig.Game,
        serializer = ChildConfig.serializer(),
        handleBackButton = true,
        childFactory = ::createChild
    ).toStateFlow(lifecycle)

    private fun createChild(
        childConfig: ChildConfig,
        componentContext: ComponentContext
    ): RootComponent.Child = when (childConfig) {
        ChildConfig.Game -> RootComponent.Child.Game(
            componentFactory.createGameComponent(
                componentContext,
                ::onGameOutput
            )
        )
    }

    private fun onGameOutput(output: GameComponent.Output) {
        when (output) {
            GameComponent.Output.GameDraw -> TODO()
            GameComponent.Output.GameLost -> TODO()
            GameComponent.Output.GameWon -> TODO()
        }
    }

    @Serializable
    sealed interface ChildConfig {
        @Serializable
        data object Game : ChildConfig
    }
}