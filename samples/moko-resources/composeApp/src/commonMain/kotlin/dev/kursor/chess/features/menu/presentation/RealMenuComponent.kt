package dev.kursor.chess.features.menu.presentation

import com.arkivanov.decompose.ComponentContext

class RealMenuComponent(
    componentContext: ComponentContext,
    private val onOutput: (MenuComponent.Output) -> Unit
) : ComponentContext by componentContext, MenuComponent {

    override fun onClassicGameClick() {
        onOutput(MenuComponent.Output.ClassicGameRequested)
    }

    override fun onAiGameClick() {
        onOutput(MenuComponent.Output.AiGameRequested)
    }
}