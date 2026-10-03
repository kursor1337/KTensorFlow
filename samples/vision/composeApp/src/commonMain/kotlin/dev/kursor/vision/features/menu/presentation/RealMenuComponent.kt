package dev.kursor.vision.features.menu.presentation

import com.arkivanov.decompose.ComponentContext

class RealMenuComponent(
    componentContext: ComponentContext,
    private val onOutput: (MenuComponent.Output) -> Unit
) : ComponentContext by componentContext, MenuComponent {
    override fun onLiveDetectionClick() {
        onOutput(MenuComponent.Output.LiveDetectionRequested)
    }
}