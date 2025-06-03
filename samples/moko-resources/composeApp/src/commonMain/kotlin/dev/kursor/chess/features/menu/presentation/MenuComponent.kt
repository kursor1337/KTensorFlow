package dev.kursor.chess.features.menu.presentation

interface MenuComponent {

    fun onClassicGameClick()

    fun onAiGameClick()

    sealed interface Output {
        data object ClassicGameRequested : Output
        data object AiGameRequested : Output
    }
}