package dev.kursor.media.features.menu.presentation

interface MenuComponent {

    fun onLiveDetectionClick()

    sealed interface Output {
        data object LiveDetectionRequested : Output
    }
}