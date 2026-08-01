package dev.kursor.vision.features.menu.presentation

interface MenuComponent {

    fun onLiveDetectionClick()

    sealed interface Output {
        data object LiveDetectionRequested : Output
    }
}