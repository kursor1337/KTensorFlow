package dev.kursor.media.features.live.presentation

import dev.kursor.ktensorflow.vision.Image
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

class FakeLiveDetectionComponent : LiveDetectionComponent {
    override val detectedDigit: StateFlow<Int> = MutableStateFlow(0)

    override fun onFrame(image: Image) = Unit
}