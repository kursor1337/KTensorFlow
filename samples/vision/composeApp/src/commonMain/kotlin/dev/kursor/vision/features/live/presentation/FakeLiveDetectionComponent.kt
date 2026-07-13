package dev.kursor.vision.features.live.presentation

import dev.kursor.ktensorflow.vision.Image
import dev.kursor.vision.features.live.domain.DetectionResult
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

class FakeLiveDetectionComponent : LiveDetectionComponent {
    override val detectionResults: StateFlow<DetectionResult> =
        MutableStateFlow(DetectionResult(emptyList()))

    override fun onFrame(image: Image) = Unit
}