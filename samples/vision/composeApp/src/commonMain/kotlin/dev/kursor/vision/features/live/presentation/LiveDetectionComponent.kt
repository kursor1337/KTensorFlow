package dev.kursor.vision.features.live.presentation

import dev.kursor.ktensorflow.vision.Image
import dev.kursor.vision.features.live.domain.DetectionResult
import kotlinx.coroutines.flow.StateFlow

interface LiveDetectionComponent {
    val detectionResults: StateFlow<DetectionResult>
    fun onFrame(image: Image)
}