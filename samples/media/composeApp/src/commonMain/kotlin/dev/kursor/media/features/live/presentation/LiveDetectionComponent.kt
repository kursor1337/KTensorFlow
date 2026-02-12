package dev.kursor.media.features.live.presentation

import dev.kursor.ktensorflow.media.Image
import kotlinx.coroutines.flow.StateFlow

interface LiveDetectionComponent {
    val detectedDigit: StateFlow<Int>
    fun onFrame(image: Image)
}