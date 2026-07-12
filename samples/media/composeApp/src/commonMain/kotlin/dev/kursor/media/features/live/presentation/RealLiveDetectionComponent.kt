package dev.kursor.media.features.live.presentation

import com.arkivanov.decompose.ComponentContext
import dev.kursor.ktensorflow.vision.Image
import dev.kursor.media.core.utils.componentScope
import dev.kursor.media.features.live.data.LiveDetectionRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.launch

class RealLiveDetectionComponent(
    componentContext: ComponentContext,
    private val liveDetectionRepository: LiveDetectionRepository
) : ComponentContext by componentContext, LiveDetectionComponent {
    override val detectedDigit = MutableStateFlow(0)

    override fun onFrame(image: Image) {
        componentScope.launch {
            detectedDigit.value = liveDetectionRepository.detectDigit(image)
        }
    }
}