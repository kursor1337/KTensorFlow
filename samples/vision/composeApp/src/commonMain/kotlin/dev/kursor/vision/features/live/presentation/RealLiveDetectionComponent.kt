package dev.kursor.vision.features.live.presentation

import com.arkivanov.decompose.ComponentContext
import dev.kursor.ktensorflow.vision.Image
import dev.kursor.vision.core.utils.componentScope
import dev.kursor.vision.features.live.data.LiveDetectionRepository
import dev.kursor.vision.features.live.domain.DetectionResult
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

class RealLiveDetectionComponent(
    componentContext: ComponentContext,
    private val liveDetectionRepository: LiveDetectionRepository
) : ComponentContext by componentContext, LiveDetectionComponent {

    override val detectionResults = MutableStateFlow(DetectionResult(emptyList()))

    override fun onFrame(image: Image) {
        componentScope.launch {
            detectionResults.value = liveDetectionRepository.detectObjects(image)
        }
    }
}