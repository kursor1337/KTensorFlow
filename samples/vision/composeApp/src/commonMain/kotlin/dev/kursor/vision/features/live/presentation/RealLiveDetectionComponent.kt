package dev.kursor.vision.features.live.presentation

import com.arkivanov.decompose.ComponentContext
import dev.kursor.ktensorflow.vision.Image
import dev.kursor.vision.core.utils.componentScope
import dev.kursor.vision.features.live.data.LiveDetectionRepository
import dev.kursor.vision.features.live.domain.DetectionResult
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.channels.onFailure
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.consumeAsFlow
import kotlinx.coroutines.launch

class RealLiveDetectionComponent(
    componentContext: ComponentContext,
    liveDetectionRepository: LiveDetectionRepository
) : ComponentContext by componentContext, LiveDetectionComponent {

    override val detectionResults = MutableStateFlow(DetectionResult(emptyList()))

    private val frames = Channel<Image>(Channel.CONFLATED, onUndeliveredElement = { it.close() })

    init {
        componentScope.launch {
            liveDetectionRepository
                .detectObjects(frames.consumeAsFlow())
                .collect { detectionResults.value = it }
        }
    }

    override fun onFrame(image: Image) {
        frames.trySend(image).onFailure { image.close() }
    }
}
