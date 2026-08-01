package dev.kursor.vision.features.live.presentation

import com.arkivanov.decompose.ComponentContext
import dev.kursor.ktensorflow.vision.Image
import dev.kursor.vision.core.utils.componentScope
import dev.kursor.vision.features.live.data.LiveDetectionRepository
import dev.kursor.vision.features.live.domain.DetectionResult
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.launch
import kotlin.concurrent.atomics.AtomicBoolean
import kotlin.concurrent.atomics.ExperimentalAtomicApi

@OptIn(ExperimentalAtomicApi::class)
class RealLiveDetectionComponent(
    componentContext: ComponentContext,
    private val liveDetectionRepository: LiveDetectionRepository
) : ComponentContext by componentContext, LiveDetectionComponent {

    override val detectionResults = MutableStateFlow(DetectionResult(emptyList()))

    private val isProcessing = AtomicBoolean(false)

    override fun onFrame(image: Image) {
        if (!isProcessing.compareAndSet(expectedValue = false, newValue = true)) {
            image.close()
            return
        }

        componentScope.launch {
            try {
                detectionResults.value = liveDetectionRepository.detectObjects(image)
            } finally {
                image.close()
                isProcessing.store(false)
            }
        }
    }
}