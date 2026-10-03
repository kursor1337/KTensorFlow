package dev.kursor.vision.features.live.data

import dev.kursor.ktensorflow.vision.Image
import dev.kursor.vision.features.live.domain.DetectionResult
import kotlinx.coroutines.flow.Flow

interface LiveDetectionRepository {

    /**
     * Detects objects on camera frames. A frame that arrives while the previous one is still
     * being processed is dropped. Every frame is closed once it is processed or dropped.
     */
    fun detectObjects(frames: Flow<Image>): Flow<DetectionResult>
}
