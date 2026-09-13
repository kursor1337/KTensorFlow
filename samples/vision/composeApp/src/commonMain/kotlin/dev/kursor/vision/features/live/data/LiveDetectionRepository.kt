package dev.kursor.vision.features.live.data

import dev.kursor.ktensorflow.vision.Image
import dev.kursor.vision.features.live.domain.DetectionResult

interface LiveDetectionRepository {

    suspend fun detectObjects(image: Image): DetectionResult
}