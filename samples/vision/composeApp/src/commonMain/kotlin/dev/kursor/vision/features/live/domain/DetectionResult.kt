package dev.kursor.vision.features.live.domain

import dev.kursor.ktensorflow.vision.Rect

data class DetectionResult(
    val objects: List<DetectedObject>
)

data class DetectedObject(
    val label: String,
    val confidence: Float,
    val boundingBox: Rect
)
