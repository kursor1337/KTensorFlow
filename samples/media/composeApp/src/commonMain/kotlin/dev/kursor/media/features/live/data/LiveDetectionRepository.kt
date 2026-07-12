package dev.kursor.media.features.live.data

import dev.kursor.ktensorflow.vision.Image

interface LiveDetectionRepository {

    suspend fun detectDigit(image: Image): Int
}