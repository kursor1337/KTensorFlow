package dev.kursor.media.features.live.data

import dev.kursor.ktensorflow.media.Image

interface LiveDetectionRepository {

    suspend fun detectDigit(image: Image): Int
}