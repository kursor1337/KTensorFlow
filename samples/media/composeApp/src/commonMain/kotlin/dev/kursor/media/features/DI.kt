package dev.kursor.media.features

import dev.kursor.media.features.live.liveDetectionModule
import org.koin.core.module.Module

val allFeatureModules = listOf(
    liveDetectionModule
)