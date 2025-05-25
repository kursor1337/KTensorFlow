package dev.kursor.chess.features

import dev.kursor.chess.features.game.gameModule
import org.koin.core.module.Module

val allFeatureModules = listOf<Module>(
    gameModule
)