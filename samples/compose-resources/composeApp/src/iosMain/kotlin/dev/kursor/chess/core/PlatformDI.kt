package dev.kursor.chess.core

import dev.kursor.chess.core.configuration.Configuration
import dev.kursor.chess.core.model.IosModelLoader
import dev.kursor.chess.core.model.ModelLoader
import org.koin.dsl.module

actual fun platformCoreModule(configuration: Configuration) = module {
    single<ModelLoader> { IosModelLoader() }
}