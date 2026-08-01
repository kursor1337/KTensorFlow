package dev.kursor.vision.core

import dev.kursor.vision.core.configuration.Configuration
import org.koin.core.module.Module
import org.koin.dsl.module

fun commonCoreModule(configuration: Configuration) = module {
    single { configuration }
}

expect fun platformCoreModule(configuration: Configuration): Module