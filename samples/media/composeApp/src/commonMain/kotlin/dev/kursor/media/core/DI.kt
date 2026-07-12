package dev.kursor.media.core

import dev.kursor.media.core.configuration.Configuration
import org.koin.core.module.Module
import org.koin.dsl.module

fun commonCoreModule(configuration: Configuration) = module {
    single { configuration }
}

expect fun platformCoreModule(configuration: Configuration): Module