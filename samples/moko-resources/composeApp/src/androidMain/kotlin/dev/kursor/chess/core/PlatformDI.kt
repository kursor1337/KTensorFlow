package dev.kursor.chess.core

import android.app.Application
import android.content.Context
import dev.kursor.chess.core.configuration.Configuration
import org.koin.dsl.module

actual fun platformCoreModule(configuration: Configuration) = module {
    single<Configuration> { configuration }
    single<Application> { get<Configuration>().platform.application }
    single<Context> { get<Application>() }
}
