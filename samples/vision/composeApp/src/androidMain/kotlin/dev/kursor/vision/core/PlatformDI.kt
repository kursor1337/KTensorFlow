package dev.kursor.vision.core

import android.app.Application
import android.content.Context
import dev.kursor.vision.core.configuration.Configuration
import org.koin.dsl.module

actual fun platformCoreModule(configuration: Configuration) = module {
    single<Configuration> { configuration }
    single<Application> { get<Configuration>().platform.application }
    single<Context> { get<Application>() }
}
