package dev.kursor.media.core

import android.app.Application
import android.content.Context
import dev.icerock.moko.permissions.PermissionsController
import dev.icerock.moko.permissions.PermissionsControllerImpl
import dev.kursor.media.core.configuration.Configuration
import org.koin.dsl.module

actual fun platformCoreModule(configuration: Configuration) = module {
    single<Configuration> { configuration }
    single<Application> { get<Configuration>().platform.application }
    single<Context> { get<Application>() }
}
