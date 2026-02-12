package dev.kursor.media

import android.app.Application
import dev.kursor.media.core.SharedApplication
import dev.kursor.media.core.SharedApplicationProvider
import dev.kursor.media.core.configuration.BuildType
import dev.kursor.media.core.configuration.Configuration
import dev.kursor.media.core.configuration.Platform
import org.jetbrains.compose.resources.ExperimentalResourceApi

class App : Application(), SharedApplicationProvider {

    override lateinit var sharedApplication: SharedApplication
        private set

    @OptIn(ExperimentalResourceApi::class)
    override fun onCreate() {
        super.onCreate()
        sharedApplication = SharedApplication(getConfiguration())
    }

    @Suppress("SENSELESS_COMPARISON")
    private fun getConfiguration() = Configuration(
        platform = Platform(this),
        buildType = if (BuildConfig.DEBUG) BuildType.Debug else BuildType.Release
    )
}