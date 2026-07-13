package dev.kursor.vision

import android.app.Application
import dev.kursor.vision.core.SharedApplication
import dev.kursor.vision.core.SharedApplicationProvider
import dev.kursor.vision.core.configuration.BuildType
import dev.kursor.vision.core.configuration.Configuration
import dev.kursor.vision.core.configuration.Platform
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