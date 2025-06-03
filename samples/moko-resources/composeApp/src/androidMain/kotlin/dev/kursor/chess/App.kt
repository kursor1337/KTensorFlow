package dev.kursor.chess

import android.app.Application
import dev.kursor.chess.core.SharedApplication
import dev.kursor.chess.core.SharedApplicationProvider
import dev.kursor.chess.core.configuration.BuildType
import dev.kursor.chess.core.configuration.Configuration
import dev.kursor.chess.core.configuration.Platform

class App : Application(), SharedApplicationProvider {

    override lateinit var sharedApplication: SharedApplication
        private set

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