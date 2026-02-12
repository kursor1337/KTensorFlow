package dev.kursor.media.core.configuration

import android.app.Application
import dev.kursor.media.core.configuration.PlatformType

actual class Platform(
    val application: Application,
) {
    actual val type = PlatformType.Android
}