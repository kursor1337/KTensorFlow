package dev.kursor.chess.core.configuration

import android.app.Application

actual class Platform(
    val application: Application,
) {
    actual val type = PlatformType.Android
}