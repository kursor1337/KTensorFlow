package dev.kursor.chess.core.configuration

enum class PlatformType {
    Android,
    Ios,
    Desktop,
    Web
}

expect class Platform {
    val type: PlatformType
}