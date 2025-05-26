package dev.kursor.ktensorflow.api

actual sealed interface ModelDesc {

    data class PathInBundle(val pathInBundle: String) : ModelDesc
}
