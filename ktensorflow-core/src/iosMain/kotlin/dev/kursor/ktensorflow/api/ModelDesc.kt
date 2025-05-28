package dev.kursor.ktensorflow.api

/**
 * Model description options on iOS.
 */
actual sealed interface ModelDesc {

    /**
     * Model description from a file in bundle.
     * @param pathInBundle Path to the model file in bundle. Needs to be without the "file://" prefix.
     */
    data class PathInBundle(val pathInBundle: String) : ModelDesc
}
