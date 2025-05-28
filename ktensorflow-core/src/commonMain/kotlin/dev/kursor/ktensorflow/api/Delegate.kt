package dev.kursor.ktensorflow.api

/**
 * Provides hardware acceleration for inference.
 */
expect interface Delegate {

    /**
     * Specifies whether the delegate is available on the current device.
     */
    val isAvailable: Boolean
}