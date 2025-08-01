package dev.kursor.ktensorflow

/**
 * Provides hardware acceleration for inference.
 */
expect interface Delegate {

    /**
     * Specifies whether the delegate is available on the current device.
     */
    val isAvailable: Boolean
}