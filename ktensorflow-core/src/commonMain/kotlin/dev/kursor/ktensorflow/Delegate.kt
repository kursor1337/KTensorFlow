package dev.kursor.ktensorflow

/**
 * Provides hardware acceleration for inference.
 *
 * A delegate owns native resources (a GPU context, an NNAPI or CoreML model) that are released by
 * [close]. Close it only after every [Interpreter] created with it has been closed: an interpreter
 * keeps using the delegate for as long as it is open. Closing a delegate twice is safe, and once
 * it is closed it can no longer be passed to a new interpreter.
 */
expect interface Delegate : AutoCloseable {

    /**
     * Specifies whether the delegate is available on the current device.
     */
    val isAvailable: Boolean

    /**
     * Releases the native resources of the delegate.
     */
    override fun close()
}
