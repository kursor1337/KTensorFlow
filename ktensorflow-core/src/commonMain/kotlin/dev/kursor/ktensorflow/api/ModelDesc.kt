package dev.kursor.ktensorflow.api

/**
 * Model description class
 * It is used to load model from different sources
 * Avaliable sources vary depending on the platform:
 * Android: ModelDesc.File, ModelDesc.ByteBuffer
 * iOS: ModelDesc.PathInBundle
 */
expect sealed interface ModelDesc {
    companion object
}
