package dev.kursor.ktensorflow

/**
 * Model description class
 * It is used to load model from different sources
 * Available sources vary depending on the platform:
 * Android: ModelDesc.File, ModelDesc.ByteBuffer
 * iOS: ModelDesc.PathInBundle
 */
expect sealed interface ModelDesc {
    companion object
}
