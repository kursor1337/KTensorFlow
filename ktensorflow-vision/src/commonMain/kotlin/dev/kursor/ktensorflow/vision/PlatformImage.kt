package dev.kursor.ktensorflow.vision

/**
 * Represents a platform-specific image object used for vision processing.
 *
 * This expect class provides a common abstraction for image data across different platforms
 * (e.g., `Bitmap` on Android or `ByteArray` on iOS).
 */
expect class PlatformImage