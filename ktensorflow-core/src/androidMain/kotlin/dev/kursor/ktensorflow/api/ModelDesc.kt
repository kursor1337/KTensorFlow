package dev.kursor.ktensorflow.api

import java.io.File as JavaFile
import java.nio.ByteBuffer as JavaByteBuffer

/**
 * Model description options on Android.
 */
actual sealed interface ModelDesc {

    /**
     * ModelDesc that allows to load model from a file.
     */
    data class File(val file: JavaFile) : ModelDesc

    /**
     * ModelDesc that allows to load model from a byte buffer.
     */
    data class ByteBuffer(val buffer: JavaByteBuffer) : ModelDesc
}
