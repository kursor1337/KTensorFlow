package dev.kursor.ktensorflow

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
     *
     * A direct buffer, such as a `MappedByteBuffer` of a model file, is used as is and must stay
     * unchanged while the interpreter is open. Any other buffer is copied once into native
     * memory when the interpreter is created.
     */
    data class ByteBuffer(val buffer: JavaByteBuffer) : ModelDesc

    actual companion object
}
