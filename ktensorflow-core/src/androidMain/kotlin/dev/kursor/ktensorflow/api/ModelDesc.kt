package dev.kursor.ktensorflow.api

import java.io.File as JavaFile
import java.nio.ByteBuffer as JavaByteBuffer

actual sealed interface ModelDesc {

    data class File(val file: JavaFile) : ModelDesc

    data class ByteBuffer(val buffer: JavaByteBuffer) : ModelDesc
}