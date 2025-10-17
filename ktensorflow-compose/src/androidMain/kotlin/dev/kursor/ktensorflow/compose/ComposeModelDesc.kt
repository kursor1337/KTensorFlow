package dev.kursor.ktensorflow.compose

import dev.kursor.ktensorflow.ModelDesc
import java.io.FileInputStream
import java.nio.channels.FileChannel

actual fun ModelDesc.Companion.ComposeUri(uri: String): ModelDesc {
    val fileDescriptor = appContext
        .assets
        .openFd(
            uri
                .removePrefix("file:///android_asset/")
        )
    val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
    val fileChannel = inputStream.channel
    val byteBuffer = fileChannel.map(
        FileChannel.MapMode.READ_ONLY,
        fileDescriptor.startOffset,
        fileDescriptor.declaredLength
    )
    return ModelDesc.ByteBuffer(byteBuffer)
}