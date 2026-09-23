package dev.kursor.ktensorflow.compose

import dev.kursor.ktensorflow.ModelDesc
import dev.kursor.ktensorflow.TensorFlowException
import java.io.FileInputStream
import java.io.IOException
import java.nio.channels.FileChannel

actual fun ModelDesc.Companion.ComposeUri(uri: String): ModelDesc {
    val assetPath = uri.removePrefix("file:///android_asset/")

    // Отказ приводится к TensorFlowException: на iOS отсутствующая модель приходит наружу
    // именно им (уже из Interpreter), и общий код обязан ловить одно и то же на обеих
    // платформах, а не FileNotFoundException здесь и TensorFlowException там.
    val byteBuffer = try {
        // Дескриптор ассета и поток закрываются сразу: отображённый буфер остаётся валидным
        // и после закрытия канала, а без этого каждая загрузка модели утекала двумя файловыми
        // дескрипторами - в приложении, которое переоткрывает модель, это копится до отказа.
        appContext.assets.openFd(assetPath).use { descriptor ->
            FileInputStream(descriptor.fileDescriptor).use { stream ->
                stream.channel.map(
                    FileChannel.MapMode.READ_ONLY,
                    descriptor.startOffset,
                    descriptor.declaredLength
                )
            }
        }
    } catch (e: IOException) {
        throw TensorFlowException("Failed to open the model asset '$assetPath'", e)
    }

    return ModelDesc.ByteBuffer(byteBuffer)
}
