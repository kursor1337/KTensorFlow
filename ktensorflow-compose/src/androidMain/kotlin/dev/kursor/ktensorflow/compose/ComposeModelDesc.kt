package dev.kursor.ktensorflow.compose

import dev.kursor.ktensorflow.ModelDesc
import dev.kursor.ktensorflow.TensorFlowException
import java.io.FileInputStream
import java.io.IOException
import java.io.InputStream
import java.net.URL
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

actual fun ModelDesc.Companion.ComposeUri(uri: String): ModelDesc {
    // Ресурс не из assets compose-resources отдаёт как java-ресурс APK: jar:file:/…/base.apk!/…
    // Раньше такой URI уходил в AssetManager как путь и не находился
    if (uri.startsWith("jar:")) {
        return ModelDesc.ByteBuffer(readModel(uri) { URL(uri).openStream() })
    }

    val assetPath = uri.removePrefix("file:///android_asset/")

    // Отказ приводится к TensorFlowException: на iOS отсутствующая модель приходит наружу
    // именно им (уже из Interpreter), и общий код обязан ловить одно и то же на обеих
    // платформах, а не FileNotFoundException здесь и TensorFlowException там.
    val byteBuffer = try {
        mapAsset(assetPath)
    } catch (_: IOException) {
        // openFd работает только с несжатыми ассетами. Сжатый (любое расширение, кроме тех,
        // что AGP не сжимает, например .tflite) читается потоком; отсутствующий ассет
        // упадёт и здесь - уже с TensorFlowException
        readModel(assetPath) { appContext.assets.open(assetPath) }
    }

    return ModelDesc.ByteBuffer(byteBuffer)
}

/** Отображает несжатый ассет в память без копирования. */
private fun mapAsset(assetPath: String): ByteBuffer =
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

/** Читает модель из потока в direct-буфер: TensorFlow Lite не принимает обычный ByteBuffer. */
private inline fun readModel(name: String, open: () -> InputStream): ByteBuffer {
    val bytes = try {
        open().use { it.readBytes() }
    } catch (e: IOException) {
        throw TensorFlowException("Failed to open the model asset '$name'", e)
    }
    return ByteBuffer.allocateDirect(bytes.size)
        .order(ByteOrder.nativeOrder())
        .put(bytes)
        .apply { rewind() }
}
