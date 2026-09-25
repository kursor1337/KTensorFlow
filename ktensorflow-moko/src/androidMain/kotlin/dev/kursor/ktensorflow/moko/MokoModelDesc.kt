package dev.kursor.ktensorflow.moko

import android.content.Context
import dev.icerock.moko.resources.AssetResource
import dev.icerock.moko.resources.FileResource
import dev.kursor.ktensorflow.ModelDesc
import dev.kursor.ktensorflow.TensorFlowException
import java.io.File
import java.io.IOException
import java.io.InputStream

actual fun ModelDesc.Companion.FileResource(resource: FileResource): ModelDesc =
    ModelDesc.File(
        copyToCache(appContext, "raw_${resource.rawResId}") {
            appContext.resources.openRawResource(resource.rawResId)
        }
    )

actual fun ModelDesc.Companion.AssetResource(resource: AssetResource): ModelDesc =
    ModelDesc.File(
        copyToCache(appContext, resource.path.replace('/', '_')) {
            appContext.assets.open(resource.path)
        }
    )

/**
 * Копирует ресурс в кеш под предсказуемым именем и возвращает получившийся файл.
 *
 * TensorFlow Lite на Android умеет читать модель только из файла или ByteBuffer, поэтому
 * ресурс приходится выкладывать на диск. Имя детерминированное: раньше здесь создавался
 * File.createTempFile("prefix", "suffix"), и каждая загрузка модели оставляла в кеше новый
 * файл размером с модель, который никто не удалял.
 *
 * Новое содержимое пишется во временный файл рядом и атомарно подменяет старое переименованием.
 * Перезаписывать файл на месте нельзя: TensorFlow Lite отображает модель в память (mmap), и
 * обрезка файла под уже работающим интерпретатором роняла процесс с SIGBUS - например, когда
 * модель из того же ресурса загружалась второй раз, пока первая ещё считала. После
 * переименования старый интерпретатор продолжает читать прежний файл, пока не закроется.
 */
private inline fun copyToCache(context: Context, name: String, open: () -> InputStream): File {
    val directory = File(context.cacheDir, "ktensorflow-models").apply { mkdirs() }
    val file = File(directory, name)

    // Отказ приводится к TensorFlowException, чтобы отсутствующий или нечитаемый ресурс
    // ловился в общем коде так же, как на iOS
    val temporary = try {
        File.createTempFile("$name.", ".tmp", directory).also { temporary ->
            try {
                open().use { input -> temporary.outputStream().use(input::copyTo) }
            } catch (e: IOException) {
                temporary.delete()
                throw e
            }
        }
    } catch (e: IOException) {
        throw TensorFlowException("Failed to read the model resource '$name'", e)
    }

    if (!temporary.renameTo(file)) {
        temporary.delete()
        throw TensorFlowException("Failed to store the model resource '$name' in the cache")
    }
    return file
}
