package dev.kursor.ktensorflow.impl

import dev.kursor.ktensorflow.TensorFlowException
import kotlinx.cinterop.BetaInteropApi
import kotlinx.cinterop.CPointer
import kotlinx.cinterop.ObjCObjectVar
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.alloc
import kotlinx.cinterop.convert
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.ptr
import kotlinx.cinterop.usePinned
import kotlinx.cinterop.value
import platform.Foundation.NSData
import platform.Foundation.NSError
import platform.Foundation.create
import platform.posix.memcpy

@OptIn(BetaInteropApi::class)
internal fun <T : Any> checkError(block: (CPointer<ObjCObjectVar<NSError?>>) -> T?): T {
    val (result, error) = callWithError(block)

    if (error != null) {
        throw TensorFlowException(error.description, error.code.toInt(), null)
    }
    if (result == null) {
        throw TensorFlowException("Result is null")
    }
    return result
}

@OptIn(BetaInteropApi::class)
internal fun <T : Any> checkErrorNullable(block: (CPointer<ObjCObjectVar<NSError?>>) -> T?): T? {
    val (result, error) = callWithError(block)

    if (error != null) {
        throw TensorFlowException(error.description)
    }
    return result
}

/**
 * Вызывает [block] с указателем на NSError и возвращает результат вместе с ошибкой.
 *
 * Часть Obj-C инициализаторов объявлена в биндинге как возвращающая non-null, но при отказе
 * они отдают nil, и Kotlin/Native падает с NullPointerException ещё до того, как мы успеем
 * посмотреть на NSError. Наружу такой отказ обязан приходить как [TensorFlowException],
 * поэтому NPE здесь превращается в отсутствующий результат: ниже его разберёт NSError,
 * а если ошибки нет - общее сообщение про null.
 */
@OptIn(BetaInteropApi::class)
private fun <T : Any> callWithError(
    block: (CPointer<ObjCObjectVar<NSError?>>) -> T?
): Pair<T?, NSError?> = memScoped {
    val errorPtr = alloc<ObjCObjectVar<NSError?>>()
    val result = try {
        block(errorPtr.ptr)
    } catch (_: NullPointerException) {
        null
    }
    result to errorPtr.value
}

@OptIn(BetaInteropApi::class)
internal fun ByteArray.toNSData(): NSData = usePinned {
    NSData.create(bytes = it.addressOf(0), length = this@toNSData.size.convert())
}

internal fun NSData.toByteArray(): ByteArray = ByteArray(this@toByteArray.length.toInt()).apply {
    usePinned {
        memcpy(it.addressOf(0), this@toByteArray.bytes, this@toByteArray.length)
    }
}
