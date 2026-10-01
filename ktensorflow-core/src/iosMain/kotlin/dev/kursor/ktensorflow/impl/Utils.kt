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

/**
 * Вызывает [block] и приводит NSError к [TensorFlowException].
 *
 * @param explain понятное сообщение для известного отказа или null; исходная ошибка платформы
 * тогда остаётся в cause, как на Android.
 */
@OptIn(BetaInteropApi::class)
internal fun <T : Any> checkError(
    explain: (NSError) -> String? = { null },
    block: (CPointer<ObjCObjectVar<NSError?>>) -> T?
): T {
    val (result, error) = callWithError(block)

    if (error != null) {
        val platformError = TensorFlowException(error.description, error.code.toInt(), null)
        throw explain(error)?.let { TensorFlowException(it, platformError) } ?: platformError
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

/**
 * Вызывает [block] с NSData, которая смотрит прямо в закреплённый массив, без копии. Годится
 * только на время вызова: после него массив открепляется. Раньше NSData.create копировала
 * весь вход, а copyData копировал его ещё раз.
 */
@OptIn(BetaInteropApi::class)
internal inline fun <T> ByteArray.withNSDataView(block: (NSData) -> T): T =
    if (isEmpty()) {
        // У пустого массива нет адреса для закрепления
        block(NSData())
    } else {
        usePinned { pinned ->
            block(NSData.create(bytesNoCopy = pinned.addressOf(0), length = size.convert(), freeWhenDone = false))
        }
    }

/** Копирует содержимое NSData в начало [destination] одним memcpy, без промежуточного массива. */
internal fun NSData.copyInto(destination: ByteArray) {
    if (length == 0uL) return
    destination.usePinned { pinned ->
        memcpy(pinned.addressOf(0), bytes, length)
    }
}
