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
    val (result, error) = memScoped {
        val errorPtr = alloc<ObjCObjectVar<NSError?>>()
        block(errorPtr.ptr) to errorPtr.value
    }

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
    val (result, error) = memScoped {
        val errorPtr = alloc<ObjCObjectVar<NSError?>>()
        block(errorPtr.ptr) to errorPtr.value
    }
    if (error != null) {
        throw TensorFlowException(error.description)
    }
    return result
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
