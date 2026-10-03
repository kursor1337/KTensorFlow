package dev.kursor.ktensorflow.compose

import dev.kursor.ktensorflow.ExperimentalKTensorFlowApi
import dev.kursor.ktensorflow.ModelDesc
import platform.Foundation.NSURL

@ExperimentalKTensorFlowApi
actual fun ModelDesc.Companion.ComposeUri(uri: String): ModelDesc {
    // Res.getUri на iOS - это NSURL.fileURLWithPath(path).toString(), то есть путь с
    // percent-кодированием: пробел превращается в %20, кириллица - в %D0%9C... Просто снять
    // "file://" было мало, и модель в пути с такими символами не загружалась вовсе
    val path = if (uri.startsWith("file:")) NSURL.URLWithString(uri)?.path else null
    return ModelDesc.PathInBundle(path ?: uri)
}
