package dev.kursor.ktensorflow.compose

import dev.kursor.ktensorflow.ExperimentalKTensorFlowApi
import dev.kursor.ktensorflow.ModelDesc

@ExperimentalKTensorFlowApi
actual fun ModelDesc.Companion.ComposeUri(uri: String): ModelDesc {
    return ModelDesc.PathInBundle(uri.removePrefix("file://"))
}