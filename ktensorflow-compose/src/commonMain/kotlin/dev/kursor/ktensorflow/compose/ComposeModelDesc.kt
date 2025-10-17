package dev.kursor.ktensorflow.compose

import dev.kursor.ktensorflow.ExperimentalKTensorFlowApi
import dev.kursor.ktensorflow.ModelDesc

/**
 * Creates a new [ModelDesc] from a Compose Resources URI.
 * Use it like this:
 *
 * ```
 * val modelDesc = ModelDesc.ComposeUri(Res.getUri(<path-to-model>))
 * ```
 */
@ExperimentalKTensorFlowApi
expect fun ModelDesc.Companion.ComposeUri(uri: String): ModelDesc
