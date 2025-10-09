package dev.kursor.ktensorflow.pipeline.stage

import dev.kursor.ktensorflow.ExperimentalKTensorFlowApi

/**
 * A no-op [Stage] that returns the input unchanged.
 */
@ExperimentalKTensorFlowApi
fun <Input> Stage(): Stage<Input, Input> = Stage { it }