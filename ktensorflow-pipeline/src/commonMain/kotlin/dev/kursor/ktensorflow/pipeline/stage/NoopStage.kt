package dev.kursor.ktensorflow.pipeline.stage

import dev.kursor.ktensorflow.ExperimentalKTensorFlowApi

// Helper to create an identity stage (no-op transformation)
@ExperimentalKTensorFlowApi
fun <Input> Stage(): Stage<Input, Input> = Stage { it }