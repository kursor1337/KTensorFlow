package dev.kursor.ktensorflow.pipeline.stage

// Helper to create an identity stage (no-op transformation)
fun <Input> Stage(): Stage<Input, Input> = Stage { it }