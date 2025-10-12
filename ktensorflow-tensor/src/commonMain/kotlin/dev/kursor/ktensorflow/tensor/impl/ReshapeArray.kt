package dev.kursor.ktensorflow.tensor.impl

@OptIn(ExperimentalUnsignedTypes::class)
internal expect fun reshapeArray(flat: Any, dimentions: IntArray): Any