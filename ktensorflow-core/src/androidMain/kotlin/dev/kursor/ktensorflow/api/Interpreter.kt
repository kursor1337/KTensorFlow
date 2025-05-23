package dev.kursor.ktensorflow.api

import dev.kursor.ktensorflow.impl.AndroidInterpreter

actual fun Interpreter(
    filePath: String,
    options: InterpreterOptions
): Interpreter {
    return AndroidInterpreter(
        filePath = filePath,
        options = options
    )
}