package dev.kursor.ktensorflow.api

import dev.kursor.ktensorflow.impl.IosInterpreter

actual fun Interpreter(
    filePath: String,
    options: InterpreterOptions
): Interpreter {
    return IosInterpreter(
        filePath = filePath,
        options = options
    )
}