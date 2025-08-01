package dev.kursor.ktensorflow

import dev.kursor.ktensorflow.impl.AndroidInterpreter

actual fun Interpreter(
    modelDesc: ModelDesc,
    options: InterpreterOptions
): Interpreter {
    return AndroidInterpreter(
        modelDesc = modelDesc,
        options = options
    )
}
