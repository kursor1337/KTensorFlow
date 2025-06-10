package dev.kursor.ktensorflow

import dev.kursor.ktensorflow.impl.IosInterpreter

actual fun Interpreter(
    modelDesc: ModelDesc,
    options: InterpreterOptions
): Interpreter {
    return IosInterpreter(
        modelDesc = modelDesc,
        options = options
    )
}
