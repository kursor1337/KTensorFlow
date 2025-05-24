package dev.kursor.ktensorflow.moko

import dev.icerock.moko.resources.AssetResource
import dev.kursor.ktensorflow.api.Interpreter
import dev.kursor.ktensorflow.api.InterpreterOptions

fun Interpreter(
    assetResource: AssetResource,
    options: InterpreterOptions
) = Interpreter(
    filePath = assetResource.originalPath,
    options = options
)