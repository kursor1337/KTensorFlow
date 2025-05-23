package dev.kursor.ktensorflow.api

class TensorFlowException : Exception {
    constructor() : super()

    constructor(message: String?) : super(message)

    constructor(message: String?, cause: Throwable?) : super(message, cause)

    constructor(cause: Throwable?) : super(cause)

    constructor(
        message: String?,
        errorCode: Int,
        cause: Throwable?
    ) : super("$message (error code: $errorCode)", cause)
}