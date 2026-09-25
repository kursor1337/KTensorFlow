package dev.kursor.ktensorflow

/**
 * Error raised by the underlying TensorFlow Lite runtime.
 *
 * Every failure coming out of the runtime is reported as this type on both platforms - an
 * `NSError` on iOS and a TensorFlow Lite Java exception on Android - so loading a model or
 * running inference can be guarded from common code:
 *
 * ```
 * val interpreter = try {
 *     Interpreter(modelDesc)
 * } catch (e: TensorFlowException) {
 *     // corrupt model, unsupported ops, missing delegate, ...
 * }
 * ```
 *
 * The original platform exception, when there is one, is kept as [cause].
 */
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
