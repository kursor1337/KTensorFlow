package dev.kursor.ktensorflow.gpu

import dev.kursor.ktensorflow.Delegate

/**
 * Delegate to use GPU for inference.
 *
 * On Android the delegate runs on OpenCL where the device provides it and falls back to OpenGL
 * otherwise. With OpenGL, inference must run on the thread that created the interpreter: create
 * it with `SuspendInterpreter` from `ktensorflow-coroutines` and run it through that module with
 * `runSuspend`. If you pass a dispatcher of your own, it must be single-threaded, such as an
 * `InferenceDispatcher`, and the same one must create the interpreter and run it every time.
 * Without that module, create and run the interpreter on one thread of your own. iOS uses Metal,
 * which has no such restriction.
 */
interface GpuDelegate : Delegate

/**
 * Creates [GpuDelegate] with the specified [options].
 */
expect fun GpuDelegate(
    options: GpuDelegateOptions = GpuDelegateOptions()
): Delegate
