package dev.kursor.ktensorflow.coroutines

import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Runnable
import kotlin.coroutines.CoroutineContext

/**
 * Dispatcher for running inference: it runs one task at a time and, on Android, always on the
 * same thread of its own.
 *
 * Every function of this module takes a dispatcher and by default uses one shared instance, so
 * inference through the module runs one call at a time across the whole app. Create a dispatcher
 * per interpreter to run several models in parallel: pass it to [SuspendInterpreter] and to every
 * run of that interpreter. Keeping one dispatcher per interpreter also satisfies the GPU delegate
 * on Android, whose OpenGL backend (used on devices without OpenCL) must run inference on the
 * thread that created the interpreter. On iOS delegates are not bound to a thread, so there it
 * only runs tasks one at a time on the default pool.
 *
 * Concurrent calls on one interpreter are safe either way: the interpreter serializes them with
 * its own lock. The dispatcher decides how the waiting happens: calls queue here as suspended
 * coroutines that hold no thread, instead of each blocking a pool thread on that lock.
 *
 * On Android the dispatcher owns a thread. Keep it for as long as its interpreter lives and
 * [close] it afterwards; it must not be used once closed. On iOS [close] does nothing.
 */
class InferenceDispatcher : CoroutineDispatcher(), AutoCloseable {

    private val platform = platformInferenceDispatcher()

    override fun isDispatchNeeded(context: CoroutineContext): Boolean =
        platform.dispatcher.isDispatchNeeded(context)

    override fun dispatch(context: CoroutineContext, block: Runnable) =
        platform.dispatcher.dispatch(context, block)

    /** Stops the thread of this dispatcher on Android; does nothing on iOS. */
    override fun close() = platform.close()

    override fun toString(): String = "InferenceDispatcher"
}

/** Платформенная часть [InferenceDispatcher]: сам диспетчер и способ освободить его ресурсы. */
internal class PlatformInferenceDispatcher(
    val dispatcher: CoroutineDispatcher,
    val close: () -> Unit
)

internal expect fun platformInferenceDispatcher(): PlatformInferenceDispatcher

/**
 * Общий диспетчер по умолчанию для всех функций модуля: инференс через модуль идёт по одному
 * вызову на всё приложение. Живёт всё время процесса и не закрывается.
 */
internal val SharedInferenceDispatcher = InferenceDispatcher()
