package dev.kursor.ktensorflow.coroutines

import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers

/**
 * Dispatcher every inference in this module runs on.
 *
 * `Interpreter.run` is not thread-safe: the underlying native interpreter must not be entered
 * from two threads at once. Sequential calls are fine even from different threads, because
 * `withContext` establishes happens-before, but plain [Dispatchers.Default] is a thread pool,
 * so two concurrent `runSuspend` calls on the same interpreter would enter native code in
 * parallel and corrupt its state.
 *
 * Limiting parallelism to one makes that impossible: inference calls queue instead of racing.
 * The work itself still runs off the caller's thread, and TensorFlow Lite keeps parallelising
 * internally through its own `numThreads` option, so a single inference still uses several cores.
 *
 * The trade-off is that inference is serialized process-wide, so two different models cannot be
 * inferred at the same time. That is the safe default; direct calls to `Interpreter.run` outside
 * this module are of course not covered by it.
 */
internal val InferenceDispatcher: CoroutineDispatcher = Dispatchers.Default.limitedParallelism(1)
