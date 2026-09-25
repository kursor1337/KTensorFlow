package dev.kursor.ktensorflow.coroutines

import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers

/**
 * Dispatcher every inference in this module runs on.
 *
 * Thread safety does not depend on it: every `Interpreter` serializes its own calls with a lock,
 * so concurrent `run` calls cannot corrupt the native interpreter. What this dispatcher adds is
 * how the waiting happens. On plain [Dispatchers.Default], concurrent `runSuspend` calls on one
 * interpreter would each take a pool thread and block it on that lock, so a burst of calls could
 * park every Default thread and starve the rest of the app. Limiting parallelism to one makes
 * them queue as suspended coroutines that hold no thread.
 *
 * The work still runs off the caller's thread, and TensorFlow Lite parallelises internally
 * through its own `numThreads` option, so a single inference still uses several cores. Running
 * two models at once would mostly compete for the same cores.
 *
 * The trade-off is that inference through this module is serialized process-wide, so two
 * different models are not inferred at the same time. To run them in parallel, call
 * `Interpreter.run` from a dispatcher of your own: that is safe.
 */
internal val InferenceDispatcher: CoroutineDispatcher = Dispatchers.Default.limitedParallelism(1)
