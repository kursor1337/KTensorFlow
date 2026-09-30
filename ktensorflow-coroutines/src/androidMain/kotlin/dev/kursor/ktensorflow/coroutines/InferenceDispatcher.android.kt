package dev.kursor.ktensorflow.coroutines

import kotlinx.coroutines.asCoroutineDispatcher
import java.util.concurrent.Executors

// Свой поток, а не Dispatchers.Default.limitedParallelism(1): тот выполняет задачи по одной, но
// на любом потоке пула, а OpenGL-бэкенд GPU-делегата требует инференса на том потоке, где создан
// интерпретатор. Executor вместо newSingleThreadContext - стабильный API без DelicateCoroutinesApi
// и ExperimentalCoroutinesApi
internal actual fun platformInferenceDispatcher(): PlatformInferenceDispatcher {
    val dispatcher = Executors
        .newSingleThreadExecutor { task -> Thread(task, "KTensorFlow inference").apply { isDaemon = true } }
        .asCoroutineDispatcher()
    return PlatformInferenceDispatcher(dispatcher) { dispatcher.close() }
}
