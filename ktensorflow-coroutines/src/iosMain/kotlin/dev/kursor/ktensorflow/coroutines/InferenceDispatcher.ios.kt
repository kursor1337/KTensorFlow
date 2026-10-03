package dev.kursor.ktensorflow.coroutines

import kotlinx.coroutines.Dispatchers

// На iOS делегаты (Metal, CoreML) к потоку не привязаны, поэтому достаточно выполнять вызовы
// по одному; отдельный поток и его закрытие здесь не нужны
internal actual fun platformInferenceDispatcher(): PlatformInferenceDispatcher =
    PlatformInferenceDispatcher(Dispatchers.Default.limitedParallelism(1)) { }
