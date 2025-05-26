package dev.kursor.chess.core.utils

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.async
import kotlinx.coroutines.runBlocking
import kotlin.reflect.KProperty

class AsyncDelegate<T : Any>(
    scope: CoroutineScope,
    initializer: suspend () -> T
) {
    private val initializationDeferred: Deferred<T> = scope.async {
        initializer()
    }

    operator fun getValue(thisRef: Any?, property: KProperty<*>): T {
        return runBlocking {
            initializationDeferred.await()
        }
    }
}

fun <T : Any> CoroutineScope.async(initializer: suspend () -> T) = AsyncDelegate(this, initializer)
