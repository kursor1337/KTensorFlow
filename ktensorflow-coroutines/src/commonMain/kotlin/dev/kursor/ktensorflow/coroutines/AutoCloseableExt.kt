package dev.kursor.ktensorflow.coroutines

import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

/**
 * Applies the given transformation function to each item of the flow and automatically closes each item.
 *
 * This function is similar to [kotlinx.coroutines.flow.Flow.map], but it also automatically closes each item after the transformation is applied.
 *
 * @param transform The transformation function to apply to each item.
 * @return A new flow that applies the transformation function to each item and automatically closes each item.
 */
inline fun <T : AutoCloseable, R> Flow<T>.mapAndClose(
    crossinline transform: suspend (T) -> R
): Flow<R> = map { item ->
    try {
        transform(item)
    } finally {
        item.close()
    }
}