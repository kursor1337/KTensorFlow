package dev.kursor.ktensorflow.moko

import android.content.Context
import androidx.startup.Initializer

internal lateinit var appContext: Context

/**
 * Initializer component to pass a context to the no-arg `Settings()` call on Android. Pass a `Context` to  `create()`
 * if you're doing custom initialization logic or need to call `Settings()` from tests.
 */
public class ContextInitializer : Initializer<Context> {
    /**
     * Stores the `applicationContext` from the provided `context` so that it can be used by the no-arg `Settings()`
     * function.
     */
    override fun create(context: Context): Context {
        return context.applicationContext.also { appContext = it }
    }

    /**
     * Returns an empty list. This initializer depends only on the provided context.
     */
    override fun dependencies(): List<Class<out Initializer<*>>> = emptyList()
}