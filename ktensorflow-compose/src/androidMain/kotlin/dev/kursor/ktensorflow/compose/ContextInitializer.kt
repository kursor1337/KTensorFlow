package dev.kursor.ktensorflow.compose

import android.content.Context
import androidx.startup.Initializer

private var storedContext: Context? = null

/**
 * Application context this module loads models with.
 *
 * It is supplied by [ContextInitializer] through `androidx.startup` before the first line of
 * app code runs, so nothing has to be passed manually in a normal app.
 */
internal val appContext: Context
    get() = checkNotNull(storedContext) {
        "KTensorFlow has no Context yet. It is normally provided automatically by " +
            "androidx.startup through ContextInitializer; if the startup provider is " +
            "disabled or removed in the app, call ContextInitializer().create(context) " +
            "once before loading a model."
    }

/**
 * Captures the application context so that models can be loaded from resources without the
 * caller having to pass a `Context` around.
 *
 * It is registered in this module's manifest and runs automatically via `androidx.startup`.
 * Call it by hand only when the app removes the startup provider or initializes on its own.
 */
public class ContextInitializer : Initializer<Context> {

    /** Stores the application context of [context] and returns it. */
    override fun create(context: Context): Context =
        context.applicationContext.also { storedContext = it }

    /** Returns an empty list: this initializer depends only on the provided context. */
    override fun dependencies(): List<Class<out Initializer<*>>> = emptyList()
}
