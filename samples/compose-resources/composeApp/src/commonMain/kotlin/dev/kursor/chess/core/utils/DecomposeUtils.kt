package dev.kursor.chess.core.utils

import com.arkivanov.decompose.Child
import com.arkivanov.decompose.ComponentContext
import com.arkivanov.decompose.ExperimentalDecomposeApi
import com.arkivanov.decompose.extensions.compose.stack.animation.StackAnimation
import com.arkivanov.decompose.extensions.compose.stack.animation.fade
import com.arkivanov.decompose.extensions.compose.stack.animation.plus
import com.arkivanov.decompose.extensions.compose.stack.animation.predictiveback.PredictiveBackAnimatable
import com.arkivanov.decompose.extensions.compose.stack.animation.predictiveback.androidPredictiveBackAnimatable
import com.arkivanov.decompose.extensions.compose.stack.animation.predictiveback.predictiveBackAnimation
import com.arkivanov.decompose.extensions.compose.stack.animation.slide
import com.arkivanov.decompose.extensions.compose.stack.animation.stackAnimation
import com.arkivanov.decompose.router.slot.ChildSlot
import com.arkivanov.decompose.router.stack.ChildStack
import com.arkivanov.decompose.value.Value
import com.arkivanov.essenty.backhandler.BackCallback
import com.arkivanov.essenty.backhandler.BackDispatcher
import com.arkivanov.essenty.backhandler.BackEvent
import com.arkivanov.essenty.backhandler.BackHandler
import com.arkivanov.essenty.backhandler.BackHandlerOwner
import com.arkivanov.essenty.instancekeeper.getOrCreateSimple
import com.arkivanov.essenty.lifecycle.Lifecycle
import com.arkivanov.essenty.lifecycle.doOnDestroy
import com.arkivanov.essenty.statekeeper.StateKeeperOwner
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.serialization.KSerializer

/**
 * Creates a [ChildStack] with a single active component. Should be used to create a stack for Jetpack Compose preview.
 */
fun <T : Any> createFakeChildStack(instance: T): ChildStack<*, T> {
    return ChildStack(
        configuration = "<fake>",
        instance = instance
    )
}

fun <T : Any> createFakeChildStackStateFlow(instance: T): StateFlow<ChildStack<*, T>> {
    return MutableStateFlow(createFakeChildStack(instance))
}

/**
 * Creates a [ChildSlot] with given [configuration] and [instance]. Should be used to create a slot for Jetpack Compose preview.
 */
fun <C : Any, T : Any> createFakeChildSlot(configuration: C, instance: T): StateFlow<ChildSlot<C, T>> {
    return MutableStateFlow(
        ChildSlot(
            Child.Created(
                configuration = configuration,
                instance = instance
            )
        )
    )
}

/**
 * A fake [BackHandler] used only for Jetpack Compose previews.
 */
fun fakeBackHandler() = object : BackHandler {
    override fun isRegistered(callback: BackCallback): Boolean = true
    override fun register(callback: BackCallback) = Unit
    override fun unregister(callback: BackCallback) = Unit
}

/**
 * Converts [Value] from Decompose to [State] from Jetpack Compose.
 */
fun <T : Any> Value<T>.toStateFlow(lifecycle: Lifecycle): StateFlow<T> {
    val state: MutableStateFlow<T> = MutableStateFlow(this.value)

    if (lifecycle.state != Lifecycle.State.DESTROYED) {
        val observer = { value: T -> state.value = value }
        val cancellation = subscribe(observer)
        lifecycle.doOnDestroy {
            cancellation.cancel()
        }
    }

    return state
}

/**
 * Creates a coroutine scope tied to Decompose lifecycle. A scope is canceled when a component is destroyed.
 */
val ComponentContext.componentScope: CoroutineScope
    get() = instanceKeeper.getOrCreateSimple {
        lifecycle.coroutineScope()
    }

fun Lifecycle.coroutineScope(): CoroutineScope {
    val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)

    if (this.state != Lifecycle.State.DESTROYED) {
        this.doOnDestroy {
            scope.cancel()
        }
    } else {
        scope.cancel()
    }

    return scope
}


/**
 * A helper function to save and restore component state.
 */
inline fun <T : Any> StateKeeperOwner.persistent(
    key: String = "PersistentState",
    serializer: KSerializer<T>,
    noinline save: () -> T,
    restore: (T) -> Unit
) {
    stateKeeper.consume(key, serializer)?.run(restore)
    stateKeeper.register(key, serializer, save)
}

/**
 * A component that supports predictive back gesture handling.
 */
interface PredictiveBackComponent : BackHandlerOwner {

    fun onBack()
}

/**
 * A convenient wrapper around [com.arkivanov.decompose.extensions.compose.stack.animation.predictiveback.predictiveBackAnimation1]
 * for components implementing [PredictiveBackComponent].
 */
@OptIn(ExperimentalDecomposeApi::class)
fun <C : Any, T : Any> PredictiveBackComponent.predictiveBackAnimation(
    fallBackAnimation: StackAnimation<C, T> = stackAnimation(fade() + slide()),
    selector: (
        initialBackEvent: BackEvent,
        exitChild: Child.Created<C, T>,
        enterChild: Child.Created<C, T>,
    ) -> PredictiveBackAnimatable = { backEvent, _, _ ->
        androidPredictiveBackAnimatable(backEvent)
    },
): StackAnimation<C, T> {
    return predictiveBackAnimation(
        backHandler = backHandler,
        fallbackAnimation = fallBackAnimation,
        selector = selector,
        onBack = ::onBack
    )
}

val appBackDispatcher = BackDispatcher()
