package dev.kursor.ktensorflow.coroutines

import dev.kursor.ktensorflow.ExperimentalKTensorFlowApi
import dev.kursor.ktensorflow.pipeline.Pipeline
import dev.kursor.ktensorflow.pipeline.Tuple
import dev.kursor.ktensorflow.pipeline.tuple
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.buffer
import kotlinx.coroutines.flow.channelFlow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.withContext
import kotlin.jvm.JvmName

/**
 * Runs the pipeline with the given input off the calling thread.
 *
 * @param input The input to the pipeline.
 * @param dispatcher Dispatcher inference runs on. By default it is one shared
 *   [InferenceDispatcher], which runs one inference at a time for the whole process. Pass an
 *   [InferenceDispatcher] per interpreter to run several models in parallel.
 * @return The output of the pipeline.
 */
@ExperimentalKTensorFlowApi
suspend fun <I, O> Pipeline<I, O>.runSuspend(
    input: I,
    dispatcher: CoroutineDispatcher = SharedInferenceDispatcher
): O = withContext(dispatcher) {
    run(input)
}

/**
 * Runs the pipeline with the given input flow.
 * This function runs the pipeline for every item in the input flow.
 * The pipeline is run asynchronously on the specified [dispatcher]. By default it is the shared
 * inference dispatcher, where concurrent inference queues instead of blocking threads; any
 * other dispatcher is also safe, because the interpreter serializes its own calls.
 *
 * @param inputFlow The input flow to the pipeline.
 * @param dispatcher Dispatcher inference runs on. By default it is one shared
 *   [InferenceDispatcher], which runs one inference at a time for the whole process. Pass an
 *   [InferenceDispatcher] per interpreter to run several models in parallel.
 * @return The output flow of the pipeline.
 */
@ExperimentalKTensorFlowApi
fun <I, O> Pipeline<I, O>.processFlow(
    inputFlow: Flow<I>,
    dispatcher: CoroutineDispatcher = SharedInferenceDispatcher
): Flow<O> = inputFlow
    .map { item -> run(item) }
    .flowOn(dispatcher)

/**
 * Runs a pipeline built with the pipeline builder (which accepts a single-element [Tuple.One])
 * over the given input flow.
 *
 * Each item is wrapped into [Tuple.One] before being passed to the pipeline, so the flow can be
 * collected directly from a camera or any other source without manual wrapping.
 * The pipeline is run asynchronously on the specified [dispatcher]. By default it is the shared
 * inference dispatcher, where concurrent inference queues instead of blocking threads; any
 * other dispatcher is also safe, because the interpreter serializes its own calls.
 *
 * @param inputFlow The input flow to the pipeline.
 * @param dispatcher Dispatcher inference runs on. By default it is one shared
 *   [InferenceDispatcher], which runs one inference at a time for the whole process. Pass an
 *   [InferenceDispatcher] per interpreter to run several models in parallel.
 * @return The output flow of the pipeline.
 */
@ExperimentalKTensorFlowApi
@JvmName("processTupleFlow")
fun <I, O> Pipeline<Tuple.One<I>, O>.processFlow(
    inputFlow: Flow<I>,
    dispatcher: CoroutineDispatcher = SharedInferenceDispatcher
): Flow<O> = inputFlow
    .map { item -> run(tuple(item)) }
    .flowOn(dispatcher)

/**
 * Runs the pipeline with the given input flow, dropping items if the pipeline is already running.
 * This function runs the pipeline for every item in the input flow that arrives while it is idle.
 * The pipeline is run asynchronously on [dispatcher], one item at a time.
 *
 * The flow takes ownership of every item it receives and closes each one exactly once (using
 * [AutoCloseable.close]): an item that arrives while the pipeline is busy is closed right away,
 * and an item that is processed is closed once the pipeline has finished with it - whether it
 * succeeded, failed or the collection was cancelled. The pipeline therefore must not keep a
 * reference to its input after it returns; derive everything the output needs inside the pipeline.
 *
 * Items that never reach this flow stay with the upstream: if it buffers frames, the ones still
 * in its buffer when collection stops are never delivered here and are not closed. Close them in
 * the upstream, for example with a `Channel(capacity, onUndeliveredElement = { it.close() })`
 * consumed through `receiveAsFlow()`.
 *
 * @param inputFlow The input flow to the pipeline.
 * @param dispatcher Dispatcher inference runs on. By default it is one shared
 *   [InferenceDispatcher], which runs one inference at a time for the whole process. Pass an
 *   [InferenceDispatcher] per interpreter to run several models in parallel.
 * @return The output flow of the pipeline.
 */
@ExperimentalKTensorFlowApi
fun <I : AutoCloseable, O> Pipeline<I, O>.processFlowDropping(
    inputFlow: Flow<I>,
    dispatcher: CoroutineDispatcher = SharedInferenceDispatcher
): Flow<O> = processDropping(inputFlow, dispatcher) { item -> item.close() }

/**
 * Runs a pipeline built with the pipeline builder (which accepts a single-element [Tuple.One])
 * over the given input flow, dropping items if the pipeline is already running.
 *
 * Each item is wrapped into [Tuple.One] before being passed to the pipeline, so the flow can be
 * collected directly from a camera or any other source without manual wrapping.
 * The pipeline is run asynchronously on [dispatcher], one item at a time.
 *
 * The flow takes ownership of every item it receives and closes each one exactly once (using
 * [AutoCloseable.close]): an item that arrives while the pipeline is busy is closed right away,
 * and an item that is processed is closed once the pipeline has finished with it - whether it
 * succeeded, failed or the collection was cancelled. The pipeline therefore must not keep a
 * reference to its input after it returns; derive everything the output needs inside the pipeline.
 *
 * Items that never reach this flow stay with the upstream: if it buffers frames, the ones still
 * in its buffer when collection stops are never delivered here and are not closed. Close them in
 * the upstream, for example with a `Channel(capacity, onUndeliveredElement = { it.close() })`
 * consumed through `receiveAsFlow()`.
 *
 * @param inputFlow The input flow to the pipeline.
 * @param dispatcher Dispatcher inference runs on. By default it is one shared
 *   [InferenceDispatcher], which runs one inference at a time for the whole process. Pass an
 *   [InferenceDispatcher] per interpreter to run several models in parallel.
 * @return The output flow of the pipeline.
 */
@ExperimentalKTensorFlowApi
@JvmName("processTupleFlowDropping")
fun <I : AutoCloseable, O> Pipeline<Tuple.One<I>, O>.processFlowDropping(
    inputFlow: Flow<I>,
    dispatcher: CoroutineDispatcher = SharedInferenceDispatcher
): Flow<O> = processDropping(inputFlow.map { item -> tuple(item) }, dispatcher) { input -> input.first.close() }

@ExperimentalKTensorFlowApi
private fun <I, O> Pipeline<I, O>.processDropping(
    inputFlow: Flow<I>,
    dispatcher: CoroutineDispatcher,
    release: (I) -> Unit
): Flow<O> = channelFlow {
    val mutex = Mutex()

    inputFlow.collect { item ->
        if (!mutex.tryLock()) {
            release(item)
            return@collect
        }

        // ATOMIC: тело стартует, даже если сбор отменили раньше, чем диспетчер взял задачу.
        // С обычным запуском отменённая до старта корутина не выполняет ни строчки, и принятый
        // кадр не попадал ни в пайплайн, ни в release - он просто утекал.
        launch(dispatcher, start = CoroutineStart.ATOMIC) {
            try {
                if (isActive) send(run(item))
            } finally {
                // Кадром владеет поток: он закрывается после пайплайна при любом исходе
                release(item)
                mutex.unlock()
            }
        }
    }
}.buffer(Channel.RENDEZVOUS)